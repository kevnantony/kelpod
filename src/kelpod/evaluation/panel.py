"""Per-plan metric panel of ``docs/EVAL_PROTOCOL.md`` §3, for the locked evaluation.

:func:`plan_panel` scores one predicted :class:`~kelpod.contracts.PlanGraph` (or a crash) against
the ground truth and returns a flat ``{metric: float | None}`` dict. ``None`` means "not defined
for this plan" (e.g. window precision when nothing was predicted and the GT has no windows); such
plans drop out of that metric's mean and of its paired bootstrap.

:func:`proxy_panel` is the label-free panel for real pages without annotations (real-dev).

Conventions (fixed here so they cannot drift between runs):

* the prediction is aligned with :func:`~kelpod.evaluation.metrics.compare_to_ground_truth`
  (translation only, best of none / min corner / max corner by wall F1);
* relative errors are capped at 1.0, so one absurd plan cannot dominate a mean;
* a crash (or a rejected result) scores 0 on every rate / IoU / F1, 1.0 on every error, and the
  edit cost of building the GT from nothing.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.ops import unary_union

from kelpod.contracts import OpeningKind, PlanGraph, WallKind
from kelpod.evaluation.metrics import (
    _match_points,
    _opening_points,
    _valid_room,
    closure_metrics,
    compare_to_ground_truth,
)

CORRECT_ROOM_IOU = 0.85
CORRECT_AREA_ERROR = 0.03
EXTERIOR_MIN_THICKNESS_M = 0.30
"""A predicted wall of unknown kind counts as exterior from this thickness on."""
NODE_MATCH_TOL_M = 0.20
OPENING_TOL_M = 0.30
WALL_TOL_M = 0.10
ERROR_CAP = 1.0

# metric name -> (direction, kind); kind decides the tolerance of promotion rule 3
#   direction +1 = higher is better, -1 = lower is better
#   kind: "rate" (abs, 1 pp), "score" (abs, IoU/F1 0.01), "error" (relative, +5 %),
#         "runtime" (relative, +25 %), "info" (reported, never gates)
METRICS: dict[str, tuple[int, str]] = {
    "correct_closed": (+1, "rate"),
    "closed": (+1, "rate"),
    "room_iou": (+1, "score"),
    "room_precision": (+1, "rate"),
    "room_recall": (+1, "rate"),
    "room_area_error": (-1, "error"),
    "room_count_error": (-1, "error"),
    "total_area_error": (-1, "error"),
    "wall_precision": (+1, "score"),
    "wall_recall": (+1, "score"),
    "wall_f1": (+1, "score"),
    "thickness_class_acc": (+1, "rate"),
    "door_precision": (+1, "rate"),
    "door_recall": (+1, "rate"),
    "window_precision": (+1, "rate"),
    "window_recall": (+1, "rate"),
    "sill_error_m": (-1, "error"),
    "edit_cost": (-1, "error"),
    "ext_wall_length_error": (-1, "error"),
    "window_facade_error": (-1, "error"),
    "gross_volume_error": (-1, "error"),
    "crash": (-1, "rate"),
    "ifc_fail": (-1, "rate"),
    "runtime_s": (-1, "runtime"),
}
PROXY_METRICS: dict[str, tuple[int, str]] = {
    "closed": (+1, "rate"),
    "scale_from_dimensions": (+1, "rate"),
    "stamp_agreement": (+1, "rate"),
    "crash": (-1, "rate"),
    "ifc_fail": (-1, "rate"),
    "runtime_s": (-1, "runtime"),
}


def _cap(v: float | None) -> float | None:
    return None if v is None else float(min(v, ERROR_CAP))


def _is_exterior(kind: WallKind, thickness_m: float) -> bool:
    if kind == WallKind.EXTERIOR:
        return True
    return kind == WallKind.UNKNOWN and thickness_m >= EXTERIOR_MIN_THICKNESS_M


def _shift(graph: PlanGraph, off: tuple[float, float]) -> PlanGraph:
    g = graph.model_copy(deep=True)
    for n in g.nodes:
        n.x += off[0]
        n.y += off[1]
    for r in g.rooms:
        r.polygon = [(x + off[0], y + off[1]) for x, y in r.polygon]
    return g


def room_ious(pred: PlanGraph, gt: PlanGraph) -> list[float]:
    """IoU of every GT room with its Hungarian-matched predicted room (0 if unmatched).
    Both graphs must already be aligned."""
    pr = [Polygon(r.polygon) for r in pred.rooms if _valid_room(r.polygon)]
    gr = [Polygon(r.polygon) if _valid_room(r.polygon) else None for r in gt.rooms]
    out = [0.0] * len(gr)
    valid = [(j, g) for j, g in enumerate(gr) if g is not None]
    if not pr or not valid:
        return out
    iou = np.zeros((len(pr), len(valid)))
    for i, p in enumerate(pr):
        for k, (_, g) in enumerate(valid):
            if p.intersects(g):
                inter = p.intersection(g).area
                iou[i, k] = inter / (p.area + g.area - inter)
    rows, cols = linear_sum_assignment(-iou)
    for r, c in zip(rows, cols, strict=True):
        out[valid[c][0]] = float(iou[r, c])
    return out


def _wall_class_accuracy(pred: PlanGraph, gt: PlanGraph, step: float = 0.05) -> float | None:
    """Length-weighted share of GT centreline, covered by a parallel predicted wall within
    ``WALL_TOL_M``, whose exterior/interior class equals the GT class. Uncovered length is
    ignored here (wall recall measures it); None if nothing is covered."""
    gpos = {n.id: (n.x, n.y) for n in gt.nodes}
    ppos = {n.id: (n.x, n.y) for n in pred.nodes}
    pw = [w for w in pred.walls if w.start in ppos and w.end in ppos and w.start != w.end]
    if not pw:
        return None
    s0 = np.array([ppos[w.start] for w in pw], float)
    s1 = np.array([ppos[w.end] for w in pw], float)
    pcls = np.array([_is_exterior(w.kind, w.thickness_m) for w in pw])
    v = s1 - s0
    vlen2 = np.maximum((v**2).sum(1), 1e-12)
    vdir = v / np.sqrt(vlen2)[:, None]
    good = total = 0.0
    for w in gt.walls:
        if w.start not in gpos or w.end not in gpos:
            continue
        (x0, y0), (x1, y1) = gpos[w.start], gpos[w.end]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-9:
            continue
        n = max(1, int(math.ceil(length / step)))
        t = (np.arange(n) + 0.5) / n
        p = np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1)
        d = np.array([(x1 - x0) / length, (y1 - y0) / length])
        tt = np.clip(((p[:, None, :] - s0[None]) * v[None]).sum(-1) / vlen2[None], 0.0, 1.0)
        dist = np.linalg.norm(p[:, None, :] - (s0[None] + tt[..., None] * v[None]), axis=-1)
        parallel = np.abs(vdir @ d) > math.cos(math.radians(15))
        dist = np.where(parallel[None], dist, np.inf)
        best = dist.argmin(axis=1)
        ok = dist[np.arange(n), best] <= WALL_TOL_M
        gcls = w.kind == WallKind.EXTERIOR
        total += ok.sum() * length / n
        good += (ok & (pcls[best] == gcls)).sum() * length / n
    return float(good / total) if total > 0 else None


def _opening_pr(pred: PlanGraph, gt: PlanGraph) -> dict[str, float | None]:
    po, go = _opening_points(pred, (0.0, 0.0)), _opening_points(gt, (0.0, 0.0))
    out: dict[str, float | None] = {}
    for kind in (OpeningKind.DOOR, OpeningKind.WINDOW):
        a = [o for o in po if o[0] == kind]
        b = [o for o in go if o[0] == kind]
        m = _match_points(a, b, OPENING_TOL_M)
        out[f"{kind.value}_precision"] = len(m) / len(a) if a else (None if not b else 0.0)
        out[f"{kind.value}_recall"] = len(m) / len(b) if b else None
    return out


def _windows_with_sill(g: PlanGraph) -> list[tuple[tuple[float, float], float]]:
    """(centre, sill) of every window with a sill height."""
    pos = {n.id: (n.x, n.y) for n in g.nodes}
    walls = {w.id: w for w in g.walls}
    out = []
    for o in g.openings:
        w = walls.get(o.wall_id)
        if o.kind != OpeningKind.WINDOW or o.sill_m is None or w is None:
            continue
        if w.start not in pos or w.end not in pos or pos[w.start] == pos[w.end]:
            continue
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        t = (o.offset_m + o.width_m / 2) / math.hypot(x1 - x0, y1 - y0)
        out.append(((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), o.sill_m))
    return out


def _sill_error(pred: PlanGraph, gt: PlanGraph) -> float | None:
    """Mean |sill_pred - sill_gt| over position-matched windows that both have a sill."""
    a, b = _windows_with_sill(pred), _windows_with_sill(gt)
    m = _match_points([("w", p, 0.0) for p, _ in a], [("w", p, 0.0) for p, _ in b], OPENING_TOL_M)
    errs = [abs(a[i][1] - b[j][1]) for i, j in m]
    return float(np.mean(errs)) if errs else None


def edit_cost(pred: PlanGraph, gt: PlanGraph) -> float:
    """Graph edit proxy for expert correction time: node insertions + deletions (GT and
    predicted wall nodes matched 1:1 within ``NODE_MATCH_TOL_M``), wall-edge insertions +
    deletions (an edge is kept if both ends are matched and the GT has a wall between them),
    opening insertions + deletions + kind changes (matched within ``OPENING_TOL_M``). Both
    graphs must be aligned. Only counts, no geometry fine-tuning (moving a matched node is
    free), so it is a lower bound on the real correction effort."""
    gused = {e for w in gt.walls for e in (w.start, w.end)}
    pused = {e for w in pred.walls for e in (w.start, w.end)}
    gn = [n for n in gt.nodes if n.id in gused]
    pn = [n for n in pred.nodes if n.id in pused]
    mapping: dict[int, int] = {}
    if gn and pn:
        ga = np.array([(n.x, n.y) for n in gn])
        pa = np.array([(n.x, n.y) for n in pn])
        cost = np.linalg.norm(pa[:, None] - ga[None], axis=-1)
        rows, cols = linear_sum_assignment(np.where(cost <= NODE_MATCH_TOL_M, cost, 1e6))
        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] <= NODE_MATCH_TOL_M:
                mapping[pn[r].id] = gn[c].id
    node_edits = (len(pn) - len(mapping)) + (len(gn) - len(mapping))
    gedges = {frozenset((w.start, w.end)) for w in gt.walls if w.start != w.end}
    pedges = {frozenset((w.start, w.end)) for w in pred.walls if w.start != w.end}
    mapped = set()
    for e in pedges:
        a, b = tuple(e)
        if a in mapping and b in mapping:
            m = frozenset((mapping[a], mapping[b]))
            if m in gedges:
                mapped.add(m)
    edge_edits = (len(pedges) - len(mapped)) + (len(gedges) - len(mapped))
    po, go = _opening_points(pred, (0.0, 0.0)), _opening_points(gt, (0.0, 0.0))
    m = _match_points(po, go, OPENING_TOL_M)
    kind_changes = sum(po[i][0] != go[j][0] for i, j in m)
    opening_edits = (len(po) - len(m)) + (len(go) - len(m)) + kind_changes
    return float(node_edits + edge_edits + opening_edits)


def _gt_size(gt: PlanGraph) -> float:
    """Edit cost of building the GT from an empty prediction."""
    return edit_cost(PlanGraph(m_per_px=gt.m_per_px, scale_source="default", image_height_px=1), gt)


def footprint(graph: PlanGraph):
    """Gross footprint: union of wall bodies (centreline +- thickness/2) and rooms, holes
    filled. Gross volume = footprint x storey height, so its relative error equals this one's."""
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    parts = []
    for w in graph.walls:
        if w.start in pos and w.end in pos and pos[w.start] != pos[w.end]:
            (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
            length = math.hypot(x1 - x0, y1 - y0)
            dx, dy = (x1 - x0) / length, (y1 - y0) / length
            t = max(w.thickness_m, 0.01) / 2
            ex, ey = dx * t, dy * t
            parts.append(
                Polygon(
                    [
                        (x0 - ex - dy * t, y0 - ey + dx * t),
                        (x1 + ex - dy * t, y1 + ey + dx * t),
                        (x1 + ex + dy * t, y1 + ey - dx * t),
                        (x0 - ex + dy * t, y0 - ey - dx * t),
                    ]
                )
            )
    parts += [Polygon(r.polygon) for r in graph.rooms if _valid_room(r.polygon)]
    if not parts:
        return Polygon()
    u = unary_union(parts).buffer(0)
    geoms = getattr(u, "geoms", [u])
    return unary_union([Polygon(g.exterior) for g in geoms if not g.is_empty])


def _exterior_length(graph: PlanGraph) -> float:
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    return sum(
        math.dist(pos[w.start], pos[w.end])
        for w in graph.walls
        if w.start in pos and w.end in pos and _is_exterior(w.kind, w.thickness_m)
    )


def _window_facades(graph: PlanGraph) -> dict[str, float]:
    """Window area per facade (N/E/S/W by the outward side of the wall relative to the
    footprint centroid). Width x height; a missing height counts as 1.0 m (so width only)."""
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    walls = {w.id: w for w in graph.walls}
    fp = footprint(graph)
    if fp.is_empty:
        return {}
    cx, cy = fp.centroid.x, fp.centroid.y
    out: dict[str, float] = {}
    for o in graph.openings:
        if o.kind != OpeningKind.WINDOW or o.wall_id not in walls:
            continue
        w = walls[o.wall_id]
        if w.start not in pos or w.end not in pos:
            continue
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        mx, my = (x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy
        if abs(x1 - x0) >= abs(y1 - y0):  # horizontal wall -> N or S facade
            side = "N" if my > 0 else "S"
        else:
            side = "E" if mx > 0 else "W"
        out[side] = out.get(side, 0.0) + o.width_m * (o.height_m or 1.0)
    return out


def envelope_errors(pred: PlanGraph, gt: PlanGraph) -> dict[str, float | None]:
    gl = _exterior_length(gt)
    ext = abs(_exterior_length(pred) - gl) / gl if gl > 0 else None
    gw, pw = _window_facades(gt), _window_facades(pred)
    gtot = sum(gw.values())
    win = (
        sum(abs(pw.get(k, 0.0) - gw.get(k, 0.0)) for k in set(gw) | set(pw)) / gtot
        if gtot > 0
        else None
    )
    ga = footprint(gt).area
    vol = abs(footprint(pred).area - ga) / ga if ga > 0 else None
    return {
        "ext_wall_length_error": _cap(ext),
        "window_facade_error": _cap(win),
        "gross_volume_error": _cap(vol),
    }


def _room_count_error(pred: PlanGraph, gt: PlanGraph) -> float | None:
    """|predicted rooms - GT rooms| / GT rooms, capped at 1."""
    return _cap(abs(len(pred.rooms) - len(gt.rooms)) / len(gt.rooms)) if gt.rooms else None


SCALEFREE_NA = (
    "room_area_error", "total_area_error", "wall_precision", "wall_recall", "wall_f1",
    "thickness_class_acc", "door_precision", "door_recall", "window_precision", "window_recall",
    "sill_error_m", "edit_cost", "ext_wall_length_error", "window_facade_error",
    "gross_volume_error",
)  # fmt: skip
"""Panel metrics that need a metric scale: ``None`` on scale-free splits (CubiCasa5k)."""


def similarity_align(pred: PlanGraph, gt: PlanGraph, iters: int = 3) -> PlanGraph | None:
    """Scale and translate the predicted rooms onto the GT rooms (no rotation: both come from
    the same upright page). Scale = sqrt of the ratio of total room areas, translation = centroid
    of the room union, then ``iters`` refinements by the median offset of matched room
    centroids. ``None`` if either side has no valid room."""
    pr = [Polygon(r.polygon) for r in pred.rooms if _valid_room(r.polygon)]
    gr = [Polygon(r.polygon) for r in gt.rooms if _valid_room(r.polygon)]
    if not pr or not gr:
        return None
    from shapely import affinity

    s = math.sqrt(sum(g.area for g in gr) / sum(p.area for p in pr))
    cp, cg = unary_union(pr).centroid, unary_union(gr).centroid
    dx, dy = cg.x - s * cp.x, cg.y - s * cp.y
    for _ in range(iters):
        mp = [affinity.translate(affinity.scale(p, s, s, origin=(0, 0)), dx, dy) for p in pr]
        iou = np.zeros((len(mp), len(gr)))
        for i, p in enumerate(mp):
            for j, g in enumerate(gr):
                if p.intersects(g):
                    inter = p.intersection(g).area
                    iou[i, j] = inter / (p.area + g.area - inter)
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] > 0.1]
        if not pairs:
            break
        dx += float(np.median([gr[c].centroid.x - mp[r].centroid.x for r, c in pairs]))
        dy += float(np.median([gr[c].centroid.y - mp[r].centroid.y for r, c in pairs]))
    out = pred.model_copy(deep=True)
    out.rooms = [
        r.model_copy(update={"polygon": [(s * x + dx, s * y + dy) for x, y in r.polygon]})
        for r in out.rooms
    ]
    return out


def scalefree_crash_panel(gt: PlanGraph) -> dict[str, float | None]:
    out = {k: None for k in METRICS}
    out.update(
        correct_closed=0.0, closed=0.0, room_iou=0.0, room_precision=0.0, room_recall=0.0,
        room_count_error=ERROR_CAP, crash=1.0, ifc_fail=1.0, runtime_s=None,
    )  # fmt: skip
    return out


def scalefree_panel(
    pred: PlanGraph | None,
    gt: PlanGraph,
    *,
    runtime_s: float | None = None,
    ifc_ok: bool | None = None,
) -> dict[str, float | None]:
    """Panel for plans without a metric scale (CubiCasa5k): every metric that needs metres is
    ``None`` (``SCALEFREE_NA``). Rooms are compared after :func:`similarity_align`. Correct-closed
    drops the area condition (the aligned areas are equal by construction): closed, every GT room
    matched with IoU >= 0.85, and the room count equal."""
    if pred is None:
        return scalefree_crash_panel(gt)
    closed = closure_metrics(pred).is_closed
    out: dict[str, float | None] = {k: None for k in METRICS}
    out.update(
        closed=float(closed), crash=0.0, ifc_fail=0.0 if ifc_ok else 1.0, runtime_s=runtime_s,
        room_count_error=_room_count_error(pred, gt),
    )  # fmt: skip
    al = similarity_align(pred, gt)
    if al is None:
        out.update(correct_closed=0.0, room_iou=0.0, room_precision=0.0, room_recall=0.0)
        return out
    ious = room_ious(al, gt)
    pr = [r for r in al.rooms if _valid_room(r.polygon)]
    # precision / recall at IoU 0.5 as in compare_to_ground_truth (room_iou_threshold default)
    cmp = compare_to_ground_truth(al, gt, align="none", wall_tol_m=WALL_TOL_M)
    out.update(
        correct_closed=float(
            closed and min(ious) >= CORRECT_ROOM_IOU and len(pred.rooms) == len(gt.rooms)
        ),
        room_iou=float(sum(ious) / len(gt.rooms)),
        room_precision=cmp.room_precision if pr else 0.0,
        room_recall=cmp.room_recall,
    )
    return out


def crash_panel(gt: PlanGraph) -> dict[str, float | None]:
    """Scores of a plan whose run crashed or was rejected."""
    out: dict[str, float | None] = {k: 0.0 for k, (d, _) in METRICS.items() if d > 0}
    out.update({k: ERROR_CAP for k, (d, kind) in METRICS.items() if d < 0 and kind == "error"})
    kinds = {o.kind for o in gt.openings}
    for k in ("door", "window"):
        has = OpeningKind(k) in kinds
        out[f"{k}_recall"] = 0.0 if has else None
        out[f"{k}_precision"] = 0.0 if has else None
    if not any(o.kind == OpeningKind.WINDOW and o.sill_m is not None for o in gt.openings):
        out["sill_error_m"] = None
    out["edit_cost"] = _gt_size(gt)
    out["thickness_class_acc"] = 0.0
    out["crash"] = 1.0
    out["ifc_fail"] = 1.0
    out["runtime_s"] = None
    return out


def plan_panel(
    pred: PlanGraph | None,
    gt: PlanGraph,
    *,
    runtime_s: float | None = None,
    ifc_ok: bool | None = None,
) -> dict[str, float | None]:
    """Full §3 panel for one plan; ``pred=None`` = crash."""
    if pred is None:
        return crash_panel(gt)
    cmp = compare_to_ground_truth(pred, gt, wall_tol_m=WALL_TOL_M)
    al = _shift(pred, cmp.offset_m)
    closed = closure_metrics(pred).is_closed
    ious = room_ious(al, gt)
    total_err = cmp.total_area_error if cmp.total_area_error is not None else 1.0
    correct = (
        closed
        and bool(ious)
        and min(ious) >= CORRECT_ROOM_IOU
        and len(pred.rooms) == len(gt.rooms)
        and total_err <= CORRECT_AREA_ERROR
    )
    out: dict[str, float | None] = {
        "correct_closed": float(correct),
        "closed": float(closed),
        "room_iou": cmp.room_iou_mean,
        "room_precision": cmp.room_precision,
        "room_recall": cmp.room_recall,
        # no matched room: the room area is entirely wrong
        "room_area_error": _cap(cmp.room_area_error if cmp.room_area_error is not None else 1.0),
        "room_count_error": _room_count_error(pred, gt),
        "total_area_error": _cap(total_err),
        "wall_precision": cmp.wall_precision,
        "wall_recall": cmp.wall_recall,
        "wall_f1": cmp.wall_f1,
        "thickness_class_acc": _wall_class_accuracy(al, gt),
        **_opening_pr(al, gt),
        "sill_error_m": _sill_error(al, gt),
        "edit_cost": edit_cost(al, gt),
        **envelope_errors(al, gt),
        "crash": 0.0,
        "ifc_fail": 0.0 if ifc_ok else 1.0,
        "runtime_s": runtime_s,
    }
    if out["thickness_class_acc"] is None and gt.walls:
        out["thickness_class_acc"] = 0.0  # no wall overlaps: no class is right
    return out


def proxy_panel(
    pred: PlanGraph | None, *, runtime_s: float | None = None, ifc_ok: bool | None = None
) -> dict[str, float | None]:
    """Label-free panel (real-dev): closure, scale from dimensions, room-stamp agreement (share
    of rooms with a stamp whose computed area is within 10 % of it; None if no stamps)."""
    if pred is None:
        return {
            "closed": 0.0,
            "scale_from_dimensions": 0.0,
            "stamp_agreement": None,
            "crash": 1.0,
            "ifc_fail": 1.0,
            "runtime_s": None,
        }
    stamped = [r for r in pred.rooms if r.stamp_area_m2]
    good = [r for r in stamped if abs(r.area_m2 - r.stamp_area_m2) <= 0.10 * r.stamp_area_m2]
    return {
        "closed": float(closure_metrics(pred).is_closed),
        "scale_from_dimensions": float(pred.scale_source == "dimensions"),
        "stamp_agreement": len(good) / len(stamped) if stamped else None,
        "crash": 0.0,
        "ifc_fail": 0.0 if ifc_ok else 1.0,
        "runtime_s": runtime_s,
    }


__all__ = [
    "METRICS",
    "PROXY_METRICS",
    "crash_panel",
    "edit_cost",
    "envelope_errors",
    "footprint",
    "plan_panel",
    "proxy_panel",
    "room_ious",
    "scalefree_panel",
    "SCALEFREE_NA",
    "similarity_align",
]
