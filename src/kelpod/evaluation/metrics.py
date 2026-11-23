"""Closure metric (is the model closed?) and ground-truth comparison (is it *right*?).

Both are self-contained (NetworkX, Shapely, SciPy) so they do not depend on ``kelpod.geometry``.
Definitions follow ``docs/ARCHITECTURE.md``, "Closure metric (honest version)".
"""

from __future__ import annotations

import math
from typing import Literal

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from kelpod.contracts import ConsistencyReport, OpeningKind, PlanGraph


def _wall_graph(graph: PlanGraph) -> nx.MultiGraph:
    g = nx.MultiGraph()
    g.add_nodes_from(n.id for n in graph.nodes)
    for w in graph.walls:
        if w.start != w.end:
            g.add_edge(w.start, w.end, wall=w.id)
    return g


def _valid_room(polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    poly = Polygon(polygon)
    return poly.is_valid and poly.area > 1e-4


def closure_metrics(graph: PlanGraph) -> ConsistencyReport:
    """Closure of a plan graph, computed from the graph alone.

    * ``node_closure``: share of nodes with degree >= 2 (no loose wall ends);
    * ``room_closure``: share of rooms that are valid closed polygons (0 if there are no rooms);
    * ``n_components``: connected components of the wall graph (isolated nodes count);
    * ``dimension_residual_m`` is copied from ``graph.report`` if the geometry step set it.
    """
    g = _wall_graph(graph)
    issues: list[str] = []
    ids = {n.id for n in graph.nodes}
    missing = {e for w in graph.walls for e in (w.start, w.end)} - ids
    if missing:
        issues.append(f"{len(missing)} wall end(s) refer to missing nodes")
    degree = dict(g.degree())
    dangling = sorted(n for n, d in degree.items() if d == 1)
    isolated = sorted(n for n, d in degree.items() if d == 0)
    n_nodes = g.number_of_nodes()
    node_closure = (sum(1 for d in degree.values() if d >= 2) / n_nodes) if n_nodes else 0.0
    n_comp = nx.number_connected_components(g) if n_nodes else 0
    valid = [_valid_room(r.polygon) for r in graph.rooms]
    room_closure = (sum(valid) / len(valid)) if valid else 0.0

    if not graph.walls:
        issues.append("no walls")
    if n_comp > 1:
        issues.append(f"the walls form {n_comp} separate groups instead of one")
    if dangling:
        issues.append(f"{len(dangling)} wall end(s) do not meet another wall")
    if isolated:
        issues.append(f"{len(isolated)} node(s) without any wall")
    if not graph.rooms:
        issues.append("no rooms")
    elif not all(valid):
        issues.append(f"{len(valid) - sum(valid)} room(s) are not valid closed polygons")
    for r in graph.rooms:
        if r.stamp_area_m2 and _valid_room(r.polygon):
            a = Polygon(r.polygon).area
            if abs(a - r.stamp_area_m2) > 0.10 * r.stamp_area_m2:
                issues.append(
                    f"room {r.name or r.id}: area {a:.2f} m² differs from the stamp "
                    f"{r.stamp_area_m2:.2f} m² by more than 10 %"
                )
    residual = graph.report.dimension_residual_m if graph.report else None
    return ConsistencyReport(
        n_nodes=n_nodes, n_walls=len(graph.walls), n_rooms=len(graph.rooms), n_components=n_comp,
        dangling_nodes=dangling, node_closure=node_closure, room_closure=room_closure,
        dimension_residual_m=residual, issues=issues,
    )


# --------------------------------------------------------------------------------------------
# Comparison with ground truth
# --------------------------------------------------------------------------------------------


class GroundTruthComparison(BaseModel):
    """How well a predicted graph matches the ground truth (synthetic plans only)."""

    model_config = ConfigDict(extra="forbid")

    offset_m: tuple[float, float] = (0.0, 0.0)
    """Translation added to the prediction before comparing (bounding-box alignment)."""
    n_rooms_pred: int = 0
    n_rooms_gt: int = 0
    room_iou_mean: float = 0.0
    """Mean IoU over ground-truth rooms after Hungarian matching (unmatched rooms count as 0)."""
    room_precision: float = 0.0
    room_recall: float = 0.0
    """Matched pairs with IoU >= ``room_iou_threshold`` / number of predicted (ground-truth) rooms."""
    room_area_error: float | None = None
    """Mean relative area error |A_pred - A_gt| / A_gt over matched rooms."""
    total_area_error: float | None = None
    """Relative error of the summed room area."""
    room_name_accuracy: float | None = None
    """Share of matched rooms whose predicted name equals the ground truth (if any names)."""
    wall_precision: float = 0.0
    wall_recall: float = 0.0
    wall_f1: float = 0.0
    """Length-weighted: share of centreline length within ``wall_tol_m`` of a parallel centreline
    of the other graph (robust to walls being split at different junctions)."""
    opening_recall: dict[str, float | None] = Field(default_factory=dict)
    """Per kind (``door``, ``window``) and ``any`` (position only, kind ignored); None if the
    ground truth has none of that kind."""
    opening_precision: float | None = None
    opening_width_error_m: float | None = None
    """Mean |width_pred - width_gt| over kind-matched openings."""


def _bbox_min(graph: PlanGraph) -> tuple[float, float] | None:
    used = {e for w in graph.walls for e in (w.start, w.end)}
    pts = [(n.x, n.y) for n in graph.nodes if n.id in used] or [(n.x, n.y) for n in graph.nodes]
    if not pts:
        return None
    xs, ys = zip(*pts, strict=True)
    return (min(xs), min(ys))


def _segments(graph: PlanGraph, offset: tuple[float, float]) -> np.ndarray:
    pos = {n.id: (n.x + offset[0], n.y + offset[1]) for n in graph.nodes}
    segs = [
        (*pos[w.start], *pos[w.end])
        for w in graph.walls
        if w.start in pos and w.end in pos and pos[w.start] != pos[w.end]
    ]
    return np.array(segs, float).reshape(-1, 4)


def _covered_share(a: np.ndarray, b: np.ndarray, tol: float, step: float = 0.05) -> float:
    """Length share of segments ``a`` within ``tol`` of a roughly parallel segment of ``b``."""
    if len(a) == 0:
        return 0.0
    if len(b) == 0:
        return 0.0
    pts, dirs, weights = [], [], []
    for x0, y0, x1, y1 in a:
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(length / step)))
        t = (np.arange(n) + 0.5) / n
        pts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1))
        dirs.append(np.repeat([[(x1 - x0) / length, (y1 - y0) / length]], n, axis=0))
        weights.append(np.full(n, length / n))
    p, d, wgt = np.concatenate(pts), np.concatenate(dirs), np.concatenate(weights)
    s0, s1 = b[:, :2], b[:, 2:]
    v = s1 - s0
    vlen2 = np.maximum((v**2).sum(1), 1e-12)
    # point-to-segment distances, (N, M)
    t = np.clip(((p[:, None, :] - s0[None]) * v[None]).sum(-1) / vlen2[None], 0.0, 1.0)
    closest = s0[None] + t[..., None] * v[None]
    dist = np.linalg.norm(p[:, None, :] - closest, axis=-1)
    vdir = v / np.sqrt(vlen2)[:, None]
    parallel = np.abs(d @ vdir.T) > math.cos(math.radians(15))
    dist = np.where(parallel, dist, np.inf)
    ok = dist.min(axis=1) <= tol
    return float(wgt[ok].sum() / wgt.sum())


def _opening_points(graph: PlanGraph, offset: tuple[float, float]):
    pos = {n.id: (n.x + offset[0], n.y + offset[1]) for n in graph.nodes}
    walls = {w.id: w for w in graph.walls}
    out = []
    for o in graph.openings:
        w = walls.get(o.wall_id)
        if w is None or w.start not in pos or w.end not in pos:
            continue
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-9:
            continue
        s = o.offset_m + o.width_m / 2
        out.append((o.kind, (x0 + (x1 - x0) * s / length, y0 + (y1 - y0) * s / length), o.width_m))
    return out


def _match_points(a, b, tol: float) -> list[tuple[int, int]]:
    if not a or not b:
        return []
    pa = np.array([p for _, p, _ in a])
    pb = np.array([p for _, p, _ in b])
    cost = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
    rows, cols = linear_sum_assignment(np.where(cost <= tol, cost, 1e6))
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= tol]


def _norm_name(name: str | None) -> str:
    return (name or "").strip().lower().rstrip(".")


def compare_to_ground_truth(
    pred: PlanGraph,
    gt: PlanGraph,
    *,
    align: Literal["bbox", "none"] = "bbox",
    wall_tol_m: float = 0.10,
    room_iou_threshold: float = 0.5,
    opening_tol_m: float = 0.30,
) -> GroundTruthComparison:
    """Compare a predicted plan graph with the ground truth.

    The prediction is first shifted so the minimum corner of its wall-node bounding box meets the
    ground truth's (``align="bbox"``); rotation is not compensated, so deskew errors show up here.
    """
    offset = (0.0, 0.0)
    if align == "bbox":
        a, b = _bbox_min(pred), _bbox_min(gt)
        if a is not None and b is not None:
            offset = (b[0] - a[0], b[1] - a[1])
    res = GroundTruthComparison(offset_m=offset, n_rooms_pred=len(pred.rooms),
                                n_rooms_gt=len(gt.rooms))

    # rooms: Hungarian matching on IoU
    def polys(graph: PlanGraph, off):
        out = []
        for r in graph.rooms:
            if _valid_room(r.polygon):
                out.append((r, Polygon([(x + off[0], y + off[1]) for x, y in r.polygon])))
        return out

    pr, gr = polys(pred, offset), polys(gt, (0.0, 0.0))
    if pr and gr:
        iou = np.zeros((len(pr), len(gr)))
        for i, (_, p) in enumerate(pr):
            for j, (_, g) in enumerate(gr):
                if p.intersects(g):
                    inter = p.intersection(g).area
                    iou[i, j] = inter / (p.area + g.area - inter)
        rows, cols = linear_sum_assignment(-iou)
        res.room_iou_mean = float(iou[rows, cols].sum() / len(gt.rooms))
        tp = [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= room_iou_threshold]
        res.room_precision = len(tp) / len(pred.rooms)
        res.room_recall = len(tp) / len(gt.rooms)
        if tp:
            res.room_area_error = float(np.mean(
                [abs(pr[r][1].area - gr[c][1].area) / gr[c][1].area for r, c in tp]
            ))
            named = [(pr[r][0], gr[c][0]) for r, c in tp if gr[c][0].name]
            if named:
                res.room_name_accuracy = sum(
                    _norm_name(p.name) == _norm_name(g.name) for p, g in named
                ) / len(named)
        gt_area = sum(g.area for _, g in gr)
        res.total_area_error = abs(sum(p.area for _, p in pr) - gt_area) / gt_area
    elif gr:
        res.total_area_error = 1.0

    # walls: length-weighted coverage of centrelines
    ps, gs = _segments(pred, offset), _segments(gt, (0.0, 0.0))
    res.wall_precision = _covered_share(ps, gs, wall_tol_m)
    res.wall_recall = _covered_share(gs, ps, wall_tol_m)
    p, r = res.wall_precision, res.wall_recall
    res.wall_f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    # openings: Hungarian on centre distance, per kind and kind-agnostic
    po, go = _opening_points(pred, offset), _opening_points(gt, (0.0, 0.0))
    recall: dict[str, float | None] = {}
    widths: list[float] = []
    n_tp = 0
    for kind in (OpeningKind.DOOR, OpeningKind.WINDOW):
        a = [o for o in po if o[0] == kind]
        b = [o for o in go if o[0] == kind]
        m = _match_points(a, b, opening_tol_m)
        n_tp += len(m)
        widths += [abs(a[i][2] - b[j][2]) for i, j in m]
        recall[kind.value] = len(m) / len(b) if b else None
    recall["any"] = len(_match_points(po, go, opening_tol_m)) / len(go) if go else None
    res.opening_recall = recall
    res.opening_precision = n_tp / len(po) if po else (None if not go else 0.0)
    res.opening_width_error_m = float(np.mean(widths)) if widths else None
    return res
