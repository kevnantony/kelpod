"""Vector floor plans (Swiss Dwellings, ResPlan, CubiCasa5k) -> PlanGraph ground truth + a page.

The out-of-distribution splits (``docs/EVAL_PROTOCOL.md`` §2, ``eval/README.md``) need the same
artefacts as a synthetic plan: an image-only ``plan.pdf`` and a ``ground_truth.json`` with a
:class:`~kelpod.contracts.PlanGraph` in metres. This module is the deterministic converter and
renderer. It is a copy-and-adapt of the ideas in ``training/kelpod_train/render.py`` (nothing is
imported from ``training/``), and it draws with the **default train style** of the synthetic
generator (same canvas, pens, hatching, fonts and scan degradation, 300 dpi, 1:100). The held-out
style options are never used here.

Ground truth, step by step (what the GT contains, and what it does not):

* walls: every wall polygon becomes one or more rectangles (centreline, thickness). Swiss Dwellings
  walls already are rectangles; ResPlan walls are rectilinear unions that are cut into thin strips
  (grid decomposition); other shapes are cut the same way in their own frame. Collinear pieces are
  merged, ends are snapped onto the wall they meet, centrelines are noded at every junction
  (``kind``: exterior if one face looks outside the footprint, else drywall below 15 cm, else
  interior). Wall area that cannot be decomposed (thick blocks, curves) is *not* in the GT and
  *not* drawn; its share is recorded and plans above ``MAX_UNSUPPORTED`` are not used.
* openings: door and window boxes are attached to the wall piece they lie in (offset, width);
  doors get sill 0 and height 2.01 m, windows have **no sill and no height** (the sources have
  none), so ``sill_error_m`` is undefined on these splits. Openings without a wall are dropped.
* rooms: the source room polygons (largest part, at least 1.5 m^2), CCW, German stamp names.
  Balconies, terraces, gardens, shafts, voids and lift cars are neither rooms nor drawn.
  Open-plan areas that the source splits without a wall stay separate GT rooms (the pipeline cannot
  separate them: this is part of what is out of distribution).
* not drawn and not in the GT: fixtures, furniture, columns, railings, stairs symbols.
* non-rectilinear geometry is kept (Swiss floors often have rotated parts). The plan is rotated so
  that the dominant wall direction is axis-parallel (as a person would draw it); the GT is in that
  rotated frame, and the page is drawn in it.

Dimension chains (two per side: wall faces, overall) and room stamps (name + area) are drawn like
in the synthetic plans, so the pipeline can find its scale from the page (the GT graph carries
``scale_source="ground_truth"`` like every GT; the locked runner rejects a prediction with it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import polylabel, unary_union

from kelpod.contracts import Node, Opening, OpeningKind, PlanGraph, Room, Wall, WallKind, m_to_px
from kelpod.evaluation.metrics import closure_metrics
from kelpod.evaluation.synth.degrade import degrade as _degrade
from kelpod.evaluation.synth.render import (
    AREA_TEXT_MM,
    DIM_TEXT_MM,
    LEAF_LINE_MM,
    NAME_TEXT_MM,
    THIN_LINE_MM,
    TICK_LEN_MM,
    TICK_LINE_MM,
    WALL_LINE_MM,
    Style,
    _Canvas,
    _polys,
)
from kelpod.evaluation.synth.text import format_area, format_length

VECTOR_VERSION = "1"
DPI = 300.0
PLAN_SCALE = 100.0
M_PER_PX = 0.0254 / DPI * PLAN_SCALE
MARGIN_M = 4.0
MIN_EXTENT_M = 3.0
MAX_EXTENT_M = 60.0
MAX_UNSUPPORTED = 0.03
"""Largest share of wall area that may be left out of the GT (and the drawing)."""
T_CAP = 0.60
"""Thickest strip taken as a wall (m); thicker blocks count as unsupported."""
HATCH_MIN_T = 0.15
"""Walls at least this thick are hatched (masonry) and classified interior/exterior; thinner
ones are plain drywall partitions."""
DOOR_HEIGHT = 2.01
MIN_ROOM_M2 = 1.5
SNAP = 0.03


# --------------------------------------------------------------------------------------------
# input container
# --------------------------------------------------------------------------------------------


@dataclass
class VectorPlan:
    """A plan in metres (y up): wall polygons, opening polygons, rooms with German names."""

    walls: list[Polygon]
    doors: list[Polygon] = field(default_factory=list)
    windows: list[Polygon] = field(default_factory=list)
    rooms: list[tuple[Polygon, str | None]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Rect:
    """A wall as a centreline ``a -> b`` with thickness ``t``; ``ea`` / ``eb`` extend the drawn
    body past the ends (half thickness of the wall it meets)."""

    a: np.ndarray
    b: np.ndarray
    t: float
    ea: float = 0.0
    eb: float = 0.0
    bridge: bool = False

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def u(self) -> np.ndarray:
        d = self.b - self.a
        return d / max(float(np.linalg.norm(d)), 1e-12)

    def polygon(self) -> Polygon:
        u = self.u
        n = np.array([-u[1], u[0]]) * self.t / 2
        a, b = self.a - u * self.ea, self.b + u * self.eb
        return Polygon([a + n, b + n, b - n, a - n])


# --------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------


def dominant_angle(polys: list[Polygon]) -> float:
    """Dominant edge direction modulo 90 degrees (length-weighted circular mean)."""
    acc = 0j
    for p in polys:
        for g in _polys(p):
            c = np.asarray(g.exterior.coords)
            d = np.diff(c, axis=0)
            acc += np.sum(np.hypot(d[:, 0], d[:, 1]) * np.exp(4j * np.arctan2(d[:, 1], d[:, 0])))
    return math.degrees(np.angle(acc) / 4) if abs(acc) > 0 else 0.0


def _rect_axes(p: Polygon) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centre, unit long axis, length and width of the minimum rotated rectangle."""
    c = np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 >= l2:
        return c.mean(0), e1 / max(l1, 1e-9), float(l1), float(l2)
    return c.mean(0), e2 / max(l2, 1e-9), float(l2), float(l1)


def opening_box(opening: Polygon, walls_u, min_depth: float = 0.08) -> Polygon:
    """The gap box of an opening polygon in its wall: the axis along the wall is the rectangle
    axis whose jambs (just beyond both ends) hit wall; the depth across is the wall thickness
    measured at the jambs (the polygon's own width if no wall is found)."""
    centre, u0, l0, w0 = _rect_axes(opening)
    best = None
    for u, length, width in ((u0, l0, w0), (np.array([-u0[1], u0[0]]), w0, l0)):
        n = np.array([-u[1], u[0]])
        depths = []
        if walls_u is not None and not walls_u.is_empty:
            for t in (-0.5, 0.5):
                c = centre + u * t * (length + 0.12)
                inter = LineString([c - n * 0.6, c + n * 0.6]).intersection(walls_u)
                for piece in getattr(inter, "geoms", [inter]):
                    if (
                        not piece.is_empty
                        and piece.distance(Point(*c)) < 0.02
                        and piece.length < 0.7
                    ):
                        depths.append(piece.length)
        score = (len(depths), length >= width)
        if best is None or score > best[0]:
            best = (score, u, n, length, max(depths) if depths else max(width, min_depth))
    _, u, n, length, depth = best
    depth = max(depth, min_depth)
    a, b = centre - u * length / 2, centre + u * length / 2
    return Polygon([a + n * depth / 2, b + n * depth / 2, b - n * depth / 2, a - n * depth / 2])


def transform(plan: VectorPlan, fn) -> VectorPlan:
    return VectorPlan(
        walls=[fn(g) for g in plan.walls],
        doors=[fn(g) for g in plan.doors],
        windows=[fn(g) for g in plan.windows],
        rooms=[(fn(g), n) for g, n in plan.rooms],
        meta=dict(plan.meta),
    )


def axis_align(plan: VectorPlan) -> tuple[VectorPlan, float]:
    """Rotate the plan so its dominant wall direction is axis-parallel. Returns the new plan and
    the rotation applied (degrees)."""
    a = dominant_angle(plan.walls)
    if abs(a) < 1e-3:
        return plan, 0.0
    return transform(plan, lambda g: affinity.rotate(g, -a, origin=(0, 0))), -a


def _clean(g):
    return g if g.is_valid else g.buffer(0)


# --------------------------------------------------------------------------------------------
# walls: polygons -> rectangles
# --------------------------------------------------------------------------------------------


def rect_of_polygon(p: Polygon, min_fill: float = 0.9) -> Rect | None:
    """The polygon as one wall rectangle if it fills its minimum rotated rectangle."""
    mrr = p.minimum_rotated_rectangle
    if mrr.area <= 1e-9 or p.area / mrr.area < min_fill:
        return None
    c, u, length, width = _rect_axes(p)
    if width < 0.03:
        return None
    return Rect(c - u * length / 2, c + u * length / 2, width)


def _cluster(vals: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def decompose_rectilinear(poly: Polygon) -> list[Rect] | None:
    """Cut a rectilinear polygon (in its own frame) into thin horizontal and vertical strips.
    ``None`` if its edges are not axis-parallel. Strips thicker than ``T_CAP`` are not walls."""
    ang = dominant_angle([poly])
    p = affinity.rotate(poly, -ang, origin=(0, 0)) if abs(ang) > 1e-6 else poly
    for ring in [p.exterior, *p.interiors]:
        d = np.diff(np.asarray(ring.coords), axis=0)
        if (np.minimum(abs(d[:, 0]), abs(d[:, 1])) > 0.01).any():
            return None
    xs = _cluster([c[0] for r in [p.exterior, *p.interiors] for c in r.coords], 0.02)
    ys = _cluster([c[1] for r in [p.exterior, *p.interiors] for c in r.coords], 0.02)
    import shapely

    shapely.prepare(p)
    filled = np.zeros((len(ys) - 1, len(xs) - 1), bool)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            filled[j, i] = p.contains(Point((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2))
    strips: list[
        list[float]
    ] = []  # [orientation(0 h, 1 v), lo_across, hi_across, lo_along, hi_along]
    for orient_, grid, across, along in ((0, filled, ys, xs), (1, filled.T, xs, ys)):
        found: list[list[float]] = []
        for j in range(grid.shape[0]):
            h = across[j + 1] - across[j]
            if h > T_CAP:
                continue
            i = 0
            while i < grid.shape[1]:
                if not grid[j, i]:
                    i += 1
                    continue
                k = i
                while k + 1 < grid.shape[1] and grid[j, k + 1]:
                    k += 1
                length = along[k + 1] - along[i]
                if length >= max(1.5 * h, 0.5):
                    found.append([orient_, across[j], across[j + 1], along[i], along[k + 1]])
                i = k + 1
        # merge strips that stack on each other with the same extent (one wall, several rows)
        found.sort(key=lambda s: (s[3], s[4], s[1]))
        merged: list[list[float]] = []
        for s in found:
            if (
                merged
                and abs(merged[-1][3] - s[3]) < 0.03
                and abs(merged[-1][4] - s[4]) < 0.03
                and abs(merged[-1][2] - s[1]) < 0.03
                and s[2] - merged[-1][1] <= T_CAP
            ):
                merged[-1][2] = s[2]
            else:
                merged.append(list(s))
        strips += merged
    rects = []
    for o, a0, a1, l0, l1 in strips:
        mid = (a0 + a1) / 2
        pa, pb = ((l0, mid), (l1, mid)) if o == 0 else ((mid, l0), (mid, l1))
        rects.append(Rect(np.array(pa, float), np.array(pb, float), a1 - a0))
    if abs(ang) > 1e-6:
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        rot = np.array([[c, -s], [s, c]])
        for r in rects:
            r.a, r.b = rot @ r.a, rot @ r.b
    return rects


def _parallel(u: np.ndarray, v: np.ndarray, cos_tol: float = 0.9994) -> bool:
    return abs(float(u @ v)) >= cos_tol


def merge_collinear(rects: list[Rect], gap: float = 0.05) -> list[Rect]:
    """Merge collinear, overlapping or touching rectangles of similar thickness."""
    rects = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                r, s = rects[i], rects[j]
                if not _parallel(r.u, s.u):
                    continue
                if abs(r.t - s.t) > 0.4 * max(r.t, s.t):
                    continue
                u, n = r.u, np.array([-r.u[1], r.u[0]])
                off = abs(float((s.a - r.a) @ n)), abs(float((s.b - r.a) @ n))
                if max(off) > 0.35 * min(r.t, s.t):
                    continue
                ra = sorted([float((r.a - r.a) @ u), float((r.b - r.a) @ u)])
                sa = sorted([float((s.a - r.a) @ u), float((s.b - r.a) @ u)])
                if sa[0] > ra[1] + gap or ra[0] > sa[1] + gap:
                    continue
                lo, hi = min(ra[0], sa[0]), max(ra[1], sa[1])
                w = r.length / (r.length + s.length)
                shift = n * (w * float((r.a - r.a) @ n) + (1 - w) * float((s.a - r.a) @ n))
                new = Rect(
                    r.a + u * lo + shift,
                    r.a + u * hi + shift,
                    w * r.t + (1 - w) * s.t,
                    bridge=r.bridge and s.bridge,
                )
                rects[i] = new
                del rects[j]
                changed = True
                break
            if changed:
                break
    return rects


def snap_ends(rects: list[Rect]) -> None:
    """Move every free wall end onto the centreline of the wall it meets (L, T corners) and
    record the body extension (half the other thickness) for the drawing."""
    polys = [r.polygon() for r in rects]
    for i, r in enumerate(rects):
        for end in ("a", "b"):
            p = getattr(r, end)
            best = None
            for j, s in enumerate(rects):
                if j == i or _parallel(r.u, s.u, 0.94):
                    continue
                d = polys[j].distance(Point(*p))
                if d <= SNAP and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                continue
            s = rects[best[1]]
            # intersection of the two infinite centrelines
            m = np.array([r.u, -s.u]).T
            try:
                k = np.linalg.solve(m, s.a - r.a)
            except np.linalg.LinAlgError:
                continue
            q = r.a + r.u * k[0]
            if np.linalg.norm(q - p) <= s.t + 0.15:
                setattr(r, end, q)
                setattr(r, "e" + end, s.t / 2)


# --------------------------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------------------------


@dataclass
class Built:
    """Result of :func:`build_graph`: the GT plus what the renderer needs."""

    graph: PlanGraph
    rects: list[Rect]
    pieces: list[tuple[np.ndarray, np.ndarray, Rect]]
    open_boxes: list[tuple[OpeningKind, np.ndarray, np.ndarray, float, float]]
    """(kind, centre, unit direction along the wall, width, wall thickness) per GT opening."""
    rooms: list[tuple[Polygon, str | None]]
    stats: dict


def _wall_rects(plan: VectorPlan, stats: dict) -> list[Rect]:
    """Wall polygons + opening bridges -> merged rectangles. Fills ``stats``."""
    walls = [_clean(w) for w in plan.walls if not w.is_empty]
    walls_u = unary_union(walls) if walls else Polygon()
    total = float(walls_u.area)
    rects: list[Rect] = []
    lost = 0.0
    if plan.meta.get("union_walls"):
        # ResPlan: one big union; add the opening boxes so that the walls stay continuous
        gaps = [opening_box(g, walls_u) for g in plan.doors + plan.windows if not g.is_empty]
        u = unary_union([walls_u, *[g.buffer(0.005) for g in gaps]])
        for g in _polys(u):
            rs = decompose_rectilinear(g)
            if rs is None:
                lost += g.area
            else:
                rects += rs
                lost += max(0.0, g.area - unary_union([r.polygon() for r in rs]).area)
    else:
        for w in walls:
            for g in _polys(w):
                r = rect_of_polygon(g)
                if r is not None:
                    rects.append(r)
                    continue
                rs = decompose_rectilinear(g)
                if rs is None:
                    lost += g.area
                else:
                    rects += rs
                    lost += max(0.0, g.area - unary_union([x.polygon() for x in rs]).area)
        for g in plan.doors + plan.windows:  # bridges: a wall under every opening
            b = opening_box(g, walls_u)
            c, u, length, width = _rect_axes(b)
            rects.append(Rect(c - u * length / 2, c + u * length / 2, width, bridge=True))
    rects = [r for r in rects if r.length >= 0.15]
    rects = merge_collinear(rects)
    body = unary_union([r.polygon() for r in rects if not r.bridge]) if rects else Polygon()
    rects = [r for r in rects if not (r.bridge and body.contains(Point(*(r.a + r.b) / 2)))]
    for r in rects:
        r.bridge = False
    stats["wall_area_m2"] = round(total, 2)
    stats["unsupported_wall_share"] = round(lost / total, 4) if total > 0 else 1.0
    return rects


def _cluster_points(points: list[np.ndarray], tol: float) -> tuple[np.ndarray, list[int]]:
    nodes: list[np.ndarray] = []
    ids = []
    for p in points:
        if nodes:
            d = np.linalg.norm(np.array(nodes) - p, axis=1)
            k = int(d.argmin())
            if d[k] <= tol:
                ids.append(k)
                continue
        nodes.append(p)
        ids.append(len(nodes) - 1)
    return np.array(nodes), ids


def build_graph(
    plan: VectorPlan, *, m_per_px: float, image_height_px: int, stats: dict | None = None
) -> Built:
    """Wall rectangles, nodes, noded walls, openings and rooms of a plan, as a PlanGraph."""
    stats = {} if stats is None else stats
    rects = _wall_rects(plan, stats)
    snap_ends(rects)
    pieces: list[tuple[np.ndarray, np.ndarray, Rect]] = []
    if rects:
        noded = unary_union([LineString([r.a, r.b]) for r in rects])
        for g in getattr(noded, "geoms", [noded]):
            c = np.asarray(g.coords)
            a, b = c[0], c[-1]
            if np.linalg.norm(b - a) < 0.05:
                continue
            mid = (a + b) / 2
            dist = [LineString([r.a, r.b]).distance(Point(*mid)) for r in rects]
            pieces.append((a, b, rects[int(np.argmin(dist))]))
    pts = [p for a, b, _ in pieces for p in (a, b)]
    nodes, ids = _cluster_points(pts, SNAP) if pts else (np.zeros((0, 2)), [])
    footprint = unary_union([r.polygon() for r in rects] + [g for g, _ in plan.rooms])
    footprint = (
        unary_union([Polygon(g.exterior) for g in _polys(footprint)]) if rects else Polygon()
    )
    walls: list[Wall] = []
    kept: list[tuple[np.ndarray, np.ndarray, Rect]] = []
    seen: dict[frozenset, int] = {}
    for k, (a, b, r) in enumerate(pieces):
        s, e = ids[2 * k], ids[2 * k + 1]
        if s == e:
            continue
        key = frozenset((s, e))
        if key in seen:
            if r.t > walls[seen[key]].thickness_m:
                walls[seen[key]].thickness_m = round(r.t, 3)
            continue
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        mid = (a + b) / 2
        outside = any(
            not footprint.contains(Point(*(mid + sgn * n * (r.t / 2 + 0.12)))) for sgn in (1, -1)
        )
        kind = (
            WallKind.EXTERIOR
            if outside
            else WallKind.DRYWALL
            if r.t < HATCH_MIN_T
            else WallKind.INTERIOR
        )
        seen[key] = len(walls)
        walls.append(Wall(id=len(walls), start=s, end=e, thickness_m=round(r.t, 3), kind=kind))
        kept.append((nodes[s], nodes[e], r))
    graph_nodes = [Node(id=i, x=float(p[0]), y=float(p[1])) for i, p in enumerate(nodes)]

    # openings
    openings: list[Opening] = []
    open_boxes = []
    walls_u = unary_union([r.polygon() for r in rects]) if rects else Polygon()
    n_dropped = 0
    taken: list[tuple[int, float, float]] = []
    for kind, geoms in ((OpeningKind.DOOR, plan.doors), (OpeningKind.WINDOW, plan.windows)):
        for g in geoms:
            if g.is_empty:
                continue
            c, u, length, _ = _rect_axes(opening_box(g, walls_u))
            best = None
            for wid, (a, b, r) in enumerate(kept):
                wl = float(np.linalg.norm(b - a))
                d_ = (b - a) / wl
                if not _parallel(d_, u, 0.94):
                    continue
                s_c = float((c - a) @ d_)
                perp = abs(float((c - a) @ np.array([-d_[1], d_[0]])))
                if (
                    -0.05 <= s_c <= wl + 0.05
                    and perp <= r.t / 2 + 0.1
                    and (best is None or perp < best[0])
                ):
                    best = (perp, wid, s_c, wl, d_)
            if best is None:
                n_dropped += 1
                continue
            _, wid, s_c, wl, d_ = best
            off = max(0.0, s_c - length / 2)
            width = min(length, wl - off)
            if width < 0.3 or any(
                w_ == wid and off < e_ - 0.05 and s_ < off + width - 0.05 for w_, s_, e_ in taken
            ):
                n_dropped += 1
                continue
            taken.append((wid, off, off + width))
            a, b, r = kept[wid]
            openings.append(
                Opening(
                    id=len(openings), kind=kind, wall_id=wid, offset_m=round(off, 3),
                    width_m=round(width, 3),
                    sill_m=0.0 if kind == OpeningKind.DOOR else None,
                    height_m=DOOR_HEIGHT if kind == OpeningKind.DOOR else None,
                )
            )  # fmt: skip
            open_boxes.append((kind, a + d_ * (off + width / 2), d_, width, r.t))
    stats["n_openings_dropped"] = n_dropped

    # rooms
    rooms: list[Room] = []
    drawn_rooms: list[tuple[Polygon, str | None]] = []
    for g, name in plan.rooms:
        parts = sorted(_polys(_clean(g)), key=lambda q: -q.area)
        if not parts or parts[0].area < MIN_ROOM_M2:
            continue
        poly = orient(Polygon(parts[0].exterior).simplify(0.01), 1.0)
        if poly.is_empty or poly.area < MIN_ROOM_M2:
            continue
        area = round(poly.area, 3)
        rooms.append(
            Room(
                id=len(rooms), polygon=[(float(x), float(y)) for x, y in poly.exterior.coords[:-1]],
                name=name, area_m2=area, stamp_area_m2=round(area, 2),
            )
        )  # fmt: skip
        drawn_rooms.append((poly, name))
    graph = PlanGraph(
        nodes=graph_nodes, walls=walls, openings=openings, rooms=rooms,
        m_per_px=m_per_px, scale_source="ground_truth", image_height_px=image_height_px,
    )  # fmt: skip
    graph.report = closure_metrics(graph)
    stats.update(n_walls=len(walls), n_openings=len(openings), n_rooms=len(rooms))
    return Built(graph, rects, kept, open_boxes, drawn_rooms, stats)


# --------------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------------


def _draw_door(cv: _Canvas, centre, u, width, t, rooms_u, rng) -> None:
    n = np.array([-u[1], u[0]])
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rooms_u is not None and not rooms_u.is_empty:
        pos = rooms_u.contains(Point(*(centre + n * (t / 2 + 0.3))))
        neg = rooms_u.contains(Point(*(centre - n * (t / 2 + 0.3))))
        if pos != neg:
            side = 1.0 if pos else -1.0
    hinge_end = -1.0 if rng.random() < 0.5 else 1.0
    face = centre + side * n * t / 2
    hinge = face + hinge_end * u * width / 2
    other = face - hinge_end * u * width / 2
    leaf = hinge + side * n * width
    cv.line(tuple(hinge), tuple(leaf), LEAF_LINE_MM)
    a0 = math.atan2(leaf[1] - hinge[1], leaf[0] - hinge[0])
    a1 = math.atan2(other[1] - hinge[1], other[0] - hinge[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    arc = [
        (
            hinge[0] + width * math.cos(a0 + da * k / 24),
            hinge[1] + width * math.sin(a0 + da * k / 24),
        )
        for k in range(25)
    ]
    cv.polyline(arc, THIN_LINE_MM)


def _draw_window(cv: _Canvas, centre, u, width, t, rng) -> None:
    n = np.array([-u[1], u[0]])
    offs = (-0.38, 0.0, 0.38) if rng.random() < 0.4 else (-0.12, 0.12)
    for o in offs:
        cv.line(
            tuple(centre - u * width / 2 + n * o * t), tuple(centre + u * width / 2 + n * o * t),
            THIN_LINE_MM,
        )  # fmt: skip


def _stops(vals: list[float], lo: float, hi: float, min_gap: float = 0.35) -> list[float]:
    """Chain stops: the overall ends plus ``vals`` at least ``min_gap`` apart."""
    out = [lo]
    for v in sorted(vals):
        if v - out[-1] >= min_gap and hi - v >= min_gap:
            out.append(v)
    return [*out, hi]


def _draw_chain(cv: _Canvas, stops, c, horizontal, outward, length_style) -> None:
    def P(s, across):
        return (s, across) if horizontal else (across, s)

    tick = TICK_LEN_MM / 1000 * 100 / 2
    gap = cv.mm(0.7)
    cv.line(P(stops[0] - 0.12, c), P(stops[-1] + 0.12, c), THIN_LINE_MM)
    for s in stops:
        cv.line(P(s, c - outward * 0.7), P(s, c + outward * 0.12), THIN_LINE_MM)
        cv.line(P(s - tick, c - tick), P(s + tick, c + tick), TICK_LINE_MM)
    for a, b in zip(stops[:-1], stops[1:], strict=True):
        text = format_length(round(b - a, 4), length_style)
        pa, pb = cv.px(P(a, c)), cv.px(P(b, c))
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        angle = 0.0 if horizontal else 90.0
        tw, th = cv.text_size(text, DIM_TEXT_MM, angle)
        centre = (mid[0], mid[1] - gap - th / 2) if horizontal else (mid[0] - gap - tw / 2, mid[1])
        bbox = (centre[0] - tw / 2, centre[1] - th / 2, centre[0] + tw / 2, centre[1] + th / 2)
        if not cv.collides(bbox):
            cv.text(text, centre, DIM_TEXT_MM, angle)


def _draw_stamp(cv: _Canvas, poly: Polygon, name: str | None) -> None:
    if not name or poly.area < MIN_ROOM_M2:
        return
    area = format_area(poly.area)
    c = polylabel(poly, 0.05)
    cx, cy = cv.px((c.x, c.y))
    for k in (1.0, 0.85, 0.7):
        nw, nh = cv.text_size(name, NAME_TEXT_MM * k)
        aw, ah = cv.text_size(area, AREA_TEXT_MM * k)
        hw, hh = max(nw, aw) / 2 * cv.m_per_px, (nh + ah + cv.mm(1.2 * k)) / 2 * cv.m_per_px
        if poly.contains(box(c.x - hw, c.y - hh, c.x + hw, c.y + hh)) or k == 0.7:
            break
    total = nh + cv.mm(1.2 * k) + ah
    c_name, c_area = (cx, cy - total / 2 + nh / 2), (cx, cy + total / 2 - ah / 2)
    bbox = (cx - max(nw, aw) / 2, cy - total / 2, cx + max(nw, aw) / 2, cy + total / 2)
    if cv.collides(bbox, pad=1.0):
        return
    cv.text(name, c_name, NAME_TEXT_MM * k)
    cv.text(area, c_area, AREA_TEXT_MM * k)


def render(built: Built, size: tuple[int, int], rng: np.random.Generator):
    """Draw the GT as a German 1:100 plan. Returns the grayscale image and the wall mask."""
    length_style = str(rng.choice(["cm", "sup", "m"]))
    cv = _Canvas(size[0], size[1], DPI, M_PER_PX, Style(length=length_style))  # type: ignore[arg-type]
    hatched, plain = [], []
    for r in built.rects:
        (hatched if r.t >= HATCH_MIN_T else plain).append(r.polygon())
    hat_u = unary_union(hatched) if hatched else Polygon()
    plain_u = unary_union(plain).difference(hat_u) if plain else Polygon()
    cuts = []
    for _, c, u, w, t in built.open_boxes:
        n = np.array([-u[1], u[0]])
        h = t / 2 + 0.01
        cuts.append(
            Polygon(
                [
                    c - u * w / 2 + n * h,
                    c + u * w / 2 + n * h,
                    c + u * w / 2 - n * h,
                    c - u * w / 2 - n * h,
                ]
            )
        )
    cut_u = unary_union(cuts) if cuts else Polygon()
    hat_u, plain_u = hat_u.difference(cut_u), plain_u.difference(cut_u)
    hmask, pmask = cv.poly_mask(hat_u), cv.poly_mask(plain_u)
    cv.hatch(hmask)
    cv.outline(hat_u, WALL_LINE_MM)
    cv.outline(plain_u, WALL_LINE_MM)
    rooms_u = unary_union([g for g, _ in built.rooms]) if built.rooms else None
    for kind, c, u, w, t in built.open_boxes:
        if kind == OpeningKind.DOOR:
            _draw_door(cv, c, u, w, t, rooms_u, rng)
        else:
            _draw_window(cv, c, u, w, t, rng)
    for poly, name in built.rooms:
        _draw_stamp(cv, poly, name)
    walls_u = unary_union(hatched + plain)
    x0, y0, x1, y1 = walls_u.bounds
    vx, hy = [], []  # faces of vertical / horizontal walls
    for r in built.rects:
        if abs(r.u[0]) < 0.03:
            vx += [float(r.a[0]) - r.t / 2, float(r.a[0]) + r.t / 2]
        elif abs(r.u[1]) < 0.03:
            hy += [float(r.a[1]) - r.t / 2, float(r.a[1]) + r.t / 2]
    for horizontal, (lo, hi), vals in ((True, (x0, x1), vx), (False, (y0, y1), hy)):
        stops = _stops(vals, lo, hi)
        for outward, face in ((-1.0, y0 if horizontal else x0), (1.0, y1 if horizontal else x1)):
            _draw_chain(cv, stops, face + outward * 0.9, horizontal, outward, length_style)
            _draw_chain(cv, [lo, hi], face + outward * 1.6, horizontal, outward, length_style)
    title = "Grundriss Erdgeschoss  M 1:100"
    tw, th = cv.text_size(title, NAME_TEXT_MM)
    tl = cv.px((x0, y0 - 2.5))
    cv.text(title, (tl[0] + tw / 2, tl[1] + th / 2), NAME_TEXT_MM)
    return cv.img, hmask | pmask


@dataclass
class VectorItem:
    """One rendered plan: degraded and clean page, GT graph and bookkeeping."""

    image: np.ndarray
    clean: np.ndarray
    graph: PlanGraph
    stats: dict


def render_plan(plan: VectorPlan, *, seed: int, degrade: float) -> VectorItem | None:
    """Axis-align, convert to GT, draw and degrade one plan (deterministic in ``seed``). ``None``
    if the plan is unusable (extent outside 3-60 m, too few walls or rooms, too much wall area
    that the GT cannot represent)."""
    rng = np.random.default_rng([int(seed), 0x6F6F6431])
    plan, rot = axis_align(plan)
    walls_u = unary_union([_clean(w) for w in plan.walls if not w.is_empty])
    if walls_u.is_empty:
        return None
    x0, y0, x1, y1 = walls_u.bounds
    ext = max(x1 - x0, y1 - y0)
    if ext > MAX_EXTENT_M or min(x1 - x0, y1 - y0) < MIN_EXTENT_M:
        return None
    dx, dy = MARGIN_M - x0, MARGIN_M - y0
    plan = transform(plan, lambda g: affinity.translate(g, dx, dy))
    w_px = int(math.ceil((x1 - x0 + 2 * MARGIN_M) / M_PER_PX))
    h_px = int(math.ceil((y1 - y0 + 2 * MARGIN_M) / M_PER_PX))
    stats = {"rotation_deg": round(rot, 3), "extent_m": round(ext, 1), **plan.meta.get("stats", {})}
    built = build_graph(plan, m_per_px=M_PER_PX, image_height_px=h_px, stats=stats)
    if (
        len(built.graph.walls) < 4
        or len(built.graph.rooms) < 2
        or stats["unsupported_wall_share"] > MAX_UNSUPPORTED
    ):
        return None
    clean, _ = render(built, (w_px, h_px), rng)
    image, info = _degrade(clean, float(degrade), np.random.default_rng([int(seed), 0x64656772]))
    stats["degradation"] = {"strength": info.strength, "rotation_deg": round(info.rotation_deg, 3)}
    return VectorItem(image, clean, built.graph, stats)


__all__ = [
    "Built",
    "Rect",
    "VectorItem",
    "VectorPlan",
    "axis_align",
    "build_graph",
    "m_to_px",
    "opening_box",
    "render_plan",
]
