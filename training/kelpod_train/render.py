"""Re-render vector floor plans from public datasets in Kelpod's DIN 1356-1 drawing style.

A :class:`VectorPlan` holds plan-space polygons (metres, y up): wall bodies, openings (door /
window boxes spanning the wall), rooms (with a German name for the stamp), fixtures (sanitary
objects, kitchen units, stairs: drawn as thin outlines, labelled as room interior) and railings.
:func:`render_sample` draws it like ``kelpod.evaluation.synth`` draws its own plans (hatched
masonry, plain partitions, door leaf + quarter arc, 2-3 window lines, room stamps, an overall
dimension chain), degrades it like a scan and returns a cleaned 200-dpi page with its class map.

This reuses the synth generator's canvas, text renderer and degradation, so dataset plans and
synthetic plans look alike (same pens, fonts, hatching and scan artefacts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from kelpod.evaluation.synth.render import (
    AREA_TEXT_MM,
    HATCH_SPACING_MM,
    LEAF_LINE_MM,
    NAME_TEXT_MM,
    THIN_LINE_MM,
    TICK_LEN_MM,
    TICK_LINE_MM,
    WALL_LINE_MM,
    _Canvas,
    _polys,
)
from kelpod.evaluation.synth.text import format_area, format_length
from kelpod_train.labels import (
    BG,
    DOOR,
    IGNORE,
    ROOM,
    WALL,
    WINDOW,
    Sample,
    random_degrade,
    to_page,
)

SRC_DPI = 300.0
M_PER_PX = 0.0254 / SRC_DPI * 100.0
MARGIN_M = 3.5
MAX_EXTENT_M = 60.0


@dataclass
class VectorPlan:
    walls: list[Polygon]
    doors: list[Polygon] = field(default_factory=list)
    windows: list[Polygon] = field(default_factory=list)
    rooms: list[tuple[Polygon, str | None]] = field(default_factory=list)
    """Room polygon and German stamp name (``None``: no stamp)."""
    ignore: list[Polygon] = field(default_factory=list)
    """Areas whose room label is unknown (balconies, terraces): labelled 255."""
    fixtures: list[Polygon] = field(default_factory=list)
    railings: list = field(default_factory=list)
    columns: list[Polygon] = field(default_factory=list)


def dominant_angle(polys: list[Polygon]) -> float:
    """Dominant edge direction modulo 90 degrees (length-weighted circular mean)."""
    acc = 0j
    for p in polys:
        for g in _polys(p):
            c = np.asarray(g.exterior.coords)
            d = np.diff(c, axis=0)
            ln = np.hypot(d[:, 0], d[:, 1])
            ang = np.arctan2(d[:, 1], d[:, 0])
            acc += np.sum(ln * np.exp(4j * ang))
    return math.degrees(np.angle(acc) / 4) if abs(acc) > 0 else 0.0


def transform(plan: VectorPlan, fn) -> VectorPlan:
    def f(gs):
        return [fn(g) for g in gs]

    return VectorPlan(
        walls=f(plan.walls),
        doors=f(plan.doors),
        windows=f(plan.windows),
        rooms=[(fn(g), n) for g, n in plan.rooms],
        ignore=f(plan.ignore),
        fixtures=f(plan.fixtures),
        railings=f(plan.railings),
        columns=f(plan.columns),
    )


def axis_align(plan: VectorPlan) -> VectorPlan:
    a = dominant_angle(plan.walls)
    if abs(a) < 1e-3:
        return plan
    return transform(plan, lambda g: affinity.rotate(g, -a, origin=(0, 0)))


def _thickness(p: Polygon) -> float:
    """Rough width of an elongated polygon: 2 * area / perimeter."""
    return 2 * p.area / max(p.length, 1e-9)


def _rect_axes(p: Polygon):
    """Centre, unit long axis, length and width of a polygon's minimum rotated rectangle."""
    r = p.minimum_rotated_rectangle
    c = np.asarray(r.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 >= l2:
        return c.mean(0), e1 / max(l1, 1e-9), l1, l2
    return c.mean(0), e2 / max(l2, 1e-9), l2, l1


def _draw_door(cv: _Canvas, door: Polygon, rooms_u, rng) -> None:
    centre, u, length, width = _rect_axes(door)
    n = np.array([-u[1], u[0]])
    # swing into the side that is room interior (else random)
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rooms_u is not None and not rooms_u.is_empty:
        pos = rooms_u.contains(Point(*(centre + n * (width / 2 + 0.3))))
        neg = rooms_u.contains(Point(*(centre - n * (width / 2 + 0.3))))
        if pos != neg:
            side = 1.0 if pos else -1.0
    hinge_end = -1.0 if rng.random() < 0.5 else 1.0
    face = centre + side * n * width / 2
    hinge = face + hinge_end * u * length / 2
    other = face - hinge_end * u * length / 2
    leaf = hinge + side * n * length
    cv.line(tuple(hinge), tuple(leaf), LEAF_LINE_MM)
    a0 = math.atan2(leaf[1] - hinge[1], leaf[0] - hinge[0])
    a1 = math.atan2(other[1] - hinge[1], other[0] - hinge[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    arc = [
        (hinge[0] + length * math.cos(a0 + da * k / 20), hinge[1] + length * math.sin(a0 + da * k / 20))
        for k in range(21)
    ]
    cv.polyline(arc, THIN_LINE_MM)


def _draw_window(cv: _Canvas, win: Polygon, rng) -> None:
    centre, u, length, width = _rect_axes(win)
    n = np.array([-u[1], u[0]])
    offs = (-0.38, 0.0, 0.38) if rng.random() < 0.4 else (-0.12, 0.12)
    for o in offs:
        a = centre - u * length / 2 + n * o * width
        b = centre + u * length / 2 + n * o * width
        cv.line(tuple(a), tuple(b), THIN_LINE_MM)


def _draw_chain(cv: _Canvas, xs: list[float], y: float, horizontal: bool, style: str) -> None:
    """A simple dimension chain with 45° ticks and German numbers above the line."""
    xs = sorted(set(round(x, 3) for x in xs))
    if len(xs) < 2:
        return

    def P(a, b):
        return (a, b) if horizontal else (b, a)

    cv.line(P(xs[0] - 0.3, y), P(xs[-1] + 0.3, y), THIN_LINE_MM)
    t = TICK_LEN_MM * 0.01 / 2  # tick half-length in plan metres at 1:100
    for x in xs:
        cv.line(P(x - t, y - t), P(x + t, y + t), TICK_LINE_MM)
    for a, b in zip(xs[:-1], xs[1:], strict=False):
        if b - a < 0.3:
            continue
        text = format_length(b - a, style)
        c = cv.px(P((a + b) / 2, y + 0.18))
        cv.text(text, c, 2.5, 0.0 if horizontal else 90.0)


def render_vector(plan: VectorPlan, rng: np.random.Generator):
    """Draw a plan (already axis-aligned, metres) at 300 dpi. Returns (image, label) or None."""
    walls_u = unary_union([w.buffer(0) for w in plan.walls if w.is_valid or w.buffer(0).is_valid])
    if walls_u.is_empty:
        return None
    x0, y0, x1, y1 = walls_u.bounds
    if max(x1 - x0, y1 - y0) > MAX_EXTENT_M or min(x1 - x0, y1 - y0) < 2.0:
        return None
    # shift so the plan sits inside the margin (origin bottom-left of the image)
    dx, dy = MARGIN_M - x0, MARGIN_M - y0
    plan = transform(plan, lambda g: affinity.translate(g, dx, dy))
    walls_u = affinity.translate(walls_u, dx, dy)
    W = int(math.ceil((x1 - x0 + 2 * MARGIN_M) / M_PER_PX))
    Hh = int(math.ceil((y1 - y0 + 2 * MARGIN_M) / M_PER_PX))
    cv = _Canvas(W, Hh, SRC_DPI, M_PER_PX)

    cuts = unary_union([g.buffer(0.005) for g in plan.doors + plan.windows])
    hatch_min = float(rng.uniform(0.16, 0.22))
    hatched, plain = [], []
    for g in _polys(walls_u):
        (hatched if _thickness(g) >= hatch_min else plain).append(g)
    hatched_u = unary_union(hatched).difference(cuts) if hatched else Polygon()
    plain_u = unary_union(plain).difference(cuts) if plain else Polygon()
    style_solid = rng.random() < 0.15  # some old plans fill masonry solid instead of hatching
    hmask = cv.poly_mask(hatched_u)
    if style_solid:
        cv.img[hmask] = 40
    else:
        cv.hatch(hmask)
    cv.outline(hatched_u, WALL_LINE_MM)
    cv.outline(plain_u, WALL_LINE_MM)
    for c in plan.columns:
        m = cv.poly_mask(c)
        cv.img[m] = 30
    rooms_u = unary_union([r for r, _ in plan.rooms]) if plan.rooms else None
    for d in plan.doors:
        _draw_door(cv, d, rooms_u, rng)
    for w in plan.windows:
        _draw_window(cv, w, rng)
    for f in plan.fixtures:
        cv.outline(f, THIN_LINE_MM)
    for r in plan.railings:
        for g in _polys(r) if hasattr(r, "exterior") or hasattr(r, "geoms") else []:
            cv.outline(g, THIN_LINE_MM)
    # room stamps
    for poly, name in plan.rooms:
        if not name or poly.area < 2.0:
            continue
        c = poly.representative_point()
        cx, cy = cv.px((c.x, c.y))
        area = format_area(poly.area)
        nw, nh = cv.text_size(name, NAME_TEXT_MM)
        aw, ah = cv.text_size(area, AREA_TEXT_MM)
        bx = (cx - max(nw, aw) / 2, cy - nh, cx + max(nw, aw) / 2, cy + ah + 4)
        if cv.collides(bx):
            continue
        cv.text(name, (cx, cy - nh / 2), NAME_TEXT_MM)
        cv.text(area, (cx, cy + ah / 2 + 4), AREA_TEXT_MM)
    # overall dimension chains on two or four sides (exterior face stops)
    bx0, by0, bx1, by1 = walls_u.bounds
    style = str(rng.choice(["cm", "sup", "m"]))
    ext = walls_u.exterior if isinstance(walls_u, Polygon) else None
    xs = [bx0, bx1] + ([p[0] for p in ext.coords] if ext is not None else [])
    ys = [by0, by1] + ([p[1] for p in ext.coords] if ext is not None else [])
    sides = ("s", "w") if rng.random() < 0.5 else ("s", "w", "n", "e")
    for s in sides:
        if s in "sn":
            yy = by0 - 1.0 if s == "s" else by1 + 1.0
            _draw_chain(cv, [bx0, bx1], yy - (0.7 if s == "s" else -0.7), True, style)
            _draw_chain(cv, xs, yy, True, style)
        else:
            xx = bx0 - 1.0 if s == "w" else bx1 + 1.0
            _draw_chain(cv, [by0, by1], xx - (0.7 if s == "w" else -0.7), False, style)
            _draw_chain(cv, ys, xx, False, style)

    # labels in the same px grid
    lab = np.zeros(cv.img.shape, np.uint8)
    for poly, _ in plan.rooms:
        lab[cv.poly_mask(poly)] = ROOM
    for f in plan.fixtures:
        pass  # fixtures stand in rooms: keep the room label
    for g in plan.ignore:
        lab[cv.poly_mask(g) & (lab == BG)] = IGNORE
    lab[cv.poly_mask(hatched_u) | cv.poly_mask(plain_u)] = WALL
    for c in plan.columns:
        lab[cv.poly_mask(c)] = WALL
    for d in plan.doors:
        lab[cv.poly_mask(d)] = DOOR
    for w in plan.windows:
        lab[cv.poly_mask(w)] = WINDOW
    return cv.img, lab


def render_sample(plan: VectorPlan, *, seed: int) -> Sample | None:
    """Axis-align, draw, degrade and clean one plan (deterministic in ``seed``)."""
    rng = np.random.default_rng([int(seed), 0x72656E64])
    plan = axis_align(plan)
    out = render_vector(plan, rng)
    if out is None:
        return None
    img, lab = out
    strength = float(rng.choice([0.0, rng.uniform(0.2, 1.0), 1.0], p=[0.1, 0.7, 0.2]))
    deg, affine = random_degrade(img, lab, strength, seed)
    s = to_page(deg, lab, affine, SRC_DPI)
    s.meta.update(seed=seed, degrade=strength)
    return s


def opening_box(opening: Polygon, walls_u, min_depth: float = 0.08) -> Polygon:
    """Turn an opening polygon into the gap box in its wall: the axis along the wall is the
    rectangle axis whose jambs (just beyond both ends) hit wall; the depth across is the wall
    thickness measured at the jambs (the polygon's own width if no wall is found)."""
    centre, u0, l0, w0 = _rect_axes(opening)
    best = None
    for u, length, width in ((u0, l0, w0), (np.array([-u0[1], u0[0]]), w0, l0)):
        n = np.array([-u[1], u[0]])
        depths = []
        if walls_u is not None:
            for t in (-0.5, 0.5):
                c = centre + u * t * (length + 0.12)
                inter = LineString([c - n * 0.6, c + n * 0.6]).intersection(walls_u)
                # only the piece through the jamb itself (not a perpendicular wall nearby)
                for piece in getattr(inter, "geoms", [inter]):
                    if not piece.is_empty and piece.distance(Point(*c)) < 0.02:
                        if piece.length < 0.7:
                            depths.append(piece.length)
        score = (len(depths), length >= width)
        if best is None or score > best[0]:
            best = (score, u, n, length, max(depths) if depths else max(width, min_depth))
    _, u, n, length, depth = best
    depth = max(depth, min_depth)
    a = centre - u * length / 2
    b = centre + u * length / 2
    return Polygon([a + n * depth / 2, b + n * depth / 2, b - n * depth / 2, a - n * depth / 2])


__all__ = [
    "HATCH_SPACING_MM",
    "MultiPolygon",
    "VectorPlan",
    "box",
    "opening_box",
    "render_sample",
    "render_vector",
]
