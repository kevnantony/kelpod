"""Style-randomised renderer for training pages (experiment ``learned-v1``).

Extends :mod:`kelpod_train.render` (the DIN renderer for vector plans) with a
:class:`kelpod_train.style_v1.TrainStyle`: hatching variants, wall outline vs. none, line
weights per element, door and window symbol variants, fonts, and clutter (furniture, stairs, axis
grid, notes, stamps, title block, north arrow). Synthetic plans from ``kelpod.evaluation.synth``
are re-drawn through the same renderer (their geometry only, via :func:`synth_vector_plan`), so
synth, Swiss Dwellings and ResPlan pages all get the randomised styles.

Only train-style options are used (``style_v1`` checks them against the generator's held-out
flags). Labels are rasterised from the plan polygons in the same pixel grid as the drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from kelpod.evaluation.synth.render import (
    AREA_TEXT_MM,
    HATCH_LINE_MM,
    LEAF_LINE_MM,
    NAME_TEXT_MM,
    THIN_LINE_MM,
    WALL_LINE_MM,
    Style,
    _Canvas,
    _polys,
)
from kelpod.evaluation.synth.text import format_area
from kelpod_train.labels import BG, DOOR, IGNORE, ROOM, WALL, WINDOW
from kelpod_train.render import (
    M_PER_PX,
    MARGIN_M,
    MAX_EXTENT_M,
    SRC_DPI,
    VectorPlan,
    _draw_chain,
    _rect_axes,
    _thickness,
    axis_align,
    transform,
)
from kelpod_train.style_v1 import TRAIN_FONTS, TrainStyle

NOTES = ("Fliesen", "Parkett", "PVC", "Estrich", "LH 2,51", "LH 2,60", "OKFF +0,00", "Teppich",
         "Linol.", "Dielen", "Putz", "-0,15", "+2,75", "Kamin", "Abl.", "Schrank", "Hzg.",
         "neu", "Bestand", "abbrechen", "WC", "Dusche")
TITLES = ("Grundriss Erdgeschoss", "Grundriss Obergeschoss", "Kellergeschoss", "Dachgeschoss",
          "Bestandsplan", "Erdgeschoß M 1:100", "Wohnhaus", "Umbau", "Bauantrag")


@dataclass
class StyledPlan(VectorPlan):
    """A :class:`VectorPlan` whose light partitions are given explicitly (synth plans)."""

    plain_walls: list[Polygon] = field(default_factory=list)


class _StyledCanvas(_Canvas):
    def __init__(self, w: int, h: int, ts: TrainStyle) -> None:
        spec = getattr(ts, "font_spec", None) or TRAIN_FONTS[ts.font]
        if spec is not None and not spec.startswith("hershey:") and not Path(spec).is_file():
            spec = None  # font not installed on this node: the generator default
        super().__init__(w, h, SRC_DPI, M_PER_PX, Style(font=spec))
        self.ts = ts

    def scale(self, k: float) -> _StyledCanvas:
        self.style.line_scale = k
        return self

    def dashed(self, pts, width_mm: float, dash_mm: float = 2.0, gap_mm: float = 1.0,
               dot: bool = False) -> None:
        """Dashed (or dash-dot) polyline through plan-space points."""
        d = dash_mm * 0.1  # plan metres at 1:100
        g = gap_mm * 0.1
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            a, b = np.asarray(a, float), np.asarray(b, float)
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            u = (b - a) / L
            s = 0.0
            while s < L:
                e = min(s + d, L)
                self.line(tuple(a + u * s), tuple(a + u * e), width_mm)
                s = e + g
                if dot and s < L:
                    c = a + u * min(s + g * 0.3, L)
                    self.line(tuple(c), tuple(c + u * 0.01), width_mm)
                    s += g

    # ---------------------------------------------------------------- hatching
    def hatch_v1(self, mask: np.ndarray, rng: np.random.Generator) -> None:
        kind, spacing_mm = self.ts.hatch, self.ts.hatch_spacing_mm
        if kind == "none":
            return
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        h, w = y1 - y0, x1 - x0
        layer = np.full((h, w), 255, np.uint8)
        width = self.lw(HATCH_LINE_MM)
        sp = self.mm(spacing_mm)
        f = 1 << self.SHIFT
        if kind == "stipple":
            n = int(mask.sum() / (sp * sp) * 1.2)
            if n:
                yy, xx = ys[rng.integers(0, len(ys), n)], xs[rng.integers(0, len(xs), n)]
                r = max(1, int(round(self.mm(0.12) * self.style.line_scale)))
                for py, px in zip(yy, xx, strict=True):
                    if rng.random() < 0.15:  # small aggregate triangles
                        t = self.mm(0.5)
                        pts = np.array([[px, py - t], [px - t, py + t], [px + t, py + t]], np.int32)
                        cv2.polylines(layer, [pts - [x0, y0]], True, 0, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(layer, (int(px - x0), int(py - y0)), r, 0, -1, cv2.LINE_AA)
        else:
            ang = {"diag45": 45.0, "double45": 45.0, "dashed45": 45.0, "diag30": 30.0,
                   "diag60": 60.0, "horizontal": 0.0, "diag15": 15.0, "diag75": 75.0,
                   "vertical": 90.0, "diag20": 20.0, "diag70": 70.0, "diag45_sparse": 45.0}[kind]
            th = math.radians(ang)
            d = np.array([math.cos(th), -math.sin(th)])  # screen, rising to the right
            nrm = np.array([math.sin(th), math.cos(th)])
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
            cs = corners @ nrm
            c = math.floor(cs.min() / sp) * sp
            L = float(h + w)
            offs = [0.0] if kind != "double45" else [0.0, 0.3 * sp]
            while c <= cs.max() + sp:
                for o in offs:
                    p = nrm * (c + o) - np.array([x0, y0])
                    a, b = p - d * L * 2, p + d * L * 2
                    if kind == "dashed45":
                        dash = self.mm(1.5)
                        n_d = int(4 * L / dash)
                        for k in range(0, n_d, 2):
                            q0 = a + d * k * dash
                            q1 = a + d * (k + 1) * dash
                            cv2.line(layer, tuple(np.round(q0 * f).astype(int)),
                                     tuple(np.round(q1 * f).astype(int)), 0, width, cv2.LINE_AA,
                                     self.SHIFT)
                    else:
                        cv2.line(layer, tuple(np.round(a * f).astype(int)),
                                 tuple(np.round(b * f).astype(int)), 0, width, cv2.LINE_AA,
                                 self.SHIFT)
                c += sp
        crop = self.img[y0:y1, x0:x1]
        sub = mask[y0:y1, x0:x1]
        crop[sub] = np.minimum(crop[sub], layer[sub])


# -------------------------------------------------------------------- symbols


def _draw_door_v1(cv: _StyledCanvas, door: Polygon, rooms_u, rng, kind: str) -> None:
    centre, u, length, width = _rect_axes(door)
    n = np.array([-u[1], u[0]])
    side = 1.0 if rng.random() < 0.5 else -1.0
    if rooms_u is not None and not rooms_u.is_empty:
        pos = rooms_u.contains(Point(*(centre + n * (width / 2 + 0.3))))
        neg = rooms_u.contains(Point(*(centre - n * (width / 2 + 0.3))))
        if pos != neg:
            side = 1.0 if pos else -1.0
    hinge_end = -1.0 if rng.random() < 0.5 else 1.0
    face = centre + side * n * width / 2
    if kind == "gap":
        return
    if kind == "sliding":
        for k, o in ((-1, 0.18), (1, -0.18)):
            a = centre + u * (k * length * 0.05 - length / 2) + n * o * width
            b = a + u * length * 0.95
            cv.line(tuple(a), tuple(b), LEAF_LINE_MM)
        return

    def leaf_arc(hinge, other, ln, dashed=False, rect=False):
        leaf = hinge + side * n * ln
        if rect:
            t = (other - hinge) / max(np.linalg.norm(other - hinge), 1e-9) * 0.04
            cv.polyline([tuple(hinge), tuple(leaf), tuple(leaf + t), tuple(hinge + t)],
                        LEAF_LINE_MM, closed=True)
        else:
            cv.line(tuple(hinge), tuple(leaf), LEAF_LINE_MM)
        if kind == "leaf_only":
            return
        a0 = math.atan2(leaf[1] - hinge[1], leaf[0] - hinge[0])
        a1 = math.atan2(other[1] - hinge[1], other[0] - hinge[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        arc = [(hinge[0] + ln * math.cos(a0 + da * k / 20), hinge[1] + ln * math.sin(a0 + da * k / 20))
               for k in range(21)]
        if dashed:
            cv.dashed(arc, THIN_LINE_MM, 1.2, 0.8)
        else:
            cv.polyline(arc, THIN_LINE_MM)

    if kind == "double":
        left = face - u * length / 2
        right = face + u * length / 2
        leaf_arc(left, face, length / 2)
        leaf_arc(right, face, length / 2)
        return
    hinge = face + hinge_end * u * length / 2
    other = face - hinge_end * u * length / 2
    leaf_arc(hinge, other, length, dashed=kind == "dashed_arc", rect=kind == "rect_leaf")


def _draw_window_v1(cv: _StyledCanvas, win: Polygon, rooms_u, rng, kind: str) -> None:
    centre, u, length, width = _rect_axes(win)
    n = np.array([-u[1], u[0]])
    a0, a1 = centre - u * length / 2, centre + u * length / 2

    def ln(o, ext=0.0):
        cv.line(tuple(a0 - u * ext + n * o * width), tuple(a1 + u * ext + n * o * width),
                THIN_LINE_MM)

    if kind == "single":
        ln(0.0)
    elif kind == "two":
        ln(-0.12)
        ln(0.12)
    elif kind == "three":
        for o in (-0.38, 0.0, 0.38):
            ln(o)
    elif kind == "four":
        for o in (-0.3, -0.1, 0.1, 0.3):
            ln(o)
    elif kind == "frame":
        p = [a0 + n * 0.25 * width + u * 0.02, a1 + n * 0.25 * width - u * 0.02,
             a1 - n * 0.25 * width - u * 0.02, a0 - n * 0.25 * width + u * 0.02]
        cv.polyline([tuple(q) for q in p], THIN_LINE_MM, closed=True)
        ln(0.0)
    elif kind == "sill":
        ln(-0.12)
        ln(0.12)
        out = 1.0
        if rooms_u is not None and not rooms_u.is_empty:
            out = -1.0 if rooms_u.contains(Point(*(centre + n * (width / 2 + 0.3)))) else 1.0
        o = out * (0.5 + 0.06 / max(width, 0.05))  # sill projects ~6 cm beyond the face
        ln(o, ext=0.05)


# -------------------------------------------------------------------- clutter


def _draw_clutter(cv: _StyledCanvas, plan: VectorPlan, walls_u, rng: np.random.Generator) -> None:
    k = cv.ts.clutter
    if k <= 0:
        return
    for poly, _ in plan.rooms:
        if poly.area < 4 or rng.random() > 0.6 * k:
            continue
        inner = poly.buffer(-0.15)
        if inner.is_empty:
            continue
        bx0, by0, bx1, by1 = inner.bounds
        for _ in range(int(rng.integers(1, 4))):
            w, h = rng.uniform(0.5, 2.0), rng.uniform(0.4, 2.0)
            x, y = rng.uniform(bx0, max(bx0, bx1 - w)), rng.uniform(by0, max(by0, by1 - h))
            r = box(x, y, x + w, y + h)
            if not inner.contains(r):
                continue
            t = rng.random()
            if t < 0.5:
                cv.outline(r, THIN_LINE_MM)
                if rng.random() < 0.3:  # bed: pillow line
                    cv.line((x, y + h * 0.8), (x + w, y + h * 0.8), THIN_LINE_MM)
            elif t < 0.7:
                c = Point(x + w / 2, y + h / 2).buffer(min(w, h) / 2, 16)
                cv.outline(c, THIN_LINE_MM)
            elif t < 0.85 and w > 0.8 and h > 1.5:  # stairs
                cv.outline(r, THIN_LINE_MM)
                s = y + 0.27
                while s < y + h:
                    cv.line((x, s), (x + w, s), THIN_LINE_MM)
                    s += 0.27
                cv.line((x + w / 2, y + 0.1), (x + w / 2, y + h - 0.2), THIN_LINE_MM)
            else:  # wardrobe: box with diagonal
                cv.outline(r, THIN_LINE_MM)
                cv.line((x, y), (x + w, y + h), THIN_LINE_MM)
        if rng.random() < 0.5 * k:  # floor note
            c = poly.representative_point()
            txt = str(rng.choice(NOTES))
            p = cv.px((c.x + rng.uniform(-0.5, 0.5), c.y - rng.uniform(0.4, 0.9)))
            tw, th = cv.text_size(txt, 2.0)
            if not cv.collides((p[0] - tw / 2, p[1] - th / 2, p[0] + tw / 2, p[1] + th / 2)):
                cv.text(txt, p, 2.0)
    x0, y0, x1, y1 = walls_u.bounds
    if rng.random() < 0.3 * k:  # axis grid (dash-dot), crossing the drawing
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                xx = rng.uniform(x0, x1)
                cv.dashed([(xx, y0 - 1.5), (xx, y1 + 1.5)], THIN_LINE_MM, 4.0, 1.5, dot=True)
            else:
                yy = rng.uniform(y0, y1)
                cv.dashed([(x0 - 1.5, yy), (x1 + 1.5, yy)], THIN_LINE_MM, 4.0, 1.5, dot=True)
    H, W = cv.img.shape
    if rng.random() < 0.5 * k:  # title block bottom right
        bw, bh = cv.mm(rng.uniform(50, 90)), cv.mm(rng.uniform(18, 30))
        X1, Y1 = W - cv.mm(5), H - cv.mm(5)
        X0, Y0 = X1 - bw, Y1 - bh
        if X0 > 0 and Y0 > 0:
            cv2.rectangle(cv.img, (int(X0), int(Y0)), (int(X1), int(Y1)), 0, cv.lw(0.35))
            cv2.line(cv.img, (int(X0), int(Y0 + bh / 2)), (int(X1), int(Y0 + bh / 2)), 0,
                     cv.lw(0.18))
            cv.text(str(rng.choice(TITLES)), (X0 + bw / 2, Y0 + bh / 4), 2.5)
            cv.text("M 1:100", (X0 + bw / 2, Y0 + 3 * bh / 4), 2.5)
    if rng.random() < 0.3 * k:  # north arrow, top left margin
        c = np.array([cv.mm(15), cv.mm(15)])
        r = cv.mm(6)
        cv2.circle(cv.img, tuple(c.astype(int)), int(r), 0, cv.lw(0.18), cv2.LINE_AA)
        pts = np.array([c + [0, -r * 1.3], c + [-r * 0.4, r * 0.6], c + [r * 0.4, r * 0.6]])
        cv2.fillPoly(cv.img, [pts.astype(np.int32)], 0, cv2.LINE_AA)
    if rng.random() < 0.25 * k:  # round stamp
        c = (int(rng.uniform(0.1, 0.9) * W), int(rng.uniform(0.05, 0.2) * H))
        r = int(cv.mm(rng.uniform(10, 16)))
        cv2.circle(cv.img, c, r, 0, cv.lw(0.25), cv2.LINE_AA)
        cv2.circle(cv.img, c, int(r * 0.75), 0, cv.lw(0.18), cv2.LINE_AA)
        cv.text("GEPRUEFT", c, 2.0)


# -------------------------------------------------------------------- page


def render_styled(plan: VectorPlan, ts: TrainStyle, rng: np.random.Generator):
    """Draw an axis-aligned plan (metres) in style ``ts`` at 300 dpi -> (image, label) or None."""
    plain_given = isinstance(plan, StyledPlan) and plan.plain_walls
    walls_all = list(plan.walls) + (list(plan.plain_walls) if plain_given else [])
    walls_u = unary_union([w.buffer(0) for w in walls_all])
    if walls_u.is_empty:
        return None
    x0, y0, x1, y1 = walls_u.bounds
    if max(x1 - x0, y1 - y0) > MAX_EXTENT_M or min(x1 - x0, y1 - y0) < 2.0:
        return None
    dx, dy = MARGIN_M - x0, MARGIN_M - y0

    def mv(g):
        return affinity.translate(g, dx, dy)

    plain_extra = [mv(g) for g in plan.plain_walls] if plain_given else []
    plan = transform(plan, mv)
    walls_u = mv(walls_u)
    W = int(math.ceil((x1 - x0 + 2 * MARGIN_M) / M_PER_PX))
    Hh = int(math.ceil((y1 - y0 + 2 * MARGIN_M) / M_PER_PX))
    cv = _StyledCanvas(W, Hh, ts)

    cuts = unary_union([g.buffer(0.005) for g in plan.doors + plan.windows])
    if plain_given:
        hatched = [g.buffer(0) for g in plan.walls]
        plain = [g.buffer(0) for g in plain_extra]
    else:
        hatch_min = float(rng.uniform(0.16, 0.22))
        hatched, plain = [], []
        for g in _polys(walls_u):
            (hatched if _thickness(g) >= hatch_min else plain).append(g)
    if ts.hatch_all_walls:
        hatched, plain = hatched + plain, []
    hatched_u = unary_union(hatched).difference(cuts) if hatched else Polygon()
    plain_u = unary_union(plain).difference(hatched_u).difference(cuts) if plain else Polygon()

    cv.scale(ts.hatch_line_scale).hatch_v1(cv.poly_mask(hatched_u), rng)
    cv.scale(ts.wall_line_scale)
    if ts.wall_outline:
        cv.outline(hatched_u, WALL_LINE_MM)
        cv.outline(plain_u, WALL_LINE_MM)
    else:  # hatching without outline: still close light partitions with a thin outline
        cv.outline(plain_u, THIN_LINE_MM)
    for c in plan.columns:
        cv.outline(c, WALL_LINE_MM)
        cv.scale(ts.hatch_line_scale).hatch_v1(cv.poly_mask(c), rng)
        cv.scale(ts.wall_line_scale)
    rooms_u = unary_union([r for r, _ in plan.rooms]) if plan.rooms else None
    cv.scale(ts.thin_line_scale)
    for d in plan.doors:
        _draw_door_v1(cv, d, rooms_u, rng, ts.door)
    for w in plan.windows:
        _draw_window_v1(cv, w, rooms_u, rng, ts.window)
    for f in plan.fixtures:
        cv.outline(f, THIN_LINE_MM)
    for r in plan.railings:
        for g in _polys(r) if hasattr(r, "exterior") or hasattr(r, "geoms") else []:
            cv.outline(g, THIN_LINE_MM)
    for poly, name in plan.rooms:  # room stamps
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
        if rng.random() < 0.85:
            cv.text(area, (cx, cy + ah / 2 + 4), AREA_TEXT_MM)
    bx0, by0, bx1, by1 = walls_u.bounds
    style = str(rng.choice(["cm", "sup", "m"]))
    ext = walls_u.exterior if isinstance(walls_u, Polygon) else None
    xs = [bx0, bx1] + ([p[0] for p in ext.coords] if ext is not None else [])
    ys = [by0, by1] + ([p[1] for p in ext.coords] if ext is not None else [])
    r = rng.random()
    sides = () if r < 0.1 else ("s", "w") if r < 0.55 else ("s", "w", "n", "e")
    for s in sides:
        if s in "sn":
            yy = by0 - 1.0 if s == "s" else by1 + 1.0
            _draw_chain(cv, [bx0, bx1], yy - (0.7 if s == "s" else -0.7), True, style)
            _draw_chain(cv, xs, yy, True, style)
        else:
            xx = bx0 - 1.0 if s == "w" else bx1 + 1.0
            _draw_chain(cv, [by0, by1], xx - (0.7 if s == "w" else -0.7), False, style)
            _draw_chain(cv, ys, xx, False, style)
    _draw_clutter(cv, plan, walls_u, rng)

    lab = np.zeros(cv.img.shape, np.uint8)
    for poly, _ in plan.rooms:
        lab[cv.poly_mask(poly)] = ROOM
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


def synth_vector_plan(seed: int) -> StyledPlan:
    """Geometry of ``kelpod.evaluation.synth.generate(seed)`` (train style; the drawing is redone
    here) as a :class:`StyledPlan`: masonry and light walls with opening gaps, door / window gap
    boxes and named rooms."""
    from kelpod.contracts import OpeningKind
    from kelpod.evaluation.synth import MARGIN_M as SYNTH_MARGIN
    from kelpod.evaluation.synth.layout import make_layout
    from kelpod.evaluation.synth.render import _wall_regions
    from kelpod_train.labels import opening_boxes_m

    m_per_px = 0.0254 / 300.0 * 100.0
    rng = np.random.default_rng([int(seed), 0x6B656C70])  # the generator's layout stream
    layout, _ = make_layout(rng, margin_m=SYNTH_MARGIN, m_per_px=m_per_px)
    hatched, plain = _wall_regions(layout)
    g = layout.graph
    doors, windows = [], []
    for kind, quad in opening_boxes_m(g):
        (doors if kind == OpeningKind.DOOR else windows).append(Polygon(quad))
    rooms = [(Polygon(r.polygon), r.name) for r in g.rooms]
    return StyledPlan(walls=_polys(hatched), doors=doors, windows=windows, rooms=rooms,
                      plain_walls=_polys(plain))


def styled_sample(plan: VectorPlan, *, seed: int, ts: TrainStyle | None = None,
                  align: bool = True, sampler=None):
    """Draw ``plan`` in a randomised train style, degrade it (``degrade_v1``) and clean it.
    Returns a :class:`kelpod_train.labels.Sample` or None."""
    from kelpod_train.degrade_v1 import degrade_page
    from kelpod_train.labels import to_page
    from kelpod_train.style_v1 import sample_train_style

    rng = np.random.default_rng([int(seed), 0x76317374])
    if ts is None:
        ts = (sampler or sample_train_style)(rng)
    if align and not isinstance(plan, StyledPlan):
        plan = axis_align(plan)
    out = render_styled(plan, ts, rng)
    if out is None:
        return None
    img, lab = out
    deg, lab, affine = degrade_page(img, lab, ts, seed)
    s = to_page(deg, lab, affine, SRC_DPI)
    s.meta.update(seed=seed, style=ts.to_dict(), degrade=ts.scan_degrade)
    return s
