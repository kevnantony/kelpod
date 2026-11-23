"""Draw a :class:`~kelpod.evaluation.synth.layout.Layout` as a German 1:100 plan (DIN 1356-1 style).

All sizes on paper are given in millimetres and converted with the dpi, so the drawing looks the
same at any resolution. Plan space (m, y up) -> image px via :func:`kelpod.contracts.m_to_px`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from kelpod.contracts import OpeningKind, Point, WallKind, m_to_px
from kelpod.evaluation.synth.layout import Layout, Line, OpeningSpec
from kelpod.evaluation.synth.text import (
    LengthStyle,
    SillStyle,
    format_area,
    format_length,
    format_sill,
    render_text,
)

# line widths and text heights on paper (mm)
WALL_LINE_MM = 0.35
HATCH_LINE_MM = 0.13
HATCH_SPACING_MM = 1.6
THIN_LINE_MM = 0.18
LEAF_LINE_MM = 0.25
TICK_LINE_MM = 0.35
TICK_LEN_MM = 2.6
DIM_TEXT_MM = 2.5
NAME_TEXT_MM = 3.0
AREA_TEXT_MM = 2.5
BRH_TEXT_MM = 2.0
TEXT_GAP_MM = 0.7
CHAIN_OFFSETS_M = (0.9, 1.6, 2.3)
"""Distance of the three dimension chains from the outer wall face, in plan metres at 1:100."""


@dataclass
class DrawnText:
    text: str
    kind: str
    quad: list[Point]
    angle_deg: float
    value: float | None = None
    line: tuple[Point, Point] | None = None
    room_id: int | None = None
    opening_id: int | None = None


@dataclass
class Style:
    length: LengthStyle = "cm"
    sill: SillStyle = "m"


class _Canvas:
    """Grayscale uint8 drawing surface with plan-space helpers (AA lines, sub-pixel)."""

    SHIFT = 4

    def __init__(self, width: int, height: int, dpi: float, m_per_px: float) -> None:
        self.img = np.full((height, width), 255, np.uint8)
        self.h = height
        self.dpi = dpi
        self.m_per_px = m_per_px
        self.boxes: list[tuple[float, float, float, float]] = []  # placed text boxes (px)

    def mm(self, mm: float) -> float:
        return mm * self.dpi / 25.4

    def lw(self, mm: float) -> int:
        return max(1, int(round(self.mm(mm))))

    def px(self, p: Point) -> Point:
        return m_to_px(p, self.m_per_px, self.h)

    def _fix(self, p: Point) -> tuple[int, int]:
        f = 1 << self.SHIFT
        return (int(round(p[0] * f)), int(round(p[1] * f)))

    def line(self, a: Point, b: Point, width_mm: float) -> None:
        """Line between two plan-space points."""
        cv2.line(
            self.img,
            self._fix(self.px(a)),
            self._fix(self.px(b)),
            0,
            self.lw(width_mm),
            cv2.LINE_AA,
            self.SHIFT,
        )

    def polyline(self, pts: list[Point], width_mm: float, closed: bool = False) -> None:
        arr = np.array([self._fix(self.px(p)) for p in pts], np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.img, [arr], closed, 0, self.lw(width_mm), cv2.LINE_AA, self.SHIFT)

    def poly_mask(self, geom) -> np.ndarray:
        """Boolean mask of a (multi)polygon given in plan space, holes respected."""
        mask = np.zeros(self.img.shape, np.uint8)
        for poly in _polys(geom):
            ext = np.array([self._fix(self.px(p)) for p in poly.exterior.coords], np.int32)
            cv2.fillPoly(mask, [ext], 1, cv2.LINE_8, self.SHIFT)
            for ring in poly.interiors:
                hole = np.array([self._fix(self.px(p)) for p in ring.coords], np.int32)
                cv2.fillPoly(mask, [hole], 0, cv2.LINE_8, self.SHIFT)
        return mask.astype(bool)

    def outline(self, geom, width_mm: float) -> None:
        for poly in _polys(geom):
            for ring in (poly.exterior, *poly.interiors):
                self.polyline(list(ring.coords)[:-1], width_mm, closed=True)

    def hatch(self, mask: np.ndarray) -> None:
        """45° hatching clipped to ``mask``."""
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        layer = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        spacing = self.mm(HATCH_SPACING_MM)
        h, w = layer.shape
        # lines x + y = C (rising to the right on screen), C a multiple of the spacing in page
        # coordinates so the pattern is continuous across walls; c is C in crop coordinates
        c = float(-(x0 + y0) % spacing)
        f = 1 << self.SHIFT
        width = self.lw(HATCH_LINE_MM)
        while c < w + h:
            p0 = (int(round(c * f)), 0)
            p1 = (int(round((c - h) * f)), h * f)
            cv2.line(layer, p0, p1, 0, width, cv2.LINE_AA, self.SHIFT)
            c += spacing
        crop = self.img[y0:y1, x0:x1]
        sub = mask[y0:y1, x0:x1]
        crop[sub] = np.minimum(crop[sub], layer[sub])

    def text(
        self,
        text: str,
        center: Point,
        digit_mm: float,
        angle_deg: float = 0.0,
        *,
        draw: bool = True,
    ) -> tuple[list[Point], str, tuple[float, float, float, float]]:
        """Draw text centred at a px point; returns quad (clockwise from top-left as read)."""
        alpha, drawn = render_text(text, self.mm(digit_mm))
        if angle_deg == 90.0:
            alpha = np.rot90(alpha, 1)  # counter-clockwise: reads bottom-to-top
        h, w = alpha.shape
        x0 = int(round(center[0] - w / 2))
        y0 = int(round(center[1] - h / 2))
        bbox = (float(x0), float(y0), float(x0 + w), float(y0 + h))
        if draw:
            H, W = self.img.shape
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
            if cx1 > cx0 and cy1 > cy0:
                a = alpha[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
                region = self.img[cy0:cy1, cx0:cx1].astype(np.float32)
                self.img[cy0:cy1, cx0:cx1] = np.round(region * (1.0 - a)).astype(np.uint8)
            self.boxes.append(bbox)
        bx0, by0, bx1, by1 = bbox
        if angle_deg == 90.0:
            quad = [(bx0, by1), (bx0, by0), (bx1, by0), (bx1, by1)]
        else:
            quad = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        return quad, drawn, bbox

    def text_size(self, text: str, digit_mm: float, angle_deg: float = 0.0) -> tuple[int, int]:
        alpha, _ = render_text(text, self.mm(digit_mm))
        h, w = alpha.shape
        return (h, w) if angle_deg == 90.0 else (w, h)

    def collides(self, bbox: tuple[float, float, float, float], pad: float = 2.0) -> bool:
        x0, y0, x1, y1 = bbox
        return any(
            x0 < b[2] + pad and b[0] - pad < x1 and y0 < b[3] + pad and b[1] - pad < y1
            for b in self.boxes
        )


def _polys(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _wall_regions(layout: Layout):
    """Wall bodies (plan space) split into hatched (masonry) and plain (light partition) regions,
    with the opening gaps cut out."""
    g = layout.graph
    pos = {n.id: (n.x, n.y) for n in g.nodes}
    at: dict[int, list[int]] = {}
    for w in g.walls:
        at.setdefault(w.start, []).append(w.id)
        at.setdefault(w.end, []).append(w.id)

    def direction(wid: int) -> tuple[float, float]:
        w = g.walls[wid]
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        length = math.hypot(x1 - x0, y1 - y0)
        return ((x1 - x0) / length, (y1 - y0) / length)

    def ext(wid: int, node: int) -> float:
        d = direction(wid)
        best = 0.0
        for other in at[node]:
            if other == wid:
                continue
            e = direction(other)
            if abs(d[0] * e[0] + d[1] * e[1]) < 0.5:  # perpendicular
                best = max(best, g.walls[other].thickness_m / 2)
        return best

    hatched, plain = [], []
    for w in g.walls:
        (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
        dx, dy = direction(w.id)
        e0, e1 = ext(w.id, w.start), ext(w.id, w.end)
        a = (x0 - dx * e0, y0 - dy * e0)
        b = (x1 + dx * e1, y1 + dy * e1)
        t = w.thickness_m / 2
        nx, ny = -dy, dx
        rect = Polygon(
            [
                (a[0] + nx * t, a[1] + ny * t),
                (b[0] + nx * t, b[1] + ny * t),
                (b[0] - nx * t, b[1] - ny * t),
                (a[0] - nx * t, a[1] - ny * t),
            ]
        )
        (plain if w.kind == WallKind.DRYWALL else hatched).append(rect)

    cuts = (
        unary_union(
            [_opening_box(layout, op, layout.lines[op.line_id], 0.01) for op in layout.openings]
        )
        if layout.openings
        else Polygon()
    )
    hatched_u = unary_union(hatched) if hatched else Polygon()
    plain_u = unary_union(plain).difference(hatched_u) if plain else Polygon()
    hatched_u = shapely.set_precision(hatched_u.difference(cuts), 1e-5)
    plain_u = shapely.set_precision(plain_u.difference(cuts), 1e-5)
    return hatched_u, plain_u


def _to_plan(layout: Layout, line: Line, s: float, off: float) -> Point:
    """Point at position ``s`` along ``line`` and ``off`` across it (internal -> plan space)."""
    if line.axis == "h":
        return layout.to_plan(s, line.coord + off)
    return layout.to_plan(line.coord + off, s)


def _opening_box(layout: Layout, op: OpeningSpec, line: Line, extra: float) -> Polygon:
    t = line.thickness / 2 + extra
    a = _to_plan(layout, line, op.s0, -t)
    b = _to_plan(layout, line, op.s1, t)
    return box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def _room_side(layout: Layout, line: Line, room: int) -> float:
    """+1 if the room lies on the positive side of the line (larger x or y), else -1."""
    c = layout.cells[room]
    centre = (c.y0 + c.y1) / 2 if line.axis == "h" else (c.x0 + c.x1) / 2
    return 1.0 if centre > line.coord else -1.0


def _draw_openings(cv: _Canvas, layout: Layout) -> None:
    for op in layout.openings:
        line = layout.lines[op.line_id]
        t = line.thickness
        if op.kind == OpeningKind.WINDOW:
            offs = (-0.38 * t, 0.0, 0.38 * t) if op.window_lines == 3 else (-0.12 * t, 0.12 * t)
            for o in offs:
                cv.line(
                    _to_plan(layout, line, op.s0, o), _to_plan(layout, line, op.s1, o), THIN_LINE_MM
                )
            continue
        side = _room_side(layout, line, op.room)
        face = side * t / 2
        width = op.s1 - op.s0
        hinge_s, other_s = (op.s0, op.s1) if op.hinge_at_s0 else (op.s1, op.s0)
        hinge = _to_plan(layout, line, hinge_s, face)
        leaf_end = _to_plan(layout, line, hinge_s, face + side * width)
        cv.line(hinge, leaf_end, LEAF_LINE_MM)
        # quarter arc from the leaf end to the far jamb, centred at the hinge
        a0 = math.atan2(leaf_end[1] - hinge[1], leaf_end[0] - hinge[0])
        far = _to_plan(layout, line, other_s, face)
        a1 = math.atan2(far[1] - hinge[1], far[0] - hinge[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        n = 24
        arc = [
            (
                hinge[0] + width * math.cos(a0 + da * k / n),
                hinge[1] + width * math.sin(a0 + da * k / n),
            )
            for k in range(n + 1)
        ]
        cv.polyline(arc, THIN_LINE_MM)


def _chains(layout: Layout) -> dict[str, list[list[float]]]:
    """Dimension chain stops per side (``s``, ``n``, ``w``, ``e``), internal coordinates along
    the side. Up to three chains, inner to outer: openings, rooms + walls, overall."""
    te = layout.lines[0].thickness
    inner_w = layout.lines[3].coord - te / 2
    inner_h = layout.lines[1].coord - te / 2
    side_line = {"s": 0, "n": 1, "w": 2, "e": 3}
    out: dict[str, list[list[float]]] = {}
    for side, lid in side_line.items():
        length = inner_w if side in "sn" else inner_h
        lo, hi = -te, length + te
        ops = sorted((op.s0, op.s1) for op in layout.openings if op.line_id == lid)
        chain1 = sorted({lo, hi, *[s for o in ops for s in o]}) if ops else []
        faces = {lo, 0.0, length, hi}
        for ln in layout.lines[4:]:
            perp = ("v" if side in "sn" else "h") == ln.axis
            touches = (
                abs(ln.lo - layout.lines[lid].coord) < 1e-6
                or abs(ln.hi - layout.lines[lid].coord) < 1e-6
            )
            if perp and touches:
                faces |= {ln.coord - ln.thickness / 2, ln.coord + ln.thickness / 2}
        chain2 = sorted(faces)
        out[side] = [chain1, chain2, [lo, hi]]
    return out


def _draw_dimensions(cv: _Canvas, layout: Layout, style: Style, texts: list[DrawnText]) -> None:
    ox, oy = layout.origin
    x_out0, y_out0, x_out1, y_out1 = layout.outer
    tick = TICK_LEN_MM / 1000 * 100 / 2  # half tick length in plan metres at 1:100
    gap = cv.mm(TEXT_GAP_MM)
    for side, chains in _chains(layout).items():
        horizontal = side in "sn"
        outward = -1.0 if side in "sw" else 1.0
        face = {"s": y_out0, "n": y_out1, "w": x_out0, "e": x_out1}[side]
        base = ox if horizontal else oy
        for level, stops in enumerate(chains):
            if len(stops) < 2:
                continue
            c = face + outward * CHAIN_OFFSETS_M[level]

            def P(s: float, across: float, horizontal=horizontal, base=base) -> Point:
                return (base + s, across) if horizontal else (across, base + s)

            # dimension line, slightly past the end ticks
            cv.line(P(stops[0] - 0.12, c), P(stops[-1] + 0.12, c), THIN_LINE_MM)
            for s in stops:
                cv.line(P(s, face + outward * 0.25), P(s, c + outward * 0.12), THIN_LINE_MM)
                cv.line(P(s - tick, c - tick), P(s + tick, c + tick), TICK_LINE_MM)
            for a, b in zip(stops[:-1], stops[1:], strict=True):
                value = round(b - a, 4)
                if value < 0.005:
                    continue
                text = format_length(value, style.length)
                pa, pb = cv.px(P(a, c)), cv.px(P(b, c))
                mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                angle = 0.0 if horizontal else 90.0
                tw, th = cv.text_size(text, DIM_TEXT_MM, angle)
                seg_px = abs(pb[0] - pa[0]) + abs(pb[1] - pa[1])
                along = tw if horizontal else th
                # "above" the line as read: up for horizontal text, left for rotated text
                # candidates: centred above the line, then beside the span (either side, one
                # and two text widths away), then one text height further from the line
                side = seg_px / 2 + along / 2 + gap
                along_shifts = [0.0, side, -side, side + along + gap, -(side + along + gap)]
                up = (th if horizontal else tw) + gap
                if horizontal:
                    centre = (mid[0], mid[1] - gap - th / 2)
                    shifts = [(d, 0.0) for d in along_shifts] + [(0.0, -up)]
                else:
                    centre = (mid[0] - gap - tw / 2, mid[1])
                    shifts = [(0.0, -d) for d in along_shifts] + [(-up, 0.0)]
                chosen = centre
                for dx, dy in shifts:
                    cand = (centre[0] + dx, centre[1] + dy)
                    bbox = (cand[0] - tw / 2, cand[1] - th / 2, cand[0] + tw / 2, cand[1] + th / 2)
                    if not cv.collides(bbox):
                        chosen = cand
                        break
                quad, drawn, _ = cv.text(text, chosen, DIM_TEXT_MM, angle)
                texts.append(DrawnText(drawn, "dimension", quad, angle, value, (pa, pb)))


def _stamp_fits(
    cv: _Canvas, name: str, area: str, room_w: float, room_h: float, k: float, rotated: bool
) -> bool:
    nw, nh = cv.text_size(name, NAME_TEXT_MM * k)
    aw, ah = cv.text_size(area, AREA_TEXT_MM * k)
    text_w, text_h = max(nw, aw), nh + ah + cv.mm(1.2 * k)
    if rotated:
        text_w, text_h = text_h, text_w
    return text_w <= room_w and text_h <= room_h


def _draw_stamps(cv: _Canvas, layout: Layout, texts: list[DrawnText]) -> None:
    for i, cell in enumerate(layout.cells):
        cx, cy = cv.px(layout.to_plan((cell.x0 + cell.x1) / 2, (cell.y0 + cell.y1) / 2))
        name = layout.names[i]
        area = format_area(layout.stamp_areas[i])
        room_w = cell.w / cv.m_per_px - cv.mm(1.5)
        room_h = cell.h / cv.m_per_px - cv.mm(1.5)

        # largest stamp that fits: horizontal first, then rotated (narrow rooms); if nothing
        # fits, draw it small along the longer room side
        options = [(k, rot) for rot in (False, True) for k in (1.0, 0.85, 0.7)]
        k, rotated = next(
            (o for o in options if _stamp_fits(cv, name, area, room_w, room_h, *o)),
            (0.7, cell.h > cell.w),
        )
        angle, scale = (90.0 if rotated else 0.0), k
        n_mm, a_mm = NAME_TEXT_MM * scale, AREA_TEXT_MM * scale
        nw, nh = cv.text_size(name, n_mm)
        aw, ah = cv.text_size(area, a_mm)
        total = nh + cv.mm(1.2 * scale) + ah
        if angle == 0.0:
            c_name = (cx, cy - total / 2 + nh / 2)
            c_area = (cx, cy + total / 2 - ah / 2)
        else:  # rotated: the name line sits left of the area line
            c_name = (cx - total / 2 + nh / 2, cy)
            c_area = (cx + total / 2 - ah / 2, cy)
        q, drawn, _ = cv.text(name, c_name, n_mm, angle)
        texts.append(DrawnText(drawn, "room_name", q, angle, None, room_id=i))
        q, drawn, _ = cv.text(area, c_area, a_mm, angle)
        texts.append(DrawnText(drawn, "room_area", q, angle, layout.stamp_areas[i], room_id=i))


def _draw_sills(cv: _Canvas, layout: Layout, style: Style, texts: list[DrawnText]) -> None:
    for gid, op in enumerate(layout.openings):  # same order as the graph openings
        if op.kind != OpeningKind.WINDOW:
            continue
        line = layout.lines[op.line_id]
        side = _room_side(layout, line, op.room)
        text = format_sill(op.sill, style.sill)
        angle = 0.0 if line.axis == "h" else 90.0
        tw, th = cv.text_size(text, BRH_TEXT_MM, angle)
        across_px = th if line.axis == "h" else tw
        off = side * (line.thickness / 2 + 0.12 + across_px / 2 * cv.m_per_px)
        c = cv.px(_to_plan(layout, line, (op.s0 + op.s1) / 2, off))
        bbox = (c[0] - tw / 2, c[1] - th / 2, c[0] + tw / 2, c[1] + th / 2)
        if not cv.collides(bbox, pad=4.0):
            q, drawn, _ = cv.text(text, c, BRH_TEXT_MM, angle)
            texts.append(DrawnText(drawn, "sill", q, angle, op.sill, opening_id=gid))


def render(
    layout: Layout, size: tuple[int, int], dpi: float, style: Style, m_per_px: float
) -> tuple[np.ndarray, np.ndarray, list[DrawnText]]:
    """Draw the plan. Returns the grayscale image, the wall mask and the drawn texts."""
    cv = _Canvas(size[0], size[1], dpi, m_per_px)
    hatched, plain = _wall_regions(layout)
    hmask = cv.poly_mask(hatched)
    pmask = cv.poly_mask(plain)
    cv.hatch(hmask)
    cv.outline(hatched, WALL_LINE_MM)
    cv.outline(plain, WALL_LINE_MM)
    _draw_openings(cv, layout)
    texts: list[DrawnText] = []
    _draw_stamps(cv, layout, texts)  # stamps first: sill texts avoid them
    _draw_sills(cv, layout, style, texts)
    _draw_dimensions(cv, layout, style, texts)
    x0, y0 = layout.outer[0], layout.outer[1]
    title = "Grundriss Erdgeschoss  M 1:100"
    tw, th = cv.text_size(title, NAME_TEXT_MM)
    tl = cv.px((x0, y0 - CHAIN_OFFSETS_M[-1] - 0.9))
    q, drawn, _ = cv.text(title, (tl[0] + tw / 2, tl[1] + th / 2), NAME_TEXT_MM)
    texts.append(DrawnText(drawn, "title", q, 0.0))
    return cv.img, hmask | pmask, texts
