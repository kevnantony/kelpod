"""Walls drawn as two thin parallel lines (empty or lightly filled interior), at 0/45/90/135 deg.

Modern CAD plans of single-family houses draw a wall as its two faces only. The classical
segmenter needs solid or hatched bodies, so it misses them and grabs furniture, stairs and text.
Here every orientation (0, 90 and, in a rotated copy of the ink, 45 / 135 degrees) is searched
for long thin line components; two parallel lines at wall-thickness spacing that overlap along
their length and enclose little ink form a wall. The dominant spacing is read from the plan
itself (length-weighted histogram), isolated pieces (furniture, dimension chains, terrace grids)
are dropped by network connectivity, and the barrier of the paired walls plus door gaps gives
the rooms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from kelpod.contracts import Segmentation, WallKind, WallSegment
from kelpod.segment import rooms as rm


@dataclass
class PairParams:
    min_line_px: int = 24
    """Shortest straight line component that can be one face of a wall (px at ~200 dpi)."""
    max_line_th_px: int = 6
    min_spacing_px: float = 4.0
    max_spacing_px: float = 48.0
    spacing_lo: float = 0.4
    spacing_hi: float = 2.9
    """Accepted spacings relative to the dominant (length-weighted mode) wall thickness."""
    max_fill: float = 0.35
    """Largest ink share between the two faces (text, furniture hatching, stair treads)."""
    min_wall_px: float = 24.0
    min_network_share: float = 0.12
    """Connected wall groups shorter than this share of the longest group are dropped."""
    door_max_th: float = 16.0
    """Largest gap closed as a door, in wall thicknesses."""
    door_min_th: float = 1.5
    diag: bool = True
    solid: bool = True
    """Also accept heavy single strokes (fused faces, poche) as walls."""
    min_diag_px: float = 60.0


@dataclass
class _Wall:
    p0: tuple[float, float]
    p1: tuple[float, float]
    th: float
    angle: int  # 0, 90, 45, 135


def _lines(ink_r: np.ndarray, prm: PairParams) -> list[tuple[float, float, float]]:
    """Long thin horizontal line components of ``ink_r``: (x0, x1, y)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (prm.min_line_px, 1))
    op = cv2.morphologyEx(ink_r, cv2.MORPH_OPEN, k)
    n, _, st, cen = cv2.connectedComponentsWithStats(op, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if h > prm.max_line_th_px or w < prm.min_line_px:
            continue
        out.append((float(x), float(x + w), float(cen[i][1])))
    return out


def _pair_candidates(ink_r, lines, prm):
    """All (overlap0, overlap1, y_a, y_b) of parallel line pairs inside the spacing window."""
    ls = sorted(lines, key=lambda t: t[2])
    cand = []
    for i, (a0, a1, ya) in enumerate(ls):
        for b0, b1, yb in ls[i + 1 :]:
            d = yb - ya
            if d > prm.max_spacing_px:
                break
            if d < prm.min_spacing_px:
                continue
            o0, o1 = max(a0, b0), min(a1, b1)
            if o1 - o0 < prm.min_line_px:
                continue
            cand.append((o0, o1, ya, yb))
    return cand


def _fill(ink_r: np.ndarray, o0: float, o1: float, ya: float, yb: float) -> float:
    ya_i, yb_i = int(round(ya)) + 3, int(round(yb)) - 2
    if yb_i <= ya_i:
        return 0.0
    sub = ink_r[ya_i:yb_i, int(o0) : int(o1)]
    return float(sub.mean()) if sub.size else 1.0


def _solid_bars(ink_r: np.ndarray, th_mode: float, prm: PairParams):
    """Heavy single strokes (faces fused at low resolution, solid poche): (x0, x1, y-h/2, y+h/2)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(prm.min_wall_px * 2), 1))
    er = cv2.erode(ink_r, np.ones((max(4, int(0.8 * th_mode)), 1), np.uint8))
    op = cv2.morphologyEx(er, cv2.MORPH_OPEN, k)
    n, _, st, _c = cv2.connectedComponentsWithStats(op, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        out.append((float(x), float(x + w), float(y), float(y + h + int(0.8 * th_mode) - 1)))
    return out


def _walls_in_frame(ink_r: np.ndarray, prm: PairParams, th_mode: float | None):
    lines = _lines(ink_r, prm)
    cand = _pair_candidates(ink_r, lines, prm)
    out = []
    for o0, o1, ya, yb in cand:
        if th_mode is not None and not (
            prm.spacing_lo * th_mode <= yb - ya <= prm.spacing_hi * th_mode
        ):
            continue
        if _fill(ink_r, o0, o1, ya, yb) > prm.max_fill:
            continue
        out.append((o0, o1, ya, yb))
    if th_mode is not None and prm.solid:
        out += _solid_bars(ink_r, th_mode, prm)
    return out


def _dominant_spacing(cands) -> float | None:
    if not cands:
        return None
    hist = np.zeros(64)
    for o0, o1, ya, yb in cands:
        hist[min(63, int(round(yb - ya)))] += o1 - o0
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    # prefer the thickest strong peak: nested faces of a window / pillar add smaller peaks
    best = float(np.argmax(hist))
    return best if hist[int(best)] > 0 else None


def _dedupe(rects):
    """Keep the widest pair where narrower pairs lie inside it (window centre lines)."""
    rects = sorted(rects, key=lambda r: -(r[3] - r[2]))
    kept: list[tuple[float, float, float, float]] = []
    for o0, o1, ya, yb in rects:
        covered = False
        for k0, k1, ka, kb in kept:
            ov = max(0.0, min(o1, k1) - max(o0, k0))
            if ov >= 0.6 * (o1 - o0) and ka - 2 <= ya and yb <= kb + 2:
                covered = True
                break
        if not covered:
            kept.append((o0, o1, ya, yb))
    return kept


def _merge_collinear(rects, gap=3.0, tol=2.5):
    rects = sorted(rects, key=lambda r: (round((r[2] + r[3]) / 2 / 3), r[0]))
    out: list[list[float]] = []
    for o0, o1, ya, yb in rects:
        for m in out:
            if abs(m[2] - ya) <= tol and abs(m[3] - yb) <= tol and o0 <= m[1] + gap and o1 >= m[0]:
                m[0], m[1] = min(m[0], o0), max(m[1], o1)
                break
        else:
            out.append([o0, o1, ya, yb])
    return [tuple(m) for m in out]


def _frames(ink: np.ndarray, diag: bool):
    """(name, transformed ink, inverse point map) for each wall direction."""
    h, w = ink.shape
    yield 0, ink, lambda x, y: (x, y)
    yield 90, np.ascontiguousarray(ink.T), lambda x, y: (y, x)
    if diag:
        diagn = int(math.ceil(math.hypot(h, w))) + 2
        for ang in (45, 135):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), ang if ang == 45 else -45, 1.0)
            m[0, 2] += (diagn - w) / 2
            m[1, 2] += (diagn - h) / 2
            big = cv2.warpAffine(ink * 255, m, (diagn, diagn), flags=cv2.INTER_LINEAR)
            r = (big > 110).astype(np.uint8)
            inv = cv2.invertAffineTransform(m)

            def f(x, y, inv=inv):
                return (
                    inv[0, 0] * x + inv[0, 1] * y + inv[0, 2],
                    inv[1, 0] * x + inv[1, 1] * y + inv[1, 2],
                )

            yield ang, r, f


def detect_pair_walls(ink: np.ndarray, prm: PairParams | None = None) -> tuple[list[_Wall], dict]:
    prm = prm or PairParams()
    ink_u = ink.astype(np.uint8)
    info: dict = {}
    # pass 1: dominant spacing from the axis-aligned frames
    cands0 = []
    for _ang, r, _f in _frames(ink_u, False):
        c = _walls_in_frame(r, prm, None)
        cands0 += c
    th_mode = _dominant_spacing(cands0)
    info["th_mode_px"] = th_mode
    if th_mode is None:
        return [], info
    walls: list[_Wall] = []
    for ang, r, f in _frames(ink_u, prm.diag):
        rects = _walls_in_frame(r, prm, th_mode)
        rects = _merge_collinear(_dedupe(rects))
        for o0, o1, ya, yb in rects:
            if ang in (45, 135) and o1 - o0 < prm.min_diag_px:
                continue
            if o1 - o0 < prm.min_wall_px:
                continue
            yc = (ya + yb) / 2
            walls.append(_Wall(f(o0, yc), f(o1, yc), yb - ya, ang))
    return walls, info


def _network_filter(walls: list[_Wall], prm: PairParams) -> list[_Wall]:
    """Keep wall groups connected (ends touching other walls within a thickness) to the main one."""
    n = len(walls)
    if n == 0:
        return walls
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def seg_dist(p, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-9
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)

    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = walls[i], walls[j]
            tol = 0.75 * max(wi.th, wj.th) + 2
            if (
                min(seg_dist(wi.p0, wj.p0, wj.p1), seg_dist(wi.p1, wj.p0, wj.p1)) <= tol
                or min(seg_dist(wj.p0, wi.p0, wi.p1), seg_dist(wj.p1, wi.p0, wi.p1)) <= tol
            ):
                parent[find(i)] = find(j)
    length = {}
    for i, w in enumerate(walls):
        length[find(i)] = length.get(find(i), 0.0) + math.dist(w.p0, w.p1)
    top = max(length.values())
    return [w for i, w in enumerate(walls) if length[find(i)] >= prm.min_network_share * top]


def segment_pairs(
    ink: np.ndarray, prm: PairParams | None = None
) -> tuple[Segmentation | None, list[str]]:
    """Segmentation from paired thin lines, or ``None`` if no such walls are found."""
    prm = prm or PairParams()
    walls, info = detect_pair_walls(ink, prm)
    notes = []
    if not walls:
        return None, ["pairs: no parallel line pairs found"]
    walls = _network_filter(walls, prm)
    h, w = ink.shape
    th = info["th_mode_px"]
    barrier = np.zeros((h, w), np.uint8)
    for wl_ in walls:
        _draw(barrier, wl_, 0.0)
    # door gaps: close collinear gaps between wall ends so rooms are bounded
    _close_gaps(barrier, walls, prm)
    wall_mask = barrier.astype(bool)
    _, lab, outside = rm.free_space(wall_mask)
    min_area = (4.0 * th) ** 2
    rooms = rm.extract_rooms(lab, outside, min_area_px=min_area)
    segs = []
    for i, wl_ in enumerate(walls):
        ext = _touches_outside(wl_, lab, outside)
        segs.append(
            WallSegment(
                id=i,
                p0=wl_.p0,
                p1=wl_.p1,
                thickness_px=float(wl_.th),
                kind=WallKind.EXTERIOR if ext else WallKind.INTERIOR,
                confidence=0.7,
            )
        )
    notes.append(
        f"pairs: {len(segs)} walls from parallel line pairs (dominant spacing {th:.1f} px), "
        f"{len(rooms)} rooms"
    )
    seg = Segmentation(
        wall_mask=wall_mask, walls=segs, openings=[], rooms=rooms, backend="classical", notes=notes
    )
    return seg, notes


def _draw(img: np.ndarray, wl_: _Wall, grow: float) -> None:
    (x0, y0), (x1, y1) = wl_.p0, wl_.p1
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / d, (x1 - x0) / d
    t = wl_.th / 2 + grow
    pts = np.array(
        [
            (x0 + nx * t, y0 + ny * t),
            (x1 + nx * t, y1 + ny * t),
            (x1 - nx * t, y1 - ny * t),
            (x0 - nx * t, y0 - ny * t),
        ]
    )
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], 1)


def _close_gaps(barrier: np.ndarray, walls: list[_Wall], prm: PairParams) -> None:
    """Fill the gap between two collinear wall ends (door / window opening) so rooms close."""
    for i, a in enumerate(walls):
        for b in walls[i + 1 :]:
            if a.angle != b.angle:
                continue
            th = max(a.th, b.th)
            for pa in (a.p0, a.p1):
                for pb in (b.p0, b.p1):
                    d = math.dist(pa, pb)
                    if not (prm.door_min_th * 0 < d <= prm.door_max_th * th):
                        continue
                    # collinear: both ends lie on a common line within a thickness
                    ax = a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]
                    L = math.hypot(*ax) or 1.0
                    off = abs((pb[0] - pa[0]) * ax[1] - (pb[1] - pa[1]) * ax[0]) / L
                    along = abs((pb[0] - pa[0]) * ax[0] + (pb[1] - pa[1]) * ax[1]) / L
                    if off <= 0.5 * th and along >= 0.7 * d:
                        _draw(barrier, _Wall(pa, pb, min(a.th, b.th), a.angle), 1.0)


def _touches_outside(w: _Wall, lab: np.ndarray, outside: np.ndarray) -> bool:
    h, wd = lab.shape
    (x0, y0), (x1, y1) = w.p0, w.p1
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / d, (x1 - x0) / d
    off = w.th / 2 + 3
    for side in (-1, 1):
        votes = []
        for t in np.linspace(0.2, 0.8, 5):
            x = x0 + t * (x1 - x0) + side * nx * off
            y = y0 + t * (y1 - y0) + side * ny * off
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < wd and 0 <= yi < h:
                votes.append(bool(outside[lab[yi, xi]]))
            else:
                votes.append(True)
        if votes and sum(votes) > len(votes) / 2:
            return True
    return False


class PairsSegmenter:
    """Backend ``"pairs"``: walls from parallel thin line pairs (modern CAD plans)."""

    name = "pairs"

    def segment(self, page) -> Segmentation:
        seg, notes = segment_pairs(np.asarray(page.ink, dtype=bool))
        if seg is None:
            h, w = page.ink.shape
            seg = Segmentation(wall_mask=np.zeros((h, w), bool), backend="classical", notes=notes)
        return seg


__all__ = ["PairParams", "PairsSegmenter", "detect_pair_walls", "segment_pairs"]
