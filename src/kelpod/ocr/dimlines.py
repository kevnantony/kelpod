"""Find the dimension line under a dimension number and the two ticks of its span.

Convention (``docs/ARCHITECTURE.md``): the value sits centred above a thin dimension line; each
measured point has a short 45° tick stroke across the line and usually an extension line. So, in the
text's own upright frame, we look for a long thin ink run just below the text, then for crossing
strokes along it, and take the nearest crossing stroke on each side of the text centre.

Vertical chains (text reading bottom-to-top) are handled by turning the ink mask 90° clockwise,
which makes them horizontal, and mapping the result back.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from kelpod.contracts import Point
from kelpod.ocr.engines import bbox, cw_to_original, original_to_cw, quad_size


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (inclusive) of True runs in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _close_gaps(row: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 0:
        return row
    return ndimage.binary_closing(row, structure=np.ones(2 * gap + 1, bool)) | row


def find_span_horizontal(
    ink: np.ndarray, box: tuple[float, float, float, float], dil: np.ndarray | None = None
) -> tuple[tuple[float, float], float, float] | None:
    """Horizontal case. ``box`` = (x0, y0, x1, y1) of upright text; ``dil`` is ``ink`` dilated by
    3x3 (computed if not given).

    Returns ``((x_left_tick, x_right_tick), y_line, score)`` or ``None``.
    """
    H, W = ink.shape
    x0, y0, x1, y1 = box
    th = max(4.0, y1 - y0)  # text height
    cx = (x0 + x1) / 2
    icx = int(round(min(max(cx, 0), W - 1)))
    gap = max(1, int(round(th * 0.08)))

    # 1. Candidate rows just below the text (preferred), else just above it.
    def best_row(r0: int, r1: int) -> tuple[int, int, int] | None:
        best = None
        for y in range(max(0, r0), min(H, r1)):
            row = _close_gaps(ink[y], gap)
            if not row[icx]:
                # The text centre may sit on a gap of the line (e.g. a tick); look nearby.
                near = np.flatnonzero(row[max(0, icx - gap - 1) : icx + gap + 2])
                if near.size == 0:
                    continue
            for a, b in _runs(row):
                if a <= icx + gap + 1 and b >= icx - gap - 1:
                    outside = max(0.0, x0 - a) + max(0.0, b - x1)
                    if outside < th:  # must extend clearly beyond the text: not a text stroke
                        break
                    length = b - a + 1
                    dist = abs(y - y1) if y >= y1 - 1 else abs(y0 - y)
                    if best is None or length > best[2] * 1.05 or (
                        length >= best[2] * 0.95 and dist < best[3]
                    ):
                        best = (y, a, length, dist)
                    break
        return None if best is None else (best[0], best[1], best[2])

    found = best_row(int(y1 - 0.3 * th), int(y1 + 1.5 * th))
    if found is None:
        found = best_row(int(y0 - 1.5 * th), int(y0 + 0.3 * th))
    if found is None:
        return None
    yl, a, length = found
    b = a + length - 1

    # Line thickness: neighbouring rows that also carry the run at the text centre.
    top = yl
    while top - 1 >= 0 and ink[top - 1, max(a, icx - 2) : min(b, icx + 3)].mean() > 0.6:
        top -= 1
        if yl - top > th * 0.3:
            break
    bot = yl
    while bot + 1 < H and ink[bot + 1, max(a, icx - 2) : min(b, icx + 3)].mean() > 0.6:
        bot += 1
        if bot - yl > th * 0.3:
            break
    yc = (top + bot) / 2
    half = (bot - top) / 2
    # Recompute the run over the full line thickness (an edge row may be broken by aliasing).
    band = _close_gaps(ink[max(0, top - 1) : bot + 2].any(axis=0), gap)
    for ra, rb in _runs(band):
        if ra <= icx + gap + 1 and rb >= icx - gap - 1:
            a, b = min(a, ra), max(b, rb)
            break

    # 2. Crossing strokes along the line: a 45° tick (either diagonal) or a perpendicular
    #    extension line, present on BOTH sides of the line (text strokes are only above it).
    if dil is None:
        dil = ndimage.binary_dilation(ink, structure=np.ones((3, 3), bool))
    L = max(3, int(round(th * 0.3)))
    xs = np.arange(a, b + 1)
    ycl = int(round(yc))

    def side_score(dx: int, dy: int) -> np.ndarray:
        hits = np.zeros(xs.size, float)
        for k in range(1, L + 1):
            d = int(np.ceil(half)) + k
            yy = ycl + dy * d
            if not (0 <= yy < H):
                return np.zeros(xs.size)
            xx = np.clip(xs + dx * d, 0, W - 1)
            hits += dil[yy, xx]
        return hits / L

    diag1 = np.minimum(side_score(+1, -1), side_score(-1, +1))  # "/"
    diag2 = np.minimum(side_score(-1, -1), side_score(+1, +1))  # "\"
    perp = np.minimum(side_score(0, -1), side_score(0, +1))
    score = np.maximum(np.maximum(diag1, diag2), perp)
    is_tick = score >= 0.7
    ticks = [(xs[s] + xs[e]) / 2 for s, e in _runs(is_tick)]
    left = [t for t in ticks if t < cx - 1]
    right = [t for t in ticks if t > cx + 1]
    if not left or not right:
        return None
    xl, xr = max(left), min(right)
    dl, dr = cx - xl, xr - cx
    # The text is centred on its span, so a missed tick shows up as asymmetry. Allow a text
    # height of slack for short spans (OCR boxes are a few pixels loose).
    asym = abs(dl - dr) / (dl + dr)
    if asym > 0.15 and abs(dl - dr) > th:
        return None
    return (float(xl), float(xr)), float(yc), float(1.0 - asym)


class DimensionLineFinder:
    """Finds dimension spans on one page; caches the dilated and rotated ink masks."""

    def __init__(self, ink: np.ndarray) -> None:
        self.ink = np.asarray(ink, bool)
        self._dil: np.ndarray | None = None
        self._rot: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def dil(self) -> np.ndarray:
        if self._dil is None:
            self._dil = ndimage.binary_dilation(self.ink, structure=np.ones((3, 3), bool))
        return self._dil

    @property
    def rotated(self) -> tuple[np.ndarray, np.ndarray]:
        if self._rot is None:
            self._rot = (
                np.ascontiguousarray(np.rot90(self.ink, -1)),
                np.ascontiguousarray(np.rot90(self.dil, -1)),
            )
        return self._rot

    def find(
        self, quad: list[Point], angle_deg: float
    ) -> tuple[tuple[Point, Point], float] | None:
        """Endpoints (ticks) of the span measured by a dimension text, image space, plus a score
        in [0, 1]. Handles horizontal (about 0°) and bottom-to-top (about 90°) text."""
        a = ((angle_deg + 180) % 360) - 180
        if abs(a) <= 20:
            res = find_span_horizontal(self.ink, bbox(quad), self.dil)
            if res is None:
                return None
            (xl, xr), y, s = res
            return ((xl, y), (xr, y)), s
        if abs(a - 90) <= 20:
            H = self.ink.shape[0]
            rot, rdil = self.rotated
            q = [original_to_cw(p, H) for p in quad]
            res = find_span_horizontal(rot, bbox(q), rdil)
            if res is None:
                return None
            (xl, xr), y, s = res
            return (cw_to_original((xl, y), H), cw_to_original((xr, y), H)), s
        return None


def find_dimension_span(
    ink: np.ndarray, quad: list[Point], angle_deg: float
) -> tuple[tuple[Point, Point], float] | None:
    """One-off convenience wrapper around :class:`DimensionLineFinder`."""
    return DimensionLineFinder(ink).find(quad, angle_deg)


__all__ = ["DimensionLineFinder", "find_dimension_span", "find_span_horizontal", "quad_size"]
