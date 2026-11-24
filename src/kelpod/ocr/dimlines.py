"""Find the dimension line under a dimension number and the two ticks of its span.

Convention (``docs/ARCHITECTURE.md``): the value sits centred above a thin dimension line; each
measured point has a short 45° tick stroke across the line and usually an extension line. So, in the
text's own upright frame, we look for a long thin ink run just below the text, then for crossing
strokes along it, and take the nearest crossing stroke on each side of the text centre.

Vertical chains (text reading bottom-to-top) are handled by turning the ink mask 90° clockwise,
which makes them horizontal, and mapping the result back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from kelpod.contracts import Point
from kelpod.ocr.engines import bbox, cw_to_original, original_to_cw


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


@dataclass
class LineTicks:
    """A dimension line found under a text, in the text's upright frame."""

    ticks: list[float]
    """x of the crossing strokes (ticks / extension lines) along the line."""
    y: float
    """y of the line centre."""
    cx: float
    """x of the text centre."""
    text_h: float


def pick_span(
    lt: LineTicks, expected_len: float | None = None
) -> tuple[float, float, float] | None:
    """Choose the measured span ``(x_left, x_right, score)`` among the ticks.

    Without ``expected_len``: the nearest tick on each side of the text centre, rejected when
    clearly asymmetric (a missed tick). With ``expected_len`` (value x a page-wide px/m ratio):
    the pair of ticks (up to two apart) whose length matches best and whose middle is near the
    text centre; ``None`` if no pair is within 8 %.
    """
    t, cx, th = lt.ticks, lt.cx, lt.text_h
    if expected_len is None:
        left = [x for x in t if x < cx - 1]
        right = [x for x in t if x > cx + 1]
        if not left or not right:
            return None
        xl, xr = max(left), min(right)
        dl, dr = cx - xl, xr - cx
        # The text is centred on its span, so a missed tick shows up as asymmetry. Allow a text
        # height of slack for short spans (OCR boxes are a few pixels loose).
        asym = abs(dl - dr) / (dl + dr)
        if asym > 0.15 and abs(dl - dr) > th:
            return None
        return float(xl), float(xr), float(1.0 - asym)
    best = None
    for i in range(len(t)):
        for j in range(i + 1, min(len(t), i + 3)):
            length = t[j] - t[i]
            err = abs(length - expected_len) / expected_len
            off = abs((t[i] + t[j]) / 2 - cx)
            if err > 0.08 or off > max(length / 2, th):
                continue
            cost = err + 0.5 * off / max(length, th)
            if best is None or cost < best[0]:
                best = (cost, t[i], t[j], err)
    if best is None:
        return None
    return float(best[1]), float(best[2]), float(1.0 - best[3])


def line_ticks_horizontal(
    ink: np.ndarray, box: tuple[float, float, float, float], dil: np.ndarray | None = None
) -> LineTicks | None:
    """Horizontal case. ``box`` = (x0, y0, x1, y1) of upright text; ``dil`` is ``ink`` dilated by
    3x3 (computed if not given). Finds the dimension line under (else above) the text and the
    crossing strokes on it."""
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
    peaks, _ = find_peaks(np.concatenate([[0.0], score, [0.0]]), height=0.7, distance=3)
    ticks = [float(xs[p - 1]) for p in peaks]
    return LineTicks(ticks=ticks, y=float(yc), cx=float(cx), text_h=float(th))


def find_span_horizontal(
    ink: np.ndarray, box: tuple[float, float, float, float], dil: np.ndarray | None = None
) -> tuple[tuple[float, float], float, float] | None:
    """``((x_left_tick, x_right_tick), y_line, score)`` for upright text, or ``None``."""
    lt = line_ticks_horizontal(ink, box, dil)
    if lt is None:
        return None
    sp = pick_span(lt)
    if sp is None:
        return None
    return (sp[0], sp[1]), lt.y, sp[2]


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

    def ticks(self, quad: list[Point], angle_deg: float) -> tuple[LineTicks, bool] | None:
        """Line and ticks under a text in its upright frame; the flag is True for the
        bottom-to-top (90°) frame. ``None`` if no line is found or the angle is neither."""
        a = ((angle_deg + 180) % 360) - 180
        if abs(a) <= 20:
            lt = line_ticks_horizontal(self.ink, bbox(quad), self.dil)
            return None if lt is None else (lt, False)
        if abs(a - 90) <= 20:
            H = self.ink.shape[0]
            rot, rdil = self.rotated
            q = [original_to_cw(p, H) for p in quad]
            lt = line_ticks_horizontal(rot, bbox(q), rdil)
            return None if lt is None else (lt, True)
        return None

    def to_image(self, xl: float, xr: float, y: float, vertical: bool) -> tuple[Point, Point]:
        if not vertical:
            return (xl, y), (xr, y)
        H = self.ink.shape[0]
        return cw_to_original((xl, y), H), cw_to_original((xr, y), H)

    def find(
        self, quad: list[Point], angle_deg: float, expected_len_px: float | None = None
    ) -> tuple[tuple[Point, Point], float] | None:
        """Endpoints (ticks) of the span measured by a dimension text, image space, plus a score
        in [0, 1]. Handles horizontal (about 0°) and bottom-to-top (about 90°) text. See
        :func:`pick_span` for ``expected_len_px``."""
        found = self.ticks(quad, angle_deg)
        if found is None:
            return None
        lt, vertical = found
        sp = pick_span(lt, expected_len_px)
        if sp is None:
            return None
        return self.to_image(sp[0], sp[1], lt.y, vertical), sp[2]


def find_dimension_span(
    ink: np.ndarray, quad: list[Point], angle_deg: float
) -> tuple[tuple[Point, Point], float] | None:
    """One-off convenience wrapper around :class:`DimensionLineFinder`."""
    return DimensionLineFinder(ink).find(quad, angle_deg)


__all__ = [
    "DimensionLineFinder",
    "LineTicks",
    "find_dimension_span",
    "find_span_horizontal",
    "line_ticks_horizontal",
    "pick_span",
]
