"""Dimension values written INSIDE rooms (dotted line with arrowheads, value on the line).

Older plans have no outside dimension chains. ``interior_span`` follows the line from the value
to the left and right (or up and down for rotated text) until a tall stroke (arrowhead stem, wall,
extension line) or a gap ends it, and returns the measured span in image px.
"""

from __future__ import annotations

import numpy as np

from kelpod.contracts import Point


def _scan(ink: np.ndarray, yc: int, x_start: int, step: int, h: float, max_len: int) -> int | None:
    """Walk along row ``yc`` from ``x_start``; return the x of the end stroke (or the last ink)."""
    H, W = ink.shape
    band = max(1, int(0.12 * h))
    half = max(3, int(1.1 * h))
    gap_max = max(6, int(1.2 * h))
    last_ink, x = None, x_start
    for _ in range(max_len):
        if not 0 <= x < W:
            break
        col = ink[max(0, yc - band) : yc + band + 1, x]
        if col.any():
            last_ink = x
            up = ink[max(0, yc - half) : yc + half + 1, x]
            if up.sum() >= 0.9 * h and last_ink != x_start:  # a tall stroke: arrow stem or wall
                return x
        elif last_ink is not None and abs(x - last_ink) > gap_max:
            return last_ink
        elif last_ink is None and abs(x - x_start) > gap_max:
            return None
        x += step
    return last_ink


def interior_span(
    ink: np.ndarray, quad: list[Point], angle_deg: float = 0.0, *, min_span_h: float = 3.0
) -> tuple[Point, Point] | None:
    """Measured span ``(p0, p1)`` of the dotted dimension line through a value's text box, or
    ``None``. ``angle_deg`` 0 = horizontal text, 90 = vertical (bottom-to-top) text."""
    a = angle_deg % 180
    if min(a, 180 - a) <= 10:
        transpose = False
    elif abs(a - 90) <= 10:
        transpose = True
    else:
        return None
    img = ink.T if transpose else ink
    pts = [(p[1], p[0]) if transpose else (p[0], p[1]) for p in quad]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    h = y1 - y0
    if h < 4:
        return None
    yc = int(round((y0 + y1) / 2))
    left = _scan(img, yc, int(x0) - 2, -1, h, int(30 * h))
    right = _scan(img, yc, int(x1) + 2, 1, h, int(30 * h))
    if left is None or right is None or right - left < max(min_span_h * h, 1.4 * (x1 - x0)):
        return None
    p0, p1 = (float(left), float(yc)), (float(right), float(yc))
    if transpose:
        p0, p1 = (p0[1], p0[0]), (p1[1], p1[0])
    return p0, p1
