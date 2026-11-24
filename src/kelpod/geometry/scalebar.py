"""Graphic scale bars ("Maßstab": a thin line with equally spaced ticks, optionally numbered).

Historic plans often have no dimension chains and no usable file dpi, only a bar like
``|-|-|-|-|-|-|-|`` below the plan. ``detect_scale_bar`` finds it in the ink mask,
``read_bar_numbers`` reads its labels if OCR found any, and ``assume_tick_value`` picks the tick
value (0.5 / 1 / 2 / 5 / 10 m) that makes the thickest walls plausible when no number is legible
(an ASSUMPTION, reported as such by the caller).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np

from kelpod.contracts import TextItem

TICK_VALUES_M = (0.5, 1.0, 2.0, 5.0, 10.0)
WALL_TH_RANGE_M = (0.2, 0.65)
WALL_TH_PRIOR_M = 0.38


@dataclass
class ScaleBar:
    y: float
    """Row of the bar line (px)."""
    ticks_x: list[float]
    """Tick positions along the bar (px), sorted."""
    tick_px: float
    """Median spacing between neighbouring ticks (px)."""
    tick_height_px: float
    x0: float
    x1: float


def _peaks(profile: np.ndarray, thr: float) -> list[float]:
    """Centres of the runs where ``profile`` exceeds ``thr``."""
    on = np.concatenate([[0], (profile > thr).astype(np.int8), [0]])
    d = np.diff(on)
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(s + e - 1) / 2.0 for s, e in zip(starts, ends, strict=True)]


def detect_scale_bar(
    ink: np.ndarray, *, min_ticks: int = 6, min_len_frac: float = 0.05, max_len_frac: float = 0.6
) -> ScaleBar | None:
    """Find the thin horizontal bar with regularly spaced ticks (>= ``min_ticks`` ticks)."""
    h, w = ink.shape
    m = ink.astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best: ScaleBar | None = None
    for i in range(1, n):
        x, y, cw, ch, _ = (int(v) for v in st[i])
        if cw < min_len_frac * w or cw > max_len_frac * w or ch > 0.04 * h or cw < 12 * ch:
            continue
        comp = _component_mask(m, x, y, cw, ch)
        col = comp.sum(axis=0).astype(float)
        base = float(np.median(col[col > 0]))
        if base < 1:
            continue
        ticks = _peaks(col, base + max(1.5, 0.5 * base))
        if len(ticks) < min_ticks:
            continue
        gaps = np.diff(ticks)
        med = float(np.median(gaps))
        if med < 3 or np.mean(np.abs(gaps - med) <= 0.2 * med) < 0.8:
            continue
        rows = comp.sum(axis=1)
        cand = ScaleBar(
            y=y + float(np.argmax(rows)),
            ticks_x=[x + t for t in ticks],
            tick_px=med,
            tick_height_px=float(col.max() - base),
            x0=float(x),
            x1=float(x + cw),
        )
        if best is None or len(cand.ticks_x) > len(best.ticks_x):
            best = cand
    return best


def _component_mask(m: np.ndarray, x: int, y: int, cw: int, ch: int) -> np.ndarray:
    """Ink of the connected component that fills the given bbox (re-labelled locally)."""
    sub = m[y : y + ch, x : x + cw]
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if n <= 1:
        return sub
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


_INT = re.compile(r"^\d{1,3}$")


def read_bar_numbers(bar: ScaleBar, items: list[TextItem]) -> float | None:
    """Metres per pixel from integer labels near the bar (e.g. ``0 5 10 15`` above/below the ticks).

    Needs >= 3 numbers that fit a straight line (value = a * x + b); returns ``a`` (metres per px)
    or ``None``. The unit is taken to be metres."""
    pts: list[tuple[float, float]] = []
    band = max(6 * bar.tick_height_px, 0.9 * bar.tick_px)
    for it in items:
        t = it.text.strip().rstrip("m").strip()
        if not _INT.match(t):
            continue
        cx = float(np.mean([p[0] for p in it.quad]))
        cy = float(np.mean([p[1] for p in it.quad]))
        if bar.x0 - bar.tick_px <= cx <= bar.x1 + bar.tick_px and abs(cy - bar.y) <= band:
            pts.append((cx, float(t)))
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    a, b = np.polyfit(x, v, 1)
    if a <= 0 or np.max(np.abs(a * x + b - v)) > 0.25 * max(1.0, abs(np.median(np.diff(np.sort(v))))):
        return None
    return float(a)


def assume_tick_value(bar: ScaleBar, wall_th_px: float | None) -> float | None:
    """Metres per tick that makes ``wall_th_px`` closest to a plausible wall thickness."""
    if not wall_th_px or wall_th_px <= 0:
        return None
    best, best_err = None, math.inf
    for v in TICK_VALUES_M:
        th = wall_th_px * v / bar.tick_px
        if not WALL_TH_RANGE_M[0] <= th <= WALL_TH_RANGE_M[1]:
            continue
        err = abs(math.log(th / WALL_TH_PRIOR_M))
        if err < best_err:
            best, best_err = v, err
    return best


def bar_scale(
    ink: np.ndarray, items: list[TextItem], wall_th_px: float | None
) -> tuple[float, str, ScaleBar] | None:
    """(m_per_px, source, bar): source ``scale_bar`` if numbers were read, else
    ``scale_bar_assumed`` (tick value guessed from wall thickness)."""
    bar = detect_scale_bar(ink)
    if bar is None:
        return None
    a = read_bar_numbers(bar, items)
    if a is not None:
        return a, "scale_bar", bar
    v = assume_tick_value(bar, wall_th_px)
    if v is None:
        return None
    return v / bar.tick_px, "scale_bar_assumed", bar


__all__ = ["ScaleBar", "assume_tick_value", "bar_scale", "detect_scale_bar", "read_bar_numbers"]
