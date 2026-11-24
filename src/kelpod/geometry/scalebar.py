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
WALL_TH_PRIOR_M = 0.45
"""Historic masonry (0.38-0.64 m). Chosen with the two real dev plans in view: not independent."""


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


def _regular_run(col: np.ndarray, base: float, min_ticks: int) -> list[float] | None:
    """Longest run of >= ``min_ticks`` equally spaced tick peaks (gaps within 15 % of the run's
    median). Bars next to numbers or with unequal major/minor ticks still give their regular part."""
    best: list[float] | None = None
    for extra in (max(1.5, 0.5 * base), max(0.9, 0.2 * base)):
        pk = _peaks(col, base + extra)
        i = 0
        while i < len(pk) - 1:
            j = i + 1
            while j < len(pk):
                run = pk[i : j + 1]
                med = float(np.median(np.diff(run)))
                if med < 3 or abs((pk[j] - pk[j - 1]) - med) > 0.15 * med:
                    break
                j += 1
            run = pk[i:j]
            if len(run) >= min_ticks and (best is None or len(run) > len(best)):
                best = run
            i = max(j - 1, i + 1)
    return best


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
        comp = m[y : y + ch, x : x + cw]
        col = comp.sum(axis=0).astype(float)
        base = float(np.median(col[col > 0]))
        if base < 1:
            continue
        ticks = _regular_run(col, base, min_ticks)
        if ticks is None:
            continue
        med = float(np.median(np.diff(ticks)))
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


def typical_wall_px(walls: list[tuple[float, float]]) -> float | None:
    """Length-weighted median wall thickness (px) from ``(length_px, thickness_px)`` pairs:
    the main walls count, short thick blobs (stoves, pillars) do not."""
    pairs = sorted((t, ln) for ln, t in walls if ln > 0 and t > 0)
    if not pairs:
        return None
    half = sum(ln for _, ln in pairs) / 2
    acc = 0.0
    for t, ln in pairs:
        acc += ln
        if acc >= half:
            return t
    return pairs[-1][0]


def ink_thickness_px(ink: np.ndarray) -> float | None:
    """Typical thickness (px) of the heavy strokes (walls, poche), straight from the ink mask:
    ridge values of the distance transform that are clearly thicker than the pen lines. A
    fallback when the segmentation's wall thickness is not plausible."""
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1.0)
    v = 2.0 * dt[ridge]
    if len(v) < 50:
        return None
    pen = float(np.median(v))
    thick = v[v > 2.0 * pen]
    if len(thick) < 50:
        return None
    return float(np.median(thick))


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
        v = assume_tick_value(bar, ink_thickness_px(ink))
    if v is None:
        return None
    return v / bar.tick_px, "scale_bar_assumed", bar


__all__ = ["ScaleBar", "assume_tick_value", "typical_wall_px", "bar_scale", "detect_scale_bar", "read_bar_numbers"]
