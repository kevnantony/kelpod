"""Scale (metres per pixel) and the image -> plan transform."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kelpod.contracts import DimensionLabel, Point, RasterPage

from .params import GeometryParams


@dataclass
class Frame:
    """Image space (px, y down) -> plan space (m, y up), with an optional residual rotation
    about the image centre (removes skew that the clean step left behind)."""

    m_per_px: float
    height: int
    cx: float = 0.0
    cy: float = 0.0
    rot_rad: float = 0.0
    fx: tuple[np.ndarray, np.ndarray] | None = field(default=None)
    fy: tuple[np.ndarray, np.ndarray] | None = field(default=None)
    """Piecewise-linear corrections from the dimension fit (old -> new coordinate)."""

    def rotate_px(self, p: Point) -> Point:
        if not self.rot_rad:
            return (float(p[0]), float(p[1]))
        c, s = math.cos(self.rot_rad), math.sin(self.rot_rad)
        dx, dy = p[0] - self.cx, p[1] - self.cy
        return (self.cx + c * dx - s * dy, self.cy + s * dx + c * dy)

    def to_m_raw(self, p: Point) -> Point:
        """Image px -> plan metres before the dimension fit."""
        x, y = self.rotate_px(p)
        return (x * self.m_per_px, (self.height - y) * self.m_per_px)

    def correct(self, p: Point) -> Point:
        """Apply the dimension-fit correction to a plan point."""
        return (_apply(self.fx, p[0]), _apply(self.fy, p[1]))

    def to_m(self, p: Point) -> Point:
        return self.correct(self.to_m_raw(p))


def _apply(f: tuple[np.ndarray, np.ndarray] | None, v: float) -> float:
    if f is None or len(f[0]) == 0:
        return float(v)
    old, new = f
    if len(old) == 1:
        return float(v + new[0] - old[0])
    if v <= old[0]:
        return float(v + new[0] - old[0])
    if v >= old[-1]:
        return float(v + new[-1] - old[-1])
    return float(np.interp(v, old, new))


def make_correction(pairs: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Monotone piecewise-linear map from (old, new) coordinate pairs."""
    if not pairs:
        return None
    pairs = sorted(pairs)
    old = np.array([p[0] for p in pairs], dtype=float)
    new = np.array([p[1] for p in pairs], dtype=float)
    keep = np.concatenate([[True], np.diff(old) > 1e-9])
    old, new = old[keep], new[keep]
    new = np.maximum.accumulate(new)  # never fold the plan over itself
    return old, new


def default_m_per_px(page: RasterPage) -> float:
    dpi = page.dpi if page.dpi and page.dpi > 0 else 200.0
    return 0.0254 / dpi * 100.0


def estimate_scale(
    page: RasterPage, dims: list[DimensionLabel], prm: GeometryParams
) -> tuple[float, str, list[str]]:
    """Metres per pixel: from the dimension spans if they agree, else from the plan scale."""
    issues: list[str] = []
    base = page.m_per_px_from_scale
    ratios = []
    for d in dims:
        if d.line is None or not (d.value_m and d.value_m > 0) or not math.isfinite(d.value_m):
            continue
        (x0, y0), (x1, y1) = d.line
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 5:
            continue
        ratios.append(d.value_m / length)
    if ratios:
        r = np.array(ratios)
        med = float(np.median(r))
        inl = r[np.abs(r / med - 1.0) <= prm.scale_inlier_rel]
        enough = (len(inl) >= 2 and len(inl) >= 0.5 * len(r)) or (len(inl) >= 1 and base is None)
        if enough:
            m = float(np.median(inl))
            if base is not None and abs(m / base - 1.0) > 0.15:
                issues.append(
                    f"The dimensions imply a scale {abs(m / base - 1.0) * 100:.0f} % away from "
                    f"1:{page.plan_scale:g}; the dimensions were used. Check the plan scale."
                )
            return m, "dimensions", issues
        issues.append(
            f"The {len(r)} dimension labels with measured spans disagree on the scale; "
            "the plan scale was used instead."
        )
    if base is not None and base > 0:
        return float(base), "plan_scale", issues
    issues.append("No plan scale and no usable dimensions: assumed 1:100.")
    return default_m_per_px(page), "default", issues


def estimate_skew_deg(walls_px: list[tuple[Point, Point]], max_deg: float) -> float:
    """Weighted median deviation of near-axis walls from 0/90 degrees (image space).

    The angle error of a piece falls with 1/length, so weights are length squared
    (inverse variance); short pieces would otherwise invent a skew on straight plans."""
    devs, wts = [], []
    for p0, p1 in walls_px:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        a = math.degrees(math.atan2(dy, dx))
        r = ((a + 45.0) % 90.0) - 45.0
        if abs(r) <= max_deg:
            devs.append(r)
            wts.append(length * length)
    if not devs:
        return 0.0
    order = np.argsort(devs)
    d = np.asarray(devs)[order]
    wt = np.asarray(wts)[order]
    med = float(d[np.searchsorted(np.cumsum(wt), wt.sum() / 2.0)])
    keep = np.abs(d - med) <= 2.0  # weighted mean of the inliers: uses every piece's evidence
    return float(np.sum(d[keep] * wt[keep]) / np.sum(wt[keep]))
