"""Cleaning of one grayscale page: flatten background, deskew, suppress folds, binarise, despeckle.

All sizes are given in millimetres on paper and converted with the page ``dpi``, so the same
parameters work at 150, 200 or 300 dpi.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

MM_PER_INCH = 25.4

# Background estimation: dark features narrower than this (walls incl. hatching, text, lines) are
# treated as ink; anything wider (stains, shading, scanner borders) as background. A 36.5 cm wall
# at 1:100 is 3.65 mm wide, a 49 cm wall 4.9 mm.
BACKGROUND_KERNEL_MM = 8.0
BACKGROUND_WORK_DPI = 50.0

MAX_SKEW_DEG = 10.0
MIN_ROTATE_DEG = 0.02
"""Angles smaller than this are treated as straight and the page is not resampled."""

SPECKLE_MAX_AREA_MM2 = 0.06
"""Ink blobs smaller than this are removed (about 4 px at 200 dpi; a decimal comma is ~10 px)."""
HOLE_MAX_AREA_MM2 = 0.06
"""Paper holes inside ink smaller than this are filled."""

THRESH_RANGE = (100, 190)
"""Clamp for the global Otsu threshold on the flattened page (paper is 255 after flattening)."""
WEAK_FRACTION = 0.6
"""Weak (hysteresis) threshold = t + WEAK_FRACTION * (255 - t)."""
WEAK_SMOOTH_SIGMA = 0.8
RIDGE_FRACTION = 0.35
"""Faint pixel must be within this share of (paper - local minimum) of its 3x3 minimum."""
WEAK_MIN_LENGTH_MM = 2.5
"""Faint regions not touching strong ink are kept only if at least this long."""


class CleanResult(NamedTuple):
    image: np.ndarray
    """Cleaned grayscale, uint8, paper = 255."""
    ink: np.ndarray
    """Bool ink mask, same shape."""
    deskew_deg: float
    """Rotation removed, degrees, counter-clockwise positive."""
    threshold: int
    """Gray level used for binarisation (ink < threshold)."""


def _px(mm: float, dpi: float) -> float:
    return mm / MM_PER_INCH * dpi


def _px_area(mm2: float, dpi: float) -> float:
    return mm2 * (dpi / MM_PER_INCH) ** 2


# ------------------------------------------------------------------------------------------------
# Background and folds
# ------------------------------------------------------------------------------------------------


def flatten_background(gray: np.ndarray, dpi: float) -> np.ndarray:
    """Divide by a smooth paper estimate so that the paper becomes 255 everywhere.

    The paper estimate is a grayscale closing (max then min filter) at low resolution, which
    removes all ink narrower than :data:`BACKGROUND_KERNEL_MM` and keeps stains, yellowing,
    uneven lighting and scanner shadows, which the division then cancels.
    """
    h, w = gray.shape
    f = min(1.0, BACKGROUND_WORK_DPI / dpi)
    small = cv2.resize(gray, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(_px(BACKGROUND_KERNEL_MM, dpi * f))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=k / 4.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255.0)


def suppress_folds(flat: np.ndarray, threshold: int) -> np.ndarray:
    """Brighten rows/columns whose *paper* level is below white (faint axis-aligned fold or crease
    lines that run across the page). Ink pixels are ignored when measuring the paper level, so long
    wall or dimension lines are not touched; ink in a corrected row keeps its relative darkness."""
    out = flat
    for axis in (1, 0):  # 1: per row, 0: per column
        sub = out[:, ::2] if axis == 1 else out[::2, :]
        paper = np.where(sub < threshold, 255, sub)
        level = np.median(paper, axis=axis)
        level = np.clip(level, 1, 255)
        bad = level < 248
        if not bad.any():
            continue
        gain = np.where(bad, 255.0 / level, 1.0).astype(np.float32)
        g = gain[:, None] if axis == 1 else gain[None, :]
        out = np.clip(out.astype(np.float32) * g, 0, 255).astype(np.uint8)
    return out


def weak_threshold(t: int) -> int:
    """Upper gray level for faint ink (blurred 1 px lines), half-way between ``t`` and paper."""
    return int(round(t + WEAK_FRACTION * (255 - t)))


def hysteresis(flat: np.ndarray, t_strong: int, t_weak: int, min_extent_px: float) -> np.ndarray:
    """Ink = pixels darker than ``t_strong``, plus *faint line centres*: pixels darker than
    ``t_weak`` (on a lightly smoothed page) that are also nearly the darkest pixel of their 3x3
    neighbourhood. The ridge test keeps pale or blurred 1 px lines and hatching but rejects the grey
    anti-aliasing halo beside normal ink, so line widths are not inflated. Faint pieces that do not
    touch strong ink must be at least ``min_extent_px`` long (paper noise stays below that)."""
    smooth = cv2.GaussianBlur(flat, (0, 0), WEAK_SMOOTH_SIGMA)
    strong = flat < t_strong
    local_min = cv2.erode(smooth, np.ones((3, 3), np.uint8)).astype(np.int16)
    ridge = smooth.astype(np.int16) <= local_min + RIDGE_FRACTION * (255 - local_min)
    cand = strong | ((smooth < t_weak) & ridge)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    has_strong = np.bincount(labels[strong], minlength=n) > 0
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = has_strong | (extent >= min_extent_px)
    keep[0] = False
    return keep[labels]


def ink_threshold(flat: np.ndarray) -> int:
    """Global Otsu threshold on the flattened page, clamped to :data:`THRESH_RANGE`."""
    step = max(1, int(np.sqrt(flat.size / 2e6)))
    t, _ = cv2.threshold(flat[::step, ::step], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return int(np.clip(t, *THRESH_RANGE))


# ------------------------------------------------------------------------------------------------
# Deskew
# ------------------------------------------------------------------------------------------------


def _profile_score(xs: np.ndarray, ys: np.ndarray, angles_deg: np.ndarray, bin_px: float) -> np.ndarray:
    """Sharpness (sum of squared bin counts) of the x and y projection profiles after rotating the
    points by each angle (image coordinates, cv2 convention: positive = counter-clockwise on
    screen)."""
    scores = np.empty(len(angles_deg))
    for i, a in enumerate(np.deg2rad(angles_deg)):
        c, s = np.cos(a), np.sin(a)
        u = xs * c + ys * s
        v = -xs * s + ys * c
        hu = np.bincount(((u - u.min()) / bin_px).astype(np.int64))
        hv = np.bincount(((v - v.min()) / bin_px).astype(np.int64))
        scores[i] = float(hu @ hu) + float(hv @ hv)
    return scores


def deskew_angle(ink: np.ndarray, *, max_deg: float = MAX_SKEW_DEG, max_points: int = 400_000) -> float:
    """Skew of the drawing in degrees (counter-clockwise positive, as seen on screen).

    Maximises the sharpness of the horizontal and vertical projection profiles of the ink pixels,
    which floor plans (long axis-parallel walls and dimension lines) make very peaked at the right
    angle. Coarse-to-fine search (0.5 / 0.1 / 0.02 / 0.005 deg steps with shrinking bins). Returns 0.0 for nearly empty pages.
    """
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0
    if len(xs) > max_points:
        idx = np.random.default_rng(0).choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()

    best = 0.0
    stages = ((0.5, max_deg, 8.0), (0.1, 0.6, 2.0), (0.02, 0.12, 1.0), (0.005, 0.025, 1.0))
    for step, half, bin_px in stages:
        angles = np.arange(best - half, best + half + 1e-9, step)
        angles = angles[np.abs(angles) <= max_deg + 1e-9]
        scores = _profile_score(xs, ys, angles, bin_px)
        i = int(np.argmax(scores))
        best = float(angles[i])
    # Rotating by ``best`` straightens the drawing, so the drawing is rotated by ``-best``.
    return -best


def rotate(gray: np.ndarray, angle_deg: float, *, border: int = 255) -> np.ndarray:
    """Rotate ``gray`` about its centre by ``angle_deg`` (counter-clockwise on screen), same size."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border
    )


# ------------------------------------------------------------------------------------------------
# Binary clean-up
# ------------------------------------------------------------------------------------------------


def remove_small_components(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    """Drop 8-connected True components with area < ``max_area_px``."""
    if max_area_px <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < max_area_px
    small[0] = False  # background label
    if not small[1:].any():
        return mask
    return mask & ~small[labels]


def fill_small_holes(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    """Fill 4-connected paper holes (inside ink) with area < ``max_area_px``."""
    if max_area_px <= 1:
        return mask
    inv = ~mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=4)
    small = stats[:, cv2.CC_STAT_AREA] < max_area_px
    small[0] = False
    if not small[1:].any():
        return mask
    return mask | small[labels]


# ------------------------------------------------------------------------------------------------
# Whole page
# ------------------------------------------------------------------------------------------------


def clean_image(gray: np.ndarray, dpi: float, *, deskew: bool = True) -> CleanResult:
    """Clean one grayscale page that is already at ``dpi``.

    1. flatten background (stains, yellowing, uneven light) by dividing by a closing estimate;
    2. deskew within +-10 deg (projection profiles), page size kept, white fill;
    3. suppress faint axis-parallel fold lines;
    4. binarise with a clamped global Otsu threshold (paper is uniform after step 1);
    5. remove speckles and fill pin-holes; whiten everything away from the remaining ink.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {gray.shape}")
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    flat = flatten_background(gray, dpi)
    t = ink_threshold(flat)

    angle = 0.0
    if deskew:
        rough = flat < t
        rough = remove_small_components(rough, _px_area(SPECKLE_MAX_AREA_MM2, dpi))
        angle = deskew_angle(rough)
        if abs(angle) >= MIN_ROTATE_DEG:
            flat = rotate(flat, -angle)
        else:
            angle = 0.0

    flat = suppress_folds(flat, t)
    t = ink_threshold(flat)
    ink = hysteresis(flat, t, weak_threshold(t), _px(WEAK_MIN_LENGTH_MM, dpi))
    ink = remove_small_components(ink, _px_area(SPECKLE_MAX_AREA_MM2, dpi))
    filled = fill_small_holes(ink, _px_area(HOLE_MAX_AREA_MM2, dpi))
    holes = filled & ~ink
    ink = filled

    # Keep the grey anti-aliasing next to real ink (useful for OCR); paper elsewhere.
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    image = np.where(near, flat, 255).astype(np.uint8)
    image[holes] = np.minimum(image[holes], t - 1)
    return CleanResult(image=image, ink=ink, deskew_deg=round(float(angle), 4), threshold=t)
