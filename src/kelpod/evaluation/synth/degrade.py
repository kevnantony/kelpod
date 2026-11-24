"""Scan-like degradation of a clean plan image: rotation, uneven paper, stains, a fold line, blur,
noise and JPEG artefacts. ``strength`` in [0, 1.2] scales every effect (1 = the original "strong"
setting; up to 1.2 for the harder val/test splits); 0 returns the image as is.

:func:`apply_damage` adds the **held-out** scan-damage profiles (``diazo``, ``photocopy``,
``faded``; see :mod:`kelpod.evaluation.synth.styles`) on top. They never touch geometry, so
``Degradation.affine_px`` stays valid."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MAX_STRENGTH = 1.2


class Degradation(BaseModel):
    """What was done to the clean image (so tests can map ground truth into the degraded image)."""

    model_config = ConfigDict(extra="forbid")

    strength: float = 0.0
    rotation_deg: float = 0.0
    """Rotation applied to the scan, counter-clockwise positive (as ``RasterPage.deskew_deg``)."""
    affine_px: list[list[float]] = Field(default_factory=lambda: [[1, 0, 0], [0, 1, 0]])
    """2x3 matrix mapping clean-image px to degraded-image px (same image size)."""
    background_amp: float = 0.0
    n_stains: int = 0
    fold: bool = False
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality: int | None = None


def degrade(
    img: np.ndarray, strength: float, rng: np.random.Generator
) -> tuple[np.ndarray, Degradation]:
    """Return a degraded copy of the grayscale uint8 ``img`` and the parameters used."""
    if not 0.0 <= strength <= MAX_STRENGTH:
        raise ValueError(f"degrade strength must be in [0, {MAX_STRENGTH}]")
    info = Degradation(strength=strength)
    if strength == 0.0:
        return img.copy(), info
    h, w = img.shape
    s = strength

    def amount() -> float:
        return s * float(rng.uniform(0.5, 1.0))

    # 1. the sheet lies slightly crooked on the scanner
    angle = float(rng.choice([-1.0, 1.0])) * 3.0 * amount()
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255).astype(np.float32)
    info.rotation_deg = angle
    info.affine_px = m.tolist()

    # 2. paper: uneven background (low-frequency illumination), stains, a fold line
    paper = np.ones((h, w), np.float32)
    amp = 0.25 * amount()
    gx, gy = np.meshgrid(
        np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32)
    )
    a, b, c = rng.uniform(-1, 1, 3)
    field = a * gx + b * gy + c * (gx * gx + gy * gy) / 2
    field = (field - field.min()) / max(float(field.max() - field.min()), 1e-6)
    paper -= amp * field
    info.background_amp = amp
    n_stains = int(rng.integers(0, 4) * s + 0.5)
    for _ in range(n_stains):
        stain = np.zeros((h, w), np.float32)
        centre = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (int(rng.uniform(0.03, 0.12) * w), int(rng.uniform(0.03, 0.12) * h))
        cv2.ellipse(stain, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        k = max(axes) | 1
        stain = cv2.GaussianBlur(stain, (0, 0), k / 4)
        paper -= stain * float(rng.uniform(0.08, 0.25)) * s
    info.n_stains = n_stains
    if rng.random() < s:
        info.fold = True
        vertical = rng.random() < 0.5
        pos = int(rng.uniform(0.3, 0.7) * (w if vertical else h))
        band = np.zeros((h, w), np.float32)
        if vertical:
            band[:, pos - 2 : pos + 2] = 1.0
        else:
            band[pos - 2 : pos + 2, :] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), 6.0) * 4 + cv2.GaussianBlur(band, (0, 0), 1.0)
        paper -= np.clip(band, 0, 1) * 0.18 * s
    out = out * np.clip(paper, 0.3, 1.0)

    # 3. optics and sensor: blur, noise, JPEG
    sigma = 1.2 * amount()
    if sigma > 0.25:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        info.blur_sigma = sigma
    noise = 12.0 * amount()
    out = out + rng.normal(0.0, noise, out.shape).astype(np.float32)
    info.noise_sigma = noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    quality = int(round(95 - 60 * amount()))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        info.jpeg_quality = quality
    return out, info


def _low_freq(h: int, w: int, rng: np.random.Generator, cells: int = 6) -> np.ndarray:
    """Smooth random field in [0, 1] (bicubic upsampling of a coarse grid)."""
    grid = rng.uniform(0, 1, (cells, cells)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / max(hi - lo, 1e-6)


def apply_damage(
    img: np.ndarray, profile: str, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Held-out scan-damage profile on top of :func:`degrade` (``scan`` = nothing extra)."""
    if profile == "scan" or strength <= 0.0:
        return img
    s = min(float(strength), MAX_STRENGTH)
    h, w = img.shape
    out = img.astype(np.float32)
    ink = (255.0 - out) / 255.0  # 1 = black
    if profile == "diazo":
        # Lichtpause: pale lines on tinted, mottled paper with fine grain
        ink *= 1.0 - 0.45 * s * float(rng.uniform(0.7, 1.0))
        paper = 255.0 * (1.0 - 0.12 * s - 0.10 * s * _low_freq(h, w, rng, 5))
        out = paper * (1.0 - ink)
        out += rng.normal(0.0, 4.0 + 6.0 * s, out.shape).astype(np.float32)
    elif profile == "photocopy":
        # hard contrast, toner speckle, streaks along the feed direction, dark page edge
        gamma = 1.0 + 1.5 * s
        out = 255.0 * (1.0 - np.clip(ink * gamma, 0.0, 1.0) ** (1.0 / gamma))
        speckle = rng.random((h, w)) < 0.0015 * s
        out[speckle] = rng.uniform(0, 90, int(speckle.sum()))
        speckle = cv2.dilate(speckle.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        out[speckle] = np.minimum(out[speckle], 120)
        vertical = bool(rng.random() < 0.5)
        for _ in range(int(rng.integers(1, 4))):
            pos = int(rng.uniform(0.05, 0.95) * (w if vertical else h))
            width = int(rng.integers(2, 7))
            dark = float(rng.uniform(30, 90)) * s
            if vertical:
                out[:, pos : pos + width] -= dark
            else:
                out[pos : pos + width, :] -= dark
        edge = int(rng.uniform(0.01, 0.03) * min(h, w))
        side = int(rng.integers(0, 4))
        ramp = np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None] * 160.0 * s
        if side == 0:
            out[:edge, :] -= ramp
        elif side == 1:
            out[h - edge :, :] -= ramp[::-1]
        elif side == 2:
            out[:, :edge] -= ramp.T
        else:
            out[:, w - edge :] -= ramp.T[:, ::-1]
    elif profile == "faded":
        # patchy fading (sun, old toner) and small breaks in the lines
        fade = _low_freq(h, w, rng, 4) ** 2 * 0.75 * s
        breaks = (rng.random((h // 8 + 1, w // 8 + 1)) < 0.04 * s).astype(np.uint8)
        breaks = cv2.resize(breaks, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        breaks = cv2.GaussianBlur(breaks, (0, 0), 1.5)
        ink *= np.clip(1.0 - fade - 0.85 * breaks, 0.0, 1.0)
        out = 255.0 - 255.0 * ink
    else:
        raise ValueError(f"unknown damage profile {profile!r}")
    return np.clip(out, 0, 255).astype(np.uint8)
