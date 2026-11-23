"""Scan-like degradation of a clean plan image: rotation, uneven paper, stains, a fold line, blur,
noise and JPEG artefacts. ``strength`` in [0, 1] scales every effect; 0 returns the image as is."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


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
    if not 0.0 <= strength <= 1.0:
        raise ValueError("degrade strength must be in [0, 1]")
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
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255).astype(
        np.float32
    )
    info.rotation_deg = angle
    info.affine_px = m.tolist()

    # 2. paper: uneven background (low-frequency illumination), stains, a fold line
    paper = np.ones((h, w), np.float32)
    amp = 0.25 * amount()
    gx, gy = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32),
                         np.linspace(-1, 1, h, dtype=np.float32))
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
