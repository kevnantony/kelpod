"""Synthetic floor-plan-like pages for the clean tests (drawn with OpenCV, no other module)."""

from __future__ import annotations

import cv2
import numpy as np


def mm(v: float, dpi: float) -> int:
    return int(round(v / 25.4 * dpi))


def draw_plan(dpi: float = 200.0, size_mm: tuple[float, float] = (420.0, 297.0)) -> dict:
    """A clean, straight page: hatched exterior walls, interior walls, a thin partition, a door arc,
    window lines, dimension chains with ticks, 1 px thin lines and some text.

    Returns dict with ``image`` (uint8, paper 255) and ``thin`` (bool mask of the 1 px thin lines).
    """
    w, h = mm(size_mm[0], dpi), mm(size_mm[1], dpi)
    img = np.full((h, w), 255, np.uint8)
    thin = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = mm(60, dpi), mm(50, dpi), mm(360, dpi), mm(240, dpi)
    t_ext = mm(3.65, dpi)
    t_int = mm(2.4, dpi)

    def hatched_rect(a, b, c, d):
        cv2.rectangle(img, (a, b), (c, d), 0, max(1, mm(0.35, dpi)))
        hatch = np.zeros_like(img)
        step = max(3, mm(1.2, dpi))
        for k in range(-(d - b), (c - a) + (d - b), step):
            cv2.line(hatch, (a + k, d), (a + k + (d - b), b), 255, 1)
        roi = hatch[b : d + 1, a : c + 1] > 0
        img[b : d + 1, a : c + 1][roi] = 0

    # Exterior walls (hatched).
    hatched_rect(x0, y0, x1, y0 + t_ext)
    hatched_rect(x0, y1 - t_ext, x1, y1)
    hatched_rect(x0, y0, x0 + t_ext, y1)
    hatched_rect(x1 - t_ext, y0, x1, y1)
    # Interior walls.
    xm = (x0 + x1) // 2
    hatched_rect(xm - t_int // 2, y0, xm + t_int // 2, y1)
    ym = (y0 + y1) // 2
    hatched_rect(x0, ym - t_int // 2, xm, ym + t_int // 2)
    # Light partition (outline only).
    xp = xm + (x1 - xm) // 2
    cv2.rectangle(img, (xp - mm(0.5, dpi), ym), (xp + mm(0.5, dpi), y1), 0, 1)
    # Door arc and leaf (thin).
    r = mm(9.0, dpi)
    hinge = (xm + t_int, ym)
    cv2.ellipse(thin, hinge, (r, r), 0, 0, 90, 255, 1)
    cv2.line(thin, hinge, (hinge[0], hinge[1] + r), 255, 1)
    # Window: three parallel thin lines in a gap of the top wall.
    for j in range(3):
        yy = y0 + int((j + 0.5) * t_ext / 3)
        cv2.line(thin, (x0 + mm(40, dpi), yy), (x0 + mm(60, dpi), yy), 255, 1)
    # Dimension chains: long thin lines with 45 deg ticks and numbers.
    yd = y1 + mm(15, dpi)
    cv2.line(thin, (x0, yd), (x1, yd), 255, 1)
    xd = x1 + mm(15, dpi)
    cv2.line(thin, (xd, y0), (xd, y1), 255, 1)
    tick = mm(1.5, dpi)
    for xx in (x0, xm, x1):
        cv2.line(img, (xx - tick, yd + tick), (xx + tick, yd - tick), 0, 2)
    for yy in (y0, ym, y1):
        cv2.line(img, (xd - tick, yy + tick), (xd + tick, yy - tick), 0, 2)
    font_scale = dpi / 200.0 * 0.9
    cv2.putText(
        img,
        "15,01",
        ((x0 + xm) // 2 - mm(5, dpi), yd - mm(1.5, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Wohnen",
        (x0 + mm(30, dpi), ym + mm(30, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "14,52 m2",
        (x0 + mm(30, dpi), ym + mm(36, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        0,
        2,
        cv2.LINE_AA,
    )
    img[thin > 0] = 0
    return {"image": img, "thin": thin > 0}


def rotate_ccw(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate CCW on screen by ``angle_deg`` about the centre, same size, white fill."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def degrade(
    gray: np.ndarray,
    dpi: float,
    seed: int = 0,
    *,
    angle_deg: float = 0.0,
    noise: float = 12.0,
    blur: float = 0.6,
    stains: int = 3,
    fold: bool = True,
    uneven: float = 0.25,
    jpeg: int | None = 70,
) -> np.ndarray:
    """Scan-like degradation: rotation, uneven background, stains, fold, blur, noise, JPEG."""
    rng = np.random.default_rng(seed)
    out = gray.astype(np.float32)
    if angle_deg:
        out = rotate_ccw(gray, angle_deg).astype(np.float32)
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Uneven background (multiplicative, like yellowed paper / lighting).
    shade = 1.0 - uneven * (0.5 * xx / w + 0.5 * yy / h)
    out = out * shade
    # Stains: dark smooth blobs (multiplicative).
    for _ in range(stains):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(mm(10, dpi), mm(40, dpi))
        depth = rng.uniform(0.15, 0.35)
        out *= 1.0 - depth * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    # Faint fold lines across the whole page.
    if fold:
        fy = int(h * 0.5) + int(rng.integers(-20, 20))
        fx = int(w * 0.5) + int(rng.integers(-20, 20))
        out[fy - 1 : fy + 2, :] *= 0.82
        out[:, fx - 1 : fx + 2] *= 0.82
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out
