"""Training-only page degradation for ``learned-v1``: scan geometry, own damage profiles,
selected Augraphy effects (MIT, https://github.com/sparkfish/augraphy), the generator's ordinary
``scan`` degradation, scan resolution and JPEG.

Order (all at the 300-dpi drawing resolution):

1. geometry, applied to image **and** label with the same map: hand-drawn wobble (smooth
   displacement), page curl (cylindrical bulge), perspective (photographed sheet);
2. one own damage profile (``style_v1.DAMAGE_KINDS``): ``pencil`` (grey graphite strokes with
   grain, white paper), ``ink_bleed`` (spreading ink), ``bleed_through`` (mirrored ghost of the
   back side), ``texture`` (paper fibres), ``letterpress`` (Augraphy) or nothing extra (``scan``);
3. optional Augraphy effects from ``style_v1.AUGRAPHY_ALLOWED`` (never the photocopy, low-ink,
   coloured-paper or page-border effects, which resemble held-out profiles);
4. the generator's ``scan`` degradation (rotation, uneven light, stains, fold, blur, noise, JPEG)
   at ``style.scan_degrade``; its affine is returned for the label warp;
5. scan resolution: down to ``style.scan_dpi`` and back up to 300 dpi (only below 300 dpi);
6. a final JPEG at ``style.jpeg_quality``.

None of the held-out damage profiles (``diazo``, ``photocopy``, ``faded``) is used.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from kelpod_train.labels import BG
from kelpod_train.style_v1 import AUGRAPHY_ALLOWED, TrainStyle

MAX_AUGRAPHY_PX = 30e6


def _remap(img: np.ndarray, lab: np.ndarray, mx: np.ndarray, my: np.ndarray):
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    lo = cv2.remap(lab, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
    return out, lo


def _smooth_field(h: int, w: int, rng: np.random.Generator, cell_px: float) -> np.ndarray:
    gh, gw = max(2, int(h / cell_px) + 2), max(2, int(w / cell_px) + 2)
    g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def geometry(img: np.ndarray, lab: np.ndarray, ts: TrainStyle, rng: np.random.Generator):
    h, w = img.shape
    if ts.wobble > 0 or ts.curl > 0:
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx, my = xs.copy(), ys.copy()
        if ts.wobble > 0:  # hand-drawn wobble: smooth +-1..2.5 px at 300 dpi
            amp = 1.0 + 1.5 * ts.wobble
            cell = float(rng.uniform(120, 400))
            mx += amp * _smooth_field(h, w, rng, cell)
            my += amp * _smooth_field(h, w, rng, cell)
        if ts.curl > 0:  # bulge across one axis, strongest near one edge (book / roll curl)
            amp = ts.curl * 0.008 * min(h, w)
            if rng.random() < 0.5:
                t = xs / max(w - 1, 1)
                edge = t if rng.random() < 0.5 else 1 - t
                my += amp * np.sin(np.pi * ys / max(h - 1, 1)) * edge ** 2
                mx += amp * 0.5 * edge ** 3
            else:
                t = ys / max(h - 1, 1)
                edge = t if rng.random() < 0.5 else 1 - t
                mx += amp * np.sin(np.pi * xs / max(w - 1, 1)) * edge ** 2
                my += amp * 0.5 * edge ** 3
        img, lab = _remap(img, lab, mx, my)
    if ts.perspective > 0:
        d = ts.perspective * 0.03 * min(h, w)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        lab = cv2.warpPerspective(lab, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=BG)
    return img, lab


def _augraphy(img: np.ndarray, names, seed: int) -> np.ndarray:
    if not names or img.size > MAX_AUGRAPHY_PX:
        return img
    import augraphy as A

    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    makers = {
        "InkBleed": lambda: A.InkBleed(),
        "BleedThrough": lambda: A.BleedThrough(),
        "NoiseTexturize": lambda: A.NoiseTexturize(),
        "SubtleNoise": lambda: A.SubtleNoise(),
        "Letterpress": lambda: A.Letterpress(),
        "Markup": lambda: A.Markup(),
        "Scribbles": lambda: A.Scribbles(),
    }
    for n in names:
        if n not in AUGRAPHY_ALLOWED:
            raise ValueError(f"Augraphy effect {n} is not allowed")
        try:
            out = makers[n]()(img, force=True)
        except Exception:  # noqa: BLE001 - an augmentation failure keeps the page as is
            continue
        if out is None:
            continue
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if out.shape == img.shape:
            img = out.astype(np.uint8)
    return img


def damage(img: np.ndarray, ts: TrainStyle, rng: np.random.Generator, seed: int) -> np.ndarray:
    s = ts.damage_strength
    kind = ts.damage
    if kind == "scan":
        return img
    h, w = img.shape
    x = img.astype(np.float32)
    ink = (255.0 - x) / 255.0
    if kind == "pencil":
        # graphite: grey, grainy strokes on white paper (never tinted paper)
        grain = np.clip(rng.normal(0, 0.18 * s, (h, w)).astype(np.float32), -0.35, 0.35)
        press = 0.85 + 0.15 * np.tanh(_smooth_field(h, w, rng, 600.0))
        ink = ink * np.clip((0.75 + 0.15 * (1 - s)) * press + grain, 0.0, 1.0)
        ink = cv2.GaussianBlur(ink, (0, 0), 0.6)
        x = 255.0 * (1.0 - ink)
    elif kind == "ink_bleed":
        sig = 0.6 + 1.0 * s
        spread = cv2.GaussianBlur(ink, (0, 0), sig)
        ink = np.clip(np.maximum(ink, spread * (1.3 + 0.7 * s)), 0, 1)
        x = 255.0 * (1.0 - ink)
    elif kind == "bleed_through":
        ghost = cv2.GaussianBlur(ink[:, ::-1] if rng.random() < 0.5 else ink[::-1, :], (0, 0),
                                 1.5 + 2 * s)
        x = 255.0 * (1.0 - np.clip(ink + ghost * (0.12 + 0.18 * s), 0, 1))
    elif kind == "texture":
        fib = _smooth_field(h, w, rng, 6.0) * 0.6 + _smooth_field(h, w, rng, 40.0) * 0.4
        x = x * (1.0 - 0.05 * s * (np.tanh(fib) + 1))
    elif kind == "letterpress":
        return _augraphy(img, ["Letterpress"], seed)
    else:
        raise ValueError(f"unknown train damage {kind!r}")
    return np.clip(x, 0, 255).astype(np.uint8)


def degrade_page(img: np.ndarray, lab: np.ndarray, ts: TrainStyle, seed: int):
    """Returns (degraded image, label in the pre-affine grid, 2x3 affine of the scan step)."""
    from kelpod.evaluation.synth.degrade import degrade as scan_degrade

    rng = np.random.default_rng([int(seed), 0x64656731])
    img, lab = geometry(img, lab, ts, rng)
    img = damage(img, ts, rng, seed)
    img = _augraphy(img, ts.augraphy, int(seed) * 7919 + 17)
    out, info = scan_degrade(img, float(min(ts.scan_degrade, 1.0)),
                             np.random.default_rng([int(seed), 0x64656772]))
    if ts.scan_dpi < 290:
        h, w = out.shape
        f = ts.scan_dpi / 300.0
        small = cv2.resize(out, (max(1, round(w * f)), max(1, round(h * f))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if ts.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(ts.jpeg_quality)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out, lab, info.affine_px
