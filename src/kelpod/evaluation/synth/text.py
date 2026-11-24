"""German number formats and text rendering (DejaVu Sans if present, OpenCV Hershey last)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

LengthStyle = Literal["cm", "sup", "m"]
"""How lengths are written: ``cm`` (``88,5`` / ``1,015``), ``sup`` (``88⁵`` / ``1,01⁵``) or
``m`` (``0,885`` / ``1,015``)."""
SillStyle = Literal["m", "cm"]

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_length(value_m: float, style: LengthStyle = "cm") -> str:
    """German dimension text for a length in metres, e.g. ``3.51 -> "3,51"``.

    >>> format_length(0.885, "cm"), format_length(0.885, "sup"), format_length(0.885, "m")
    ('88,5', '88⁵', '0,885')
    >>> format_length(1.015, "sup"), format_length(0.76), format_length(11.24)
    ('1,01⁵', '76', '11,24')
    """
    mm = int(round(value_m * 1000))
    cm, rest = divmod(mm, 10)
    if mm < 1000 and style != "m":
        if rest == 0:
            return str(cm)
        return f"{cm}{str(rest).translate(_SUP)}" if style == "sup" else f"{cm},{rest}"
    text = f"{cm // 100},{cm % 100:02d}"
    if rest:
        text += str(rest).translate(_SUP) if style == "sup" else str(rest)
    return text


def format_area(area_m2: float) -> str:
    """Room-stamp area, e.g. ``14.52 -> "14,52 m²"``."""
    return f"{area_m2:.2f}".replace(".", ",") + " m²"


def format_sill(sill_m: float, style: SillStyle = "m") -> str:
    """Sill height (*Brüstungshöhe*), e.g. ``BRH 0,90`` or ``BRH 87,5``."""
    if style == "m":
        return "BRH " + f"{sill_m:.2f}".replace(".", ",")
    cm = round(sill_m * 100, 1)
    return "BRH " + (f"{int(cm)}" if cm == int(cm) else f"{cm:.1f}".replace(".", ","))


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    os.path.expanduser("~/Library/Fonts/DejaVuSans.ttf"),
    # permissive fallbacks shipped with macOS / Linux distributions
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
)

# DejaVu Sans digit height is ~0.73 em
_DIGIT_EM = 0.73


@lru_cache(maxsize=1)
def font_path() -> str | None:
    """First available TrueType font (``KELPOD_SYNTH_FONT`` overrides), or None for Hershey."""
    env = os.environ.get("KELPOD_SYNTH_FONT")
    if env:
        return env if env.lower() != "hershey" and Path(env).is_file() else None
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    return None


@lru_cache(maxsize=32)
def _pil_font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


@lru_cache(maxsize=8)
def _hershey_digit_px(face: int) -> float:
    return float(cv2.getTextSize("0", face, 1.0, 1)[0][1])


_HERSHEY_MAP = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
)
_FROM_SUP = {s: d for d, s in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹", strict=True)}
SUP_SCALE = 0.62
"""Superscript digits are drawn as normal digits at this fraction of the size ..."""
SUP_RAISE = 1.0 - SUP_SCALE
"""... with their baseline raised by this fraction of the digit height (tops aligned)."""


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split into (normal text, False) and (superscript digits as normal digits, True) runs.

    Many fonts lack the superscript glyphs (they show as empty boxes), so they are never drawn
    as such.
    """
    runs: list[tuple[str, bool]] = []
    for ch in text:
        sup = ch in _FROM_SUP
        ch = _FROM_SUP.get(ch, ch)
        if runs and runs[-1][1] == sup:
            runs[-1] = (runs[-1][0] + ch, sup)
        else:
            runs.append((ch, sup))
    return runs


def _crop(img: np.ndarray, pad: int = 1) -> np.ndarray:
    ys, xs = np.nonzero(img)
    if not len(xs):
        return img[:1, :1] * 0
    return img[
        max(ys.min() - pad, 0) : ys.max() + 1 + pad, max(xs.min() - pad, 0) : xs.max() + 1 + pad
    ]


@lru_cache(maxsize=32)
def _digit_em(path: str) -> float:
    """Digit height of a font as a fraction of its size (0.73 for DejaVu Sans)."""
    f = _pil_font(path, 200)
    _, top, _, bottom = f.getbbox("0123456789", anchor="ls")
    return (bottom - top) / 200


def _resolve_font(font: str | None) -> tuple[str | None, int]:
    """(TrueType path or None for Hershey, Hershey face) for a style font spec."""
    if font is None:
        return font_path(), cv2.FONT_HERSHEY_SIMPLEX
    if font.startswith("hershey:"):
        return None, int(font.split(":", 1)[1])
    if not Path(font).is_file():
        raise FileNotFoundError(f"style font {font} is not installed")
    return font, cv2.FONT_HERSHEY_SIMPLEX


def render_text(
    text: str, digit_height_px: float, font: str | None = None
) -> tuple[np.ndarray, str]:
    """Render ``text`` into an alpha mask (float32, 1 = ink), cropped to the ink plus 1 px.

    Superscript digits (``36⁵``, ``m²``) are drawn as smaller, raised normal digits. Returns the
    mask and the text as drawn: the input text, except that Hershey spells out umlauts.
    ``font`` is a style font spec (see :mod:`kelpod.evaluation.synth.styles`); ``None`` = default.
    """
    runs = _runs(text)
    path, face = _resolve_font(font)
    if path is not None:
        from PIL import Image, ImageDraw

        em = _DIGIT_EM if font is None else _digit_em(path)
        size = max(6, int(round(digit_height_px / em)))
        sup_size = max(5, int(round(size * SUP_SCALE)))
        fonts = {False: _pil_font(path, size), True: _pil_font(path, sup_size)}
        width = sum(int(np.ceil(fonts[sup].getlength(r))) + 2 for r, sup in runs) + 8
        height = 2 * size + 8
        base = size + 4  # baseline
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        x = 4.0
        for r, sup in runs:
            y = base - (SUP_RAISE * digit_height_px if sup else 0.0)
            draw.text((x, y), r, font=fonts[sup], fill=255, anchor="ls")
            x += fonts[sup].getlength(r)
        return _crop(np.asarray(img, np.float32) / 255.0), text
    drawn = text.translate(_HERSHEY_MAP)
    # simplex digits are ~22 px high at scale 1; other faces are measured
    ref = 22.0 if face == cv2.FONT_HERSHEY_SIMPLEX else _hershey_digit_px(face)
    scale = digit_height_px / ref
    thick = max(1, int(round(scale * 2)))
    sup_scale = scale * SUP_SCALE
    sup_thick = max(1, int(round(sup_scale * 2)))
    plain = [(r.translate(_HERSHEY_MAP), sup) for r, sup in runs]
    sizes = [
        cv2.getTextSize(r, face, sup_scale if sup else scale, sup_thick if sup else thick)[0]
        for r, sup in plain
    ]
    width = sum(w for w, _ in sizes) + 8
    height = int(2 * digit_height_px) + 12
    base = int(round(1.3 * digit_height_px)) + 4
    img = np.zeros((height, width), np.uint8)
    x = 4
    for (r, sup), (w, _) in zip(plain, sizes, strict=True):
        y = base - int(round(SUP_RAISE * digit_height_px)) if sup else base
        sc, th = (sup_scale, sup_thick) if sup else (scale, thick)
        cv2.putText(img, r, (x, y), face, sc, 255, th, cv2.LINE_AA)
        x += w
    return _crop(img.astype(np.float32) / 255.0), drawn
