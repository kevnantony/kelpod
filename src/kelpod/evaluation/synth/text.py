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


_HERSHEY_MAP = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss", "²": "2"}
    | {s: d for d, s in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹", strict=True)}
)


def render_text(text: str, digit_height_px: float) -> tuple[np.ndarray, str]:
    """Render ``text`` into an alpha mask (float32, 1 = ink), tight box plus 1 px padding.

    Returns the mask and the text actually drawn (Hershey cannot draw umlauts or superscripts).
    """
    path = font_path()
    if path is not None:
        from PIL import Image, ImageDraw

        size = max(6, int(round(digit_height_px / _DIGIT_EM)))
        font = _pil_font(path, size)
        left, top, right, bottom = font.getbbox(text)
        pad = 1
        img = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
        ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=255)
        return np.asarray(img, np.float32) / 255.0, text
    drawn = text.translate(_HERSHEY_MAP)
    face = cv2.FONT_HERSHEY_SIMPLEX
    scale = digit_height_px / 22.0  # simplex digits are ~22 px high at scale 1
    thick = max(1, int(round(scale * 2)))
    (w, h), base = cv2.getTextSize(drawn, face, scale, thick)
    img = np.zeros((h + base + 4, w + 4), np.uint8)
    cv2.putText(img, drawn, (2, h + 2), face, scale, 255, thick, cv2.LINE_AA)
    ys, xs = np.nonzero(img)
    if len(xs):
        img = img[max(ys.min() - 1, 0) : ys.max() + 2, max(xs.min() - 1, 0) : xs.max() + 2]
    return img.astype(np.float32) / 255.0, drawn
