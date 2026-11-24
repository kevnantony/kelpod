"""Pillow-rendered fixtures for the OCR tests: dimension chains with ticks, room stamps, BRH."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kelpod.contracts import RasterPage

_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in _FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, px)
    return ImageFont.load_default(size=px)


@dataclass
class Chain:
    """Expected result for one rendered dimension span."""

    text: str
    value_m: float
    vertical: bool
    ticks: tuple[tuple[float, float], tuple[float, float]]
    """Tick positions in image space (as drawn)."""
    quad: list[tuple[float, float]]
    """Corners of the pasted text, clockwise from the text's top-left as read."""


@dataclass
class Fixture:
    image: np.ndarray
    chains: list[Chain] = field(default_factory=list)
    stamps: list[tuple[str, float]] = field(default_factory=list)
    sills: list[float] = field(default_factory=list)

    def page(self, dpi: float = 200.0) -> RasterPage:
        return RasterPage(image=self.image, ink=self.image < 128, dpi=dpi, source="fixture")


def _text_img(text: str, f, pad: int = 2) -> Image.Image:
    left, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - left + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - left, pad - t), text, font=f, fill=0)
    return im


def draw_chain_h(img: Image.Image, x_ticks: list[float], y: float, texts: list[str], f,
                 tick: int = 8, gap: int = 4) -> list[tuple]:
    """Horizontal dimension chain with 45° ticks and centred labels above the line."""
    d = ImageDraw.Draw(img)
    d.line([(x_ticks[0] - 15, y), (x_ticks[-1] + 15, y)], fill=0, width=2)
    for x in x_ticks:
        d.line([(x - tick, y + tick), (x + tick, y - tick)], fill=0, width=2)
        d.line([(x, y - 3 * tick), (x, y + 2 * tick)], fill=0, width=1)  # extension line
    out = []
    for xa, xb, t in zip(x_ticks[:-1], x_ticks[1:], texts, strict=True):
        ti = _text_img(t, f)
        cx = (xa + xb) / 2
        x0, y0 = int(round(cx - ti.width / 2)), int(round(y - gap - ti.height))
        img.paste(ti, (x0, y0))
        x1, y1 = x0 + ti.width, y0 + ti.height
        out.append((xa, xb, t, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    return out


def draw_chain_v(img: Image.Image, x: float, y_ticks: list[float], texts: list[str], f,
                 tick: int = 8, gap: int = 4) -> list[tuple]:
    """Vertical chain; labels rotated 90° (read bottom-to-top), left of the line."""
    d = ImageDraw.Draw(img)
    d.line([(x, min(y_ticks) - 15), (x, max(y_ticks) + 15)], fill=0, width=2)
    for y in y_ticks:
        d.line([(x - tick, y + tick), (x + tick, y - tick)], fill=0, width=2)
        d.line([(x - 3 * tick, y), (x + 2 * tick, y)], fill=0, width=1)
    out = []
    for ya, yb, t in zip(y_ticks[:-1], y_ticks[1:], texts, strict=True):
        ti = _text_img(t, f).rotate(90, expand=True)  # counter-clockwise: reads bottom-to-top
        cy = (ya + yb) / 2
        x0, y0 = int(round(x - gap - ti.width)), int(round(cy - ti.height / 2))
        img.paste(ti, (x0, y0))
        x1, y1 = x0 + ti.width, y0 + ti.height
        # Corners as read (bottom-to-top): the text's top-left is the box's bottom-left.
        out.append((ya, yb, t, [(x0, y1), (x0, y0), (x1, y0), (x1, y1)]))
    return out


def make_fixture(text_px: int = 22) -> Fixture:
    """A 1800 x 1400 sheet with a top and a left dimension chain, stamps and sill heights."""
    img = Image.new("L", (1800, 1400), 255)
    f = font(text_px)
    fx = Fixture(image=np.zeros(1))
    # A rectangle as "walls" so that the sheet looks like a plan.
    ImageDraw.Draw(img).rectangle([300, 300, 1600, 1250], outline=0, width=6)

    h_vals = [("3,51", 3.51), ("0,885", 0.885), ("4,26", 4.26), ("88,5", 0.885), ("2,01", 2.01)]
    xs = [300.0]
    for _, v in h_vals:
        xs.append(xs[-1] + v * 110)
    for xa, xb, t, q in draw_chain_h(img, xs, 200, [t for t, _ in h_vals], f):
        v = dict(h_vals)[t]
        fx.chains.append(Chain(t, v, False, ((xa, 200.0), (xb, 200.0)), q))

    v_vals = [("2,76", 2.76), ("1,26", 1.26), ("76", 0.76), ("3,01", 3.01)]
    ys = [1250.0]
    for _, v in v_vals:
        ys.append(ys[-1] - v * 110)
    for ya, yb, t, q in draw_chain_v(img, 200, ys, [t for t, _ in v_vals], f):
        v = dict(v_vals)[t]
        fx.chains.append(Chain(t, v, True, ((200.0, ya), (200.0, yb)), q))

    d = ImageDraw.Draw(img)
    for (name, area), (x, y) in zip(
        [("Wohnen", 14.52), ("Küche", 8.1), ("Schlafen", 12.75)],
        [(600, 600), (1100, 600), (850, 950)], strict=True,
    ):
        d.text((x, y), name, font=f, fill=0, anchor="mm")
        d.text((x, y + 1.6 * text_px), f"{area:.2f}".replace(".", ",") + " m²", font=f, fill=0,
               anchor="mm")
        fx.stamps.append((name, area))
    for txt, v, (x, y) in [("BRH 0,90", 0.90, (1350, 420)), ("BRH 87,5", 0.875, (1350, 1100))]:
        d.text((x, y), txt, font=f, fill=0, anchor="mm")
        fx.sills.append(v)
    fx.image = np.asarray(img, dtype=np.uint8).copy()
    return fx
