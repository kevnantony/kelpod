"""OCR engines behind one small protocol, plus the rotation handling shared by all of them.

An engine only has to read upright (left-to-right) text in a grayscale image. Text that reads
bottom-to-top (vertical dimension chains) is handled by :func:`read_page`, which runs the engine on
the page turned by 90° as well, maps the boxes back and removes duplicates. A second engine such as
docTR or PaddleOCR can be added with :func:`register_engine` without touching the rest.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

from kelpod.contracts import Point, TextItem


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def read(self, gray: np.ndarray) -> list[TextItem]:
        """Detect and recognise text in ``gray`` (uint8, white paper). Quads in the pixel space of
        ``gray``, clockwise from the top-left of the text as read."""
        ...


class NoneEngine:
    """Reads nothing; the pipeline then relies on the plan scale and defaults."""

    name = "none"

    def read(self, gray: np.ndarray) -> list[TextItem]:
        return []


class RapidOcrEngine:
    """RapidOCR (Apache-2.0): PP-OCRv4 detector and recogniser as ONNX, CPU only.

    The models ship inside the ``rapidocr-onnxruntime`` wheel, so no download is needed. The default
    ``max_side_len`` of 2000 px would shrink an A3 page at 200 dpi and make 2.5 mm text unreadable,
    so the page is read in overlapping tiles at full resolution instead.
    """

    name = "rapidocr"

    def __init__(self, *, tile: int = 1600, overlap: int = 200, text_score: float = 0.5) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR(max_side_len=max(4000, tile + 1), text_score=text_score)
        self.tile = tile
        self.overlap = overlap

    def _read_one(self, gray: np.ndarray, ox: int, oy: int) -> list[TextItem]:
        bgr = np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
        result, _ = self._ocr(bgr)
        items: list[TextItem] = []
        for quad, text, score in result or []:
            text = str(text).strip()
            if not text:
                continue
            q = [(float(x) + ox, float(y) + oy) for x, y in quad]
            items.append(TextItem(text=text, quad=q, confidence=float(score)))
        return items

    def read(self, gray: np.ndarray) -> list[TextItem]:
        h, w = gray.shape
        if max(h, w) <= self.tile + self.overlap:
            return self._read_one(gray, 0, 0)
        step = self.tile - self.overlap
        items: list[TextItem] = []
        for y0 in range(0, max(1, h - self.overlap), step):
            for x0 in range(0, max(1, w - self.overlap), step):
                y1, x1 = min(h, y0 + self.tile), min(w, x0 + self.tile)
                sub = gray[y0:y1, x0:x1]
                if (sub < 128).mean() < 1e-4:  # blank tile
                    continue
                for it in self._read_one(sub, x0, y0):
                    # Drop boxes cut by an inner tile border; the neighbouring tile sees them whole.
                    xs = [p[0] for p in it.quad]
                    ys = [p[1] for p in it.quad]
                    m = 3
                    if (x0 > 0 and min(xs) <= x0 + m) or (y0 > 0 and min(ys) <= y0 + m):
                        continue
                    if (x1 < w and max(xs) >= x1 - m) or (y1 < h and max(ys) >= y1 - m):
                        continue
                    items.append(it)
        return dedupe(items)


_REGISTRY: dict[str, Callable[[], OcrEngine]] = {
    "rapidocr": RapidOcrEngine,
    "none": NoneEngine,
}


def register_engine(name: str, factory: Callable[[], OcrEngine]) -> None:
    """Add an engine (e.g. ``"doctr"``) that :func:`get_engine` can build by name."""
    _REGISTRY[name] = factory
    get_engine.cache_clear()


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


@lru_cache(maxsize=8)
def get_engine(name: str) -> OcrEngine:
    """Build (once) and return the engine registered under ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown OCR backend {name!r}; available: {available_engines()}") from None
    return factory()


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------


def quad_angle_deg(quad: list[Point]) -> float:
    """Reading direction of a quad in image space, counter-clockwise, in (-180, 180]."""
    (x0, y0), (x1, y1) = quad[0], quad[1]
    (x3, y3), (x2, y2) = quad[3], quad[2]
    dx = (x1 - x0 + x2 - x3) / 2
    dy = (y1 - y0 + y2 - y3) / 2
    return math.degrees(math.atan2(-dy, dx))


def quad_size(quad: list[Point]) -> tuple[float, float]:
    """(length along the reading direction, height across it)."""
    p = np.asarray(quad, float)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    return float(w), float(h)


def bbox(quad: list[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_small(a: list[Point], b: list[Point]) -> float:
    """Intersection over the smaller of the two axis-aligned boxes."""
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return iw * ih / small if small > 0 else 0.0


def dedupe(items: list[TextItem], *, thresh: float = 0.6) -> list[TextItem]:
    """Greedy non-maximum suppression: keep the more confident (then longer) of overlapping items."""
    order = sorted(items, key=lambda it: (-it.confidence, -len(it.text)))
    kept: list[TextItem] = []
    for it in order:
        if all(_overlap_small(it.quad, k.quad) < thresh for k in kept):
            kept.append(it)
    return kept


def rotate_cw(gray: np.ndarray) -> np.ndarray:
    """Turn the image 90° clockwise, so text reading bottom-to-top reads left-to-right."""
    return np.ascontiguousarray(np.rot90(gray, -1))


def cw_to_original(p: Point, height: int) -> Point:
    """Map a point of :func:`rotate_cw` output back to the original image (height ``height``)."""
    return (p[1], height - p[0])


def original_to_cw(p: Point, height: int) -> Point:
    return (height - p[1], p[0])


def read_page(gray: np.ndarray, engine: OcrEngine, *, rotated: bool = True) -> list[TextItem]:
    """Read horizontal and (if ``rotated``) bottom-to-top text; boxes in ``gray`` pixels.

    Items whose box is taller than wide in a pass are left to the other pass, which sees them
    upright. ``angle_deg`` is measured from the mapped quad (about 0 or 90).
    """
    h = gray.shape[0]
    out: list[TextItem] = []
    for it in engine.read(gray):
        w_, h_ = quad_size(it.quad)
        if rotated and h_ > 1.3 * w_:
            continue
        out.append(it.model_copy(update={"angle_deg": round(quad_angle_deg(it.quad), 2)}))
    if rotated:
        for it in engine.read(rotate_cw(gray)):
            w_, h_ = quad_size(it.quad)
            if h_ > 1.3 * w_:
                continue
            q = [cw_to_original(p, h) for p in it.quad]
            out.append(it.model_copy(update={"quad": q, "angle_deg": round(quad_angle_deg(q), 2)}))
    return dedupe(out)
