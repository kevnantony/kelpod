"""Loaders: PDF (pypdfium2) and raster images (Pillow, multi-page TIFF) -> grayscale uint8 at a
target dpi."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

DEFAULT_IMAGE_DPI = 300.0
"""Assumed resolution of raster images without usable DPI metadata."""

MAX_PAGE_MM = 1300.0
"""Longest plausible paper side (A0 is 1189 mm)."""

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load(path: str | Path, dpi: float = 200.0) -> list[np.ndarray]:
    """Load every page of ``path`` as grayscale uint8 (white paper = 255) resampled to ``dpi``."""
    path = Path(path)
    try:
        is_file = path.is_file()
    except OSError:  # e.g. PermissionError on an unreadable parent directory
        is_file = False
    if not is_file:
        raise FileNotFoundError(f"input file not found or not readable: {path}")
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path, dpi)
    if suffix in IMAGE_SUFFIXES:
        return _load_image(path, dpi)
    # Unknown suffix: sniff the header.
    with open(path, "rb") as f:
        head = f.read(5)
    if head == b"%PDF-":
        return _load_pdf(path, dpi)
    return _load_image(path, dpi)


def _load_pdf(path: Path, dpi: float) -> list[np.ndarray]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                bitmap = page.render(
                    scale=dpi / 72.0, grayscale=True, fill_color=(255, 255, 255, 255)
                )
                arr = bitmap.to_numpy()
                bitmap.close()
            finally:
                page.close()
            pages.append(_to_gray(arr))
        return pages
    finally:
        pdf.close()


def _load_image(path: Path, dpi: float) -> list[np.ndarray]:
    pages = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            src_dpi = _image_dpi(frame)
            gray = _pil_to_gray(frame)
            pages.append(resample(gray, src_dpi, dpi))
    return pages


def _image_dpi(im: Image.Image) -> float:
    """DPI from metadata, else :data:`DEFAULT_IMAGE_DPI`. Implausible values are ignored: outside
    50-2400 dpi, or implying a page larger than A0 (a typical sign of a 72/96 dpi software
    default on a real scan)."""
    info = im.info.get("dpi")
    if info is not None:
        try:
            x = float(info[0]) if isinstance(info, tuple | list) else float(info)
        except (TypeError, ValueError):
            x = 0.0
        if 50.0 <= x <= 2400.0:
            longest_mm = max(im.size) / x * 25.4
            if longest_mm <= MAX_PAGE_MM:
                return x
    return DEFAULT_IMAGE_DPI


def _pil_to_gray(im: Image.Image) -> np.ndarray:
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        # Flatten transparency onto white paper.
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64)
        hi = arr.max() if arr.max() > 255 else 255.0
        return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)
    if im.mode == "F":
        arr = np.asarray(im, dtype=np.float64)
        hi = arr.max() if arr.max() > 1.0 else 1.0
        return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8).copy()


def _to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.shape[2] == 1:
        return np.ascontiguousarray(arr[:, :, 0], dtype=np.uint8)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def resample(gray: np.ndarray, src_dpi: float, dst_dpi: float) -> np.ndarray:
    """Resize ``gray`` from ``src_dpi`` to ``dst_dpi`` (area filter when shrinking)."""
    f = dst_dpi / src_dpi
    if abs(f - 1.0) < 1e-6:
        return gray
    h, w = gray.shape
    size = (max(1, round(w * f)), max(1, round(h * f)))
    interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, size, interpolation=interp)
