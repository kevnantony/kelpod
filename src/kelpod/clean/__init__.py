"""Step 1: load and clean scans (PDF/PNG/JPG/TIFF) into :class:`~kelpod.contracts.RasterPage`."""

from __future__ import annotations

from pathlib import Path

from kelpod.contracts import RasterPage


def run(
    path: str | Path, *, dpi: float = 200.0, plan_scale: float | None = 100.0
) -> list[RasterPage]:
    """Rasterise every page of ``path`` (PDF via pypdfium2, images via OpenCV/Pillow), then deskew,
    denoise, remove stains/fold marks and resample to ``dpi``. One RasterPage per page."""
    raise NotImplementedError
