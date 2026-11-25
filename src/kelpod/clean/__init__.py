"""Step 1: load and clean scans (PDF/PNG/JPG/TIFF) into :class:`~kelpod.contracts.RasterPage`.

Public API:

* :func:`run` -- file -> one cleaned :class:`RasterPage` per page (used by the pipeline).
* :func:`load` -- file -> grayscale pages resampled to the target dpi.
* :func:`clean_image` -- one grayscale page -> :class:`CleanResult` (image, ink, deskew angle).

Everything is classical image processing with OpenCV / NumPy (see ``docs/research``): background
flattening by morphological closing, deskew by projection-profile sharpness of the ink pixels,
fixed-threshold binarisation relative to the flattened paper, speckle removal by connected
components.
"""

from __future__ import annotations

from pathlib import Path

from kelpod.clean.core import CleanResult, clean_image, deskew_angle, rotate
from kelpod.clean.io import load
from kelpod.clean.sheets import PlanRegion, crop_page, split_plans
from kelpod.contracts import RasterPage

__all__ = ["PlanRegion", "crop_page", "split_plans", "CleanResult", "clean_image", "deskew_angle", "load", "rotate", "run"]


def run(
    path: str | Path, *, dpi: float = 200.0, plan_scale: float | None = 100.0
) -> list[RasterPage]:
    """Rasterise every page of ``path`` (PDF via pypdfium2, images via OpenCV/Pillow), then deskew,
    denoise, remove stains/fold marks and resample to ``dpi``. One RasterPage per page."""
    pages = load(path, dpi)
    out: list[RasterPage] = []
    for i, gray in enumerate(pages):
        res = clean_image(gray, dpi)
        out.append(
            RasterPage(
                image=res.image,
                ink=res.ink,
                dpi=float(dpi),
                source=str(path),
                page_index=i,
                deskew_deg=res.deskew_deg,
                plan_scale=plan_scale,
            )
        )
    return out
