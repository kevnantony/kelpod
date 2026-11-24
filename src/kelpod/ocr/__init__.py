"""Step 3: read dimension chains, sill heights and room stamps.

* :mod:`kelpod.ocr.engines`: ``OcrEngine`` protocol and registry (``rapidocr`` default, ``none``),
  plus the 0°/90° double pass for vertical dimension text.
* :mod:`kelpod.ocr.parse`: pure string parsers (German numbers, BRH, areas, room names).
* :mod:`kelpod.ocr.dimlines`: dimension line and tick search for a dimension text.
* :mod:`kelpod.ocr.interpret`: text items -> ``DimensionLabel`` / ``SillHeight`` / ``RoomStamp``.
"""

from __future__ import annotations

from kelpod.contracts import OcrAnnotations, RasterPage
from kelpod.ocr.engines import (
    OcrEngine,
    available_engines,
    get_engine,
    read_page,
    register_engine,
    remove_long_lines,
)
from kelpod.ocr.interpret import annotate

__all__ = ["OcrEngine", "available_engines", "get_engine", "register_engine", "run"]


def run(page: RasterPage, *, backend: str = "rapidocr") -> OcrAnnotations:
    """Detect and recognise text (including rotated text) and parse it into dimensions,
    sill heights and room stamps."""
    try:
        engine = get_engine(backend)
    except ImportError as exc:  # engine package missing: carry on without OCR
        return OcrAnnotations(backend="none", notes=[f"OCR backend {backend!r} unavailable: {exc}"])
    if backend == "none":
        return OcrAnnotations(backend="none", notes=["OCR disabled (backend 'none')"])
    # Lines longer than ~12 mm on paper are not text (labels are 2.5-5 mm high).
    min_len = round(12.0 / 25.4 * page.dpi)
    gray = remove_long_lines(page.image, page.ink, min_len)
    items = read_page(gray, engine)
    return annotate(items, page.ink, engine.name)
