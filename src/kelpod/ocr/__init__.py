"""Step 3: read dimension chains, sill heights and room stamps."""

from __future__ import annotations

from kelpod.contracts import OcrAnnotations, RasterPage


def run(page: RasterPage, *, backend: str = "rapidocr") -> OcrAnnotations:
    """Detect and recognise text (including rotated text) and parse it into dimensions,
    sill heights and room stamps."""
    raise NotImplementedError
