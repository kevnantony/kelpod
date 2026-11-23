"""Step 2: segment walls, doors, windows and rooms."""

from __future__ import annotations

from kelpod.contracts import RasterPage, Segmentation


def run(page: RasterPage, *, backend: str = "classical") -> Segmentation:
    """Segment one cleaned page. ``backend="classical"`` is the OpenCV baseline that runs on a Mac;
    learned backends (RoomFormer, Detectron2) plug in through the ``Segmenter`` protocol later."""
    raise NotImplementedError
