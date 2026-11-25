"""Step 2: segment walls, doors, windows and rooms."""

from __future__ import annotations

from kelpod.contracts import RasterPage, Segmentation
from kelpod.segment.base import Segmenter, available, get_segmenter, register
from kelpod.segment.classical import ClassicalParams, ClassicalSegmenter
from kelpod.segment.learned import LearnedSegmenter
from kelpod.segment.pairs import PairsSegmenter  # noqa: E402

register("classical", ClassicalSegmenter)
register("pairs", PairsSegmenter)
register("learned", LearnedSegmenter)

__all__ = [
    "ClassicalParams",
    "ClassicalSegmenter",
    "LearnedSegmenter",
    "Segmenter",
    "available",
    "get_segmenter",
    "register",
    "run",
]


def run(page: RasterPage, *, backend: str = "classical") -> Segmentation:
    """Segment one cleaned page. ``backend="classical"`` is the OpenCV baseline that runs on a Mac;
    learned backends (RoomFormer, Detectron2) plug in through the ``Segmenter`` protocol later."""
    return get_segmenter(backend).segment(page)
