"""Placeholder for a learned segmentation backend.

Idea (not built): a junction / centreline heatmap model (detection readout, cf. fpvec-lab,
FloorplanTransformation) or RoomFormer-style room polygons, plus a detector (Detectron2, RF-DETR,
D-FINE; all Apache-2.0) for doors and windows. No surveyed pretrained weights are commercially
usable, so it needs training on synthetic DIN-style renders plus annotated real German plans.
"""

from __future__ import annotations

from pathlib import Path

from kelpod.contracts import RasterPage, Segmentation


class LearnedSegmenter:
    name = "learned"

    def __init__(self, weights: str | Path | None = None) -> None:
        self.weights = Path(weights) if weights is not None else None

    def segment(self, page: RasterPage) -> Segmentation:
        raise NotImplementedError(
            "the learned segmentation backend is an idea, not built yet: it needs a model "
            "trained on German plans (weights path: "
            f"{self.weights}). Use backend='classical'."
        )
