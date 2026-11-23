"""Steps 4-5: fit walls to the OCR dimensions, close the graph, build rooms, check consistency."""

from __future__ import annotations

from kelpod.contracts import ConsistencyReport, OcrAnnotations, PlanGraph, RasterPage, Segmentation


def run(page: RasterPage, seg: Segmentation, ocr: OcrAnnotations) -> PlanGraph:
    """Turn pixel-space segmentation plus OCR into a closed plan-space graph with a report."""
    raise NotImplementedError


def check(graph: PlanGraph) -> ConsistencyReport:
    """Graph consistency checks (NetworkX): components, dangling nodes, open rooms."""
    raise NotImplementedError
