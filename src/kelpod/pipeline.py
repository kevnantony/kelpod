"""Runs the six steps for one input file and records the time each step takes."""

from __future__ import annotations

import time
from pathlib import Path

from kelpod import clean, export, geometry, ocr, segment
from kelpod.contracts import IfcSchema, PipelineResult, StepTiming


def convert(
    source: str | Path,
    output: str | Path | None = None,
    *,
    schema: IfcSchema = "IFC4",
    dpi: float = 200.0,
    plan_scale: float | None = 100.0,
    storey_height_m: float = 2.60,
    page_index: int = 0,
    ocr_backend: str = "rapidocr",
    segment_backend: str = "classical",
) -> PipelineResult:
    """Convert one floor-plan scan to a closed IFC model (one page = one storey for now)."""
    timings: list[StepTiming] = []

    def timed(step: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timings.append(StepTiming(step=step, seconds=time.perf_counter() - t0))
        return out

    pages = timed("clean", clean.run, source, dpi=dpi, plan_scale=plan_scale)
    page = pages[page_index]
    seg = timed("segment", segment.run, page, backend=segment_backend)
    ann = timed("ocr", ocr.run, page, backend=ocr_backend)
    graph = timed("geometry", geometry.run, page, seg, ann)
    model = timed(
        "model", export.build_model, graph, storey_height_m=storey_height_m, name=Path(source).stem
    )
    result = None
    if output is not None:
        result = timed("export", export.write_ifc, model, output, schema=schema)
    return PipelineResult(
        source=str(source), graph=graph, model=model, export=result, timings=timings
    )
