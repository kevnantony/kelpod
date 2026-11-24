"""Runs the six steps for one input file and records the time each step takes."""

from __future__ import annotations

import math
import time
from pathlib import Path

from kelpod import clean, export, geometry, ocr, segment
from kelpod.contracts import IfcSchema, PipelineResult, StepTiming


def _score(graph) -> tuple[int, int, int]:
    """Ordering of candidate graphs: some rooms, then few dangling ends / loose groups."""
    r = graph.report
    if r is None:
        return (0, 0, 0)
    return (
        int(r.n_rooms > 0),
        -(len(r.dangling_nodes) + 2 * max(0, r.n_components - 1)),
        r.n_rooms,
    )


def _try_pairs(seg, graph) -> bool:
    """Not closed and not drawn with hatched / solid walls (those are the classical path's)."""
    r = graph.report
    if r is not None and r.is_closed and r.n_rooms > 0:
        return False
    total = sum(math.dist(w.p0, w.p1) for w in seg.walls) or 1.0
    hatched = sum(math.dist(w.p0, w.p1) for w in seg.walls if w.confidence > 0.8)
    return hatched / total < 0.5


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
    if segment_backend == "classical" and _try_pairs(seg, graph):
        # thin-line (two parallel faces, no hatching) plan that did not close: try line pairs
        alt = timed("segment", segment.run, page, backend="pairs")
        alt_graph = timed("geometry", geometry.run, page, alt, ann)
        if _score(alt_graph) > _score(graph):
            seg, graph = alt, alt_graph
    model = timed(
        "model", export.build_model, graph, storey_height_m=storey_height_m, name=Path(source).stem
    )
    result = None
    if output is not None:
        result = timed("export", export.write_ifc, model, output, schema=schema)
    return PipelineResult(
        source=str(source), graph=graph, model=model, export=result, timings=timings
    )
