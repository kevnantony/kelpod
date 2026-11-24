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


def _plan_pages(page, all_plans: bool):
    """The plan crops of a sheet (largest first; just the largest unless ``all_plans``), or the
    page itself when it holds a single plan. Returns ``[(page, label)]``."""
    regions = clean.split_plans(page.ink)
    if len(regions) < 2:
        return [(page, None)]
    chosen = regions if all_plans else regions[:1]
    return [(clean.crop_page(page, r), f"plan{i + 1}") for i, r in enumerate(chosen)]


def _sheet_scale_hint(sheet, seg) -> tuple[float, str] | None:
    """Scale bar of a sheet (it sits outside the plan crops); tick value guessed from the plan's
    wall thickness if its numbers cannot be read."""
    from kelpod.geometry.scalebar import bar_scale, typical_wall_px

    th = typical_wall_px([(math.dist(w.p0, w.p1), w.thickness_px) for w in seg.walls])
    found = bar_scale(sheet.ink, [], th)
    if found is None or (found[1] == "scale_bar_assumed" and str(sheet.source).lower().endswith(".pdf")):
        return None
    return found[0], found[1]


def _convert_page(
    page, sheet, name, output, schema, storey_height_m, ocr_backend, segment_backend, timings
):
    def timed(step: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timings.append(StepTiming(step=step, seconds=time.perf_counter() - t0))
        return out

    seg = timed("segment", segment.run, page, backend=segment_backend)
    if sheet is not page:
        page = page.model_copy(update={"scale_hint": _sheet_scale_hint(sheet, seg)})
    ann = timed("ocr", ocr.run, page, backend=ocr_backend)
    graph = timed("geometry", geometry.run, page, seg, ann)
    if segment_backend == "classical" and _try_pairs(seg, graph):
        # thin-line (two parallel faces, no hatching) plan that did not close: try line pairs
        alt = timed("segment", segment.run, page, backend="pairs")
        alt_graph = timed("geometry", geometry.run, page, alt, ann)
        if _score(alt_graph) > _score(graph):
            seg, graph = alt, alt_graph
    model = timed("model", export.build_model, graph, storey_height_m=storey_height_m, name=name)
    result = None
    if output is not None:
        result = timed("export", export.write_ifc, model, output, schema=schema)
    return PipelineResult(
        source=str(page.source), graph=graph, model=model, export=result, timings=timings
    )


def convert_all(
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
    all_plans: bool = True,
) -> list[PipelineResult]:
    """Like :func:`convert`, but a sheet with several plans gives one result per plan (largest
    first, each its own model/IFC file ``<output stem>_planN.ifc``; plans are never merged).
    ``all_plans=False`` converts only the largest plan of a sheet."""
    t0 = time.perf_counter()
    pages = clean.run(source, dpi=dpi, plan_scale=plan_scale)
    sheet = pages[page_index]
    t_clean = time.perf_counter() - t0
    out = []
    for crop, label in _plan_pages(sheet, all_plans):
        timings = [StepTiming(step="clean", seconds=t_clean)]
        stem = Path(source).stem + (f"_{label}" if label else "")
        dest = None
        if output is not None:
            o = Path(output)
            dest = o if label is None else o.with_name(f"{o.stem}_{label}{o.suffix}")
        out.append(
            _convert_page(
                crop, sheet, stem, dest, schema, storey_height_m, ocr_backend, segment_backend, timings
            )
        )
    return out


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
    """Convert one floor-plan scan to a closed IFC model (one page = one storey for now). On a
    sheet with several plans the largest plan is converted (see :func:`convert_all`)."""
    return convert_all(
        source,
        output,
        schema=schema,
        dpi=dpi,
        plan_scale=plan_scale,
        storey_height_m=storey_height_m,
        page_index=page_index,
        ocr_backend=ocr_backend,
        segment_backend=segment_backend,
        all_plans=False,
    )[0]
