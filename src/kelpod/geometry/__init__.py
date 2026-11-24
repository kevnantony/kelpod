"""Steps 4-5: fit walls to the OCR dimensions, close the graph, build rooms, check consistency.

``run`` never raises for imperfect input: whatever cannot be resolved ends up as a plain-language
issue in ``PlanGraph.report``. Order of work:

1. scale (plan scale, then robustly from dimension spans) and residual skew;
2. snap angles, cluster parallel wall coordinates, merge collinear pieces, bridge opening gaps,
   extend/trim wall ends to junctions (:mod:`.lines`);
3. least-squares fit of the wall lines to the dimension chains (:mod:`.fit`);
4. noding (T-junctions, crossings), spur and loose-piece removal, rooms as holes of the wall
   union, openings placed on walls, room stamps and BRH labels (:mod:`.graph`);
5. graph checks (:mod:`.checks`).
"""

from __future__ import annotations

import logging
import math

from kelpod.contracts import (
    ConsistencyReport,
    OcrAnnotations,
    PlanGraph,
    RasterPage,
    Segmentation,
)

from .checks import check
from .fit import DimM, fit_dimensions
from .graph import (
    GraphBuilder,
    Seg,
    attach_openings,
    find_rooms,
    label_rooms,
    node_segments,
    snap_free,
)
from .lines import OpeningM, WallM, classify, cluster_lines, merge_and_bridge, snap_ends
from .params import GeometryParams
from .scale import (
    Frame,
    _apply,
    default_m_per_px,
    estimate_scale,
    estimate_skew_deg,
    make_correction,
)

__all__ = ["GeometryParams", "check", "run"]

log = logging.getLogger(__name__)


def run(
    page: RasterPage,
    seg: Segmentation,
    ocr: OcrAnnotations,
    *,
    params: GeometryParams | None = None,
) -> PlanGraph:
    """Turn pixel-space segmentation plus OCR into a closed plan-space graph with a report."""
    prm = params or GeometryParams()
    ocr = ocr if ocr is not None else OcrAnnotations()
    try:
        return _run(page, seg, ocr, prm)
    except Exception as exc:  # last resort: the pipeline must always get a graph back
        log.exception("geometry step failed")
        base = page.m_per_px_from_scale
        graph = PlanGraph(
            m_per_px=base or default_m_per_px(page),
            scale_source="plan_scale" if base else "default",
            image_height_px=page.height,
        )
        report = check(graph)
        report.issues.insert(
            0, f"The geometry step failed ({type(exc).__name__}: {exc}); returned an empty plan."
        )
        graph.report = report
        return graph


def _scale_from_bar(page, seg, ocr, m_per_px, scale_source, issues):
    """Graphic scale bar as the scale when no dimensions fixed it (see :mod:`.scalebar`)."""
    from .scalebar import bar_scale

    if page.scale_hint is not None:  # the sheet's bar, found outside this plan's crop
        issues.append(
            f"Scale from the sheet's graphic scale bar ({page.scale_hint[1]}); "
            "check the scale."
        )
        return page.scale_hint
    ths = sorted(
        (w.thickness_px for w in (seg.walls if seg is not None else []) if _finite(w.thickness_px)),
    )
    th = ths[int(0.75 * len(ths))] if ths else None  # exterior walls: upper quartile
    try:
        found = bar_scale(page.ink, ocr.items, th)
    except Exception:  # never let the optional bar break the step
        log.exception("scale bar detection failed")
        return m_per_px, scale_source
    if found is None:
        return m_per_px, scale_source
    m, src, bar = found
    issues.append(
        f"Scale from the graphic scale bar ({len(bar.ticks_x)} ticks of {bar.tick_px:.1f} px): "
        + (
            "numbers read."
            if src == "scale_bar"
            else "its numbers were not legible, the tick value was guessed from the wall "
            "thickness. Check the scale."
        )
    )
    return m, src


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _run(
    page: RasterPage, seg: Segmentation, ocr: OcrAnnotations, prm: GeometryParams
) -> PlanGraph:
    issues: list[str] = []

    # 1. scale and residual skew ---------------------------------------------------------------
    m_per_px, scale_source, scale_issues = estimate_scale(page, ocr.dimensions, prm)
    issues += scale_issues
    if scale_source != "dimensions":
        m_per_px, scale_source = _scale_from_bar(page, seg, ocr, m_per_px, scale_source, issues)
    raw = [
        w
        for w in (seg.walls if seg is not None else [])
        if _finite(*w.p0, *w.p1, w.thickness_px) and w.thickness_px > 0
    ]
    skew = estimate_skew_deg([(w.p0, w.p1) for w in raw], prm.max_skew_deg)
    frame = Frame(m_per_px, page.height, page.width / 2, page.height / 2)
    if abs(skew) >= prm.min_skew_deg:
        frame.rot_rad = math.radians(-skew)
        if abs(skew) > 0.5:
            issues.append(f"The walls were rotated by {skew:.1f}° and straightened.")

    walls_m = []
    n_thin = 0
    for w in raw:
        t = w.thickness_px * m_per_px
        if t < prm.min_thickness_m:
            n_thin += 1
            continue
        walls_m.append(WallM(frame.to_m_raw(w.p0), frame.to_m_raw(w.p1), t, w.kind))
    if n_thin:
        issues.append(
            f"Ignored {n_thin} wall piece(s) thinner than {prm.min_thickness_m * 100:.0f} cm."
        )
    openings_m = [
        OpeningM(o.kind, frame.to_m_raw(o.p0), frame.to_m_raw(o.p1))
        for o in (seg.openings if seg is not None else [])
        if _finite(*o.p0, *o.p1)
    ]

    # 2. axis lines ----------------------------------------------------------------------------
    pieces, free = classify(walls_m, prm)
    lines = cluster_lines(pieces, prm)
    pieces, synthetic = merge_and_bridge(pieces, lines, openings_m, prm, issues)
    pieces = snap_ends(pieces, prm)

    # 3. dimension fit -------------------------------------------------------------------------
    dims = _dims_m(ocr, frame)
    residual = None
    if dims and pieces:
        try:
            fit = fit_dimensions(lines, pieces, dims, prm)
        except Exception as exc:  # keep going with the drawn geometry
            log.exception("dimension fit failed")
            issues.append(f"The dimension fit failed ({exc}); walls are as drawn.")
            fit = None
        if fit is not None and fit.mode is not None:
            residual = fit.residual_m
            fx = [
                (ln.coord, c) for ln, c in zip(lines, fit.coords, strict=True) if ln.orient == "v"
            ]
            fy = [
                (ln.coord, c) for ln, c in zip(lines, fit.coords, strict=True) if ln.orient == "h"
            ]
            frame.fx, frame.fy = make_correction(fx), make_correction(fy)
            for ln, c in zip(lines, fit.coords, strict=True):
                ln.coord = c
            for p in pieces:
                p.c = lines[p.line].coord
            issues.append(
                f"Fitted the walls to {fit.n_used} of {fit.n_dims} dimension labels "
                f"(ticks on wall {'faces' if fit.mode == 'faces' else 'centrelines'}); "
                f"RMS error {fit.residual_before_m * 100:.1f} cm before, "
                f"{fit.residual_m * 100:.1f} cm after."
            )
            if fit.n_outliers:
                issues.append(
                    f"{fit.n_outliers} dimension label(s) contradict the others and were ignored "
                    "(possible OCR misreading)."
                )
        elif fit is not None:
            issues.append(
                f"None of the {len(dims)} dimension labels could be matched to walls; "
                "walls are as drawn."
            )
    elif not ocr.dimensions:
        issues.append("No dimension labels were read; wall lengths are as drawn (not verified).")

    # 4. graph ---------------------------------------------------------------------------------
    segs: list[Seg] = []
    for p in pieces:
        orient_x = frame.fx if p.orient == "h" else frame.fy
        a = lines[p.ref_a].coord if p.ref_a is not None else _apply(orient_x, p.a)
        b = lines[p.ref_b].coord if p.ref_b is not None else _apply(orient_x, p.b)
        if b - a <= 1e-6:
            continue
        if p.orient == "h":
            segs.append(Seg((a, p.c), (b, p.c), p.t, p.kind))
        else:
            segs.append(Seg((p.c, a), (p.c, b), p.t, p.kind))
    free_segs = [Seg(frame.correct(w.p0), frame.correct(w.p1), w.t, w.kind) for w in free]
    segs += snap_free(free_segs, segs, prm.free_snap_m)
    segs = node_segments(segs, eps=prm.node_tol_m / 2)

    gb = GraphBuilder(segs, prm)
    n_spurs = gb.remove_spurs()
    n_loose, len_loose = gb.prune_components()
    gb.merge_collinear()
    nodes, walls = gb.finish()
    if n_spurs:
        issues.append(f"Removed {n_spurs} short wall stub(s) sticking out of junctions.")
    if n_loose:
        issues.append(
            f"Dropped {n_loose} loose wall piece(s) ({len_loose:.1f} m in total) that are not "
            "connected to the building."
        )

    rooms, n_small = find_rooms(nodes, walls, prm)
    if n_small:
        issues.append(
            f"Ignored {n_small} enclosed area(s) smaller than {prm.min_room_area_m2:g} m² "
            "(shafts or drawing artefacts)."
        )
    stamps = [(st, frame.to_m(st.anchor)) for st in ocr.room_stamps if _finite(*st.anchor)]
    label_rooms(rooms, stamps, issues)

    all_openings = [
        OpeningM(o.kind, frame.correct(o.p0), frame.correct(o.p1), o.synthetic)
        for o in openings_m + synthetic
    ]
    sills = [(s.value_m, frame.to_m(s.anchor)) for s in ocr.sill_heights if _finite(*s.anchor)]
    openings = attach_openings(all_openings, nodes, walls, sills, prm, issues)

    graph = PlanGraph(
        nodes=nodes,
        walls=walls,
        openings=openings,
        rooms=rooms,
        m_per_px=m_per_px,
        scale_source=scale_source,
        image_height_px=page.height,
    )

    # 5. checks --------------------------------------------------------------------------------
    report = check(graph)
    report.dimension_residual_m = residual
    report.issues = issues + report.issues
    _room_closure_vs_segmentation(report, graph, seg, frame, prm)
    graph.report = report
    return graph


def _dims_m(ocr: OcrAnnotations, frame: Frame) -> list[DimM]:
    out = []
    tan10 = math.tan(math.radians(10))
    for d in ocr.dimensions:
        if d.line is None or not _finite(d.value_m, *d.line[0], *d.line[1]) or d.value_m <= 0:
            continue
        (x0, y0), (x1, y1) = frame.to_m_raw(d.line[0]), frame.to_m_raw(d.line[1])
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dy <= tan10 * dx:
            out.append(DimM("x", min(x0, x1), max(x0, x1), d.value_m, d.text))
        elif dx <= tan10 * dy:
            out.append(DimM("y", min(y0, y1), max(y0, y1), d.value_m, d.text))
    return out


def _room_closure_vs_segmentation(
    report: ConsistencyReport, graph: PlanGraph, seg: Segmentation | None, frame: Frame, prm
) -> None:
    """Room closure against the rooms the segmentation saw: how many became closed polygons."""
    if seg is None or not seg.rooms:
        return
    from shapely.geometry import Point as SPoint
    from shapely.geometry import Polygon

    min_px = prm.min_room_area_m2 / (graph.m_per_px**2)
    regions = [r for r in seg.rooms if r.area_px >= min_px and _finite(*r.seed)]
    if not regions:
        return
    polys = [Polygon(r.polygon) for r in graph.rooms]
    hits = 0
    for r in regions:
        p = SPoint(frame.to_m(r.seed))
        if any(poly.contains(p) for poly in polys):
            hits += 1
    report.room_closure = hits / len(regions)
    if hits < len(regions):
        report.issues.append(
            f"{len(regions) - hits} of {len(regions)} room regions seen by the segmentation "
            "did not become closed rooms."
        )
