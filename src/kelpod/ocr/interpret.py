"""Turn recognised text items into dimensions, sill heights and room stamps."""

from __future__ import annotations

import math

import numpy as np

from kelpod.contracts import DimensionLabel, OcrAnnotations, RoomStamp, SillHeight, TextItem
from kelpod.ocr.dimlines import DimensionLineFinder
from kelpod.ocr.engines import quad_size
from kelpod.ocr.parse import (
    is_brh_keyword,
    is_opening_label,
    match_room_name,
    parse_area,
    parse_length_m,
    parse_number,
    parse_sill_height,
    parse_sill_value,
)


def _snap_angle(a: float) -> float:
    """Snap to 0 or 90 when within 15°, otherwise keep."""
    a = ((a + 180) % 360) - 180
    for t in (0.0, 90.0):
        if abs(a - t) <= 15:
            return t
    return a


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _text_height(it: TextItem) -> float:
    return max(1.0, quad_size(it.quad)[1])


def interpret(
    items: list[TextItem], ink: np.ndarray | None = None
) -> tuple[list[DimensionLabel], list[SillHeight], list[RoomStamp], list[str]]:
    """Classify items. ``ink`` (bool mask) enables the dimension-line / tick search."""
    notes: list[str] = []
    used: set[int] = set()
    sills: list[SillHeight] = []
    stamps: list[RoomStamp] = []
    dims: list[DimensionLabel] = []

    # 1. Sill heights, in one box ("BRH 0,90") or split ("BRH" + "0,90" to its right / below).
    for i, it in enumerate(items):
        v = parse_sill_height(it.text)
        if v is not None:
            sills.append(SillHeight(value_m=v, text=it.text, anchor=it.center,
                                    confidence=it.confidence))
            used.add(i)
        elif is_brh_keyword(it.text):
            h = _text_height(it)
            best, best_d = None, 4 * h
            for j, other in enumerate(items):
                if j == i or j in used or parse_number(other.text) is None:
                    continue
                d = _dist(it.center, other.center)
                if d < best_d:
                    best, best_d = j, d
            used.add(i)
            if best is not None:
                v = parse_sill_value(items[best].text)
                if v is not None:
                    txt = f"{it.text} {items[best].text}"
                    sills.append(SillHeight(value_m=v, text=txt, anchor=it.center,
                                            confidence=min(it.confidence,
                                                           items[best].confidence)))
                    used.add(best)

    # 2. Room names and areas.
    names: list[tuple[int, str]] = []
    areas: list[tuple[int, float]] = []
    for i, it in enumerate(items):
        if i in used:
            continue
        ar = parse_area(it.text)
        if ar is not None:
            area, before = ar
            name = match_room_name(before) if before else None
            if name is not None:
                stamps.append(RoomStamp(name=name, area_m2=area, text=it.text, anchor=it.center,
                                        confidence=it.confidence))
                used.add(i)
            else:
                areas.append((i, area))
            continue
        name = match_room_name(it.text)
        if name is not None:
            names.append((i, name))

    # Pair each name with the closest area below it (stamp: name line, then area line).
    pairs = []
    for ni, name in names:
        nit = items[ni]
        h = _text_height(nit)
        for ai, area in areas:
            ait = items[ai]
            dx = ait.center[0] - nit.center[0]
            dy = ait.center[1] - nit.center[1]
            w = max(quad_size(nit.quad)[0], quad_size(ait.quad)[0])
            if 0.3 * h <= dy <= 3.5 * h and abs(dx) <= 0.75 * w + h:
                pairs.append((math.hypot(dx, dy), ni, name, ai, area))
    paired_n: set[int] = set()
    paired_a: set[int] = set()
    for _, ni, name, ai, area in sorted(pairs):
        if ni in paired_n or ai in paired_a:
            continue
        paired_n.add(ni)
        paired_a.add(ai)
        nit, ait = items[ni], items[ai]
        stamps.append(RoomStamp(
            name=name, area_m2=area, text=f"{nit.text} {ait.text}",
            anchor=((nit.center[0] + ait.center[0]) / 2, (nit.center[1] + ait.center[1]) / 2),
            confidence=min(nit.confidence, ait.confidence),
        ))
    for ni, name in names:
        if ni not in paired_n:
            nit = items[ni]
            stamps.append(RoomStamp(name=name, text=nit.text, anchor=nit.center,
                                    confidence=nit.confidence))
    used |= {ni for ni, _ in names} | {ai for ai, _ in areas}
    lone = [a for ai, a in areas if ai not in paired_a]
    if lone:
        notes.append(f"{len(lone)} area(s) without a room name: {lone}")

    # 3. Everything else that parses as a length is a dimension.
    finder = DimensionLineFinder(ink) if ink is not None else None
    n_line = 0
    skipped: list[str] = []
    for i, it in enumerate(items):
        if i in used:
            continue
        if is_opening_label(it.text):
            skipped.append(it.text)
            continue
        v = parse_length_m(it.text)
        if v is None:
            skipped.append(it.text)
            continue
        angle = _snap_angle(it.angle_deg)
        n = parse_number(it.text)
        conf = it.confidence * (0.8 if n is not None and n.fixed else 1.0)
        line = None
        if finder is not None:
            found = finder.find(it.quad, angle)
            if found is not None:
                line, _ = found
                n_line += 1
        dims.append(DimensionLabel(value_m=v, text=it.text, anchor=it.center, angle_deg=angle,
                                   line=line, confidence=conf))
    if dims:
        notes.append(f"{n_line}/{len(dims)} dimensions linked to a dimension line with ticks")
    if skipped:
        notes.append(f"{len(skipped)} text item(s) not interpreted: {skipped[:20]}")
    return dims, sills, stamps, notes


def annotate(items: list[TextItem], ink: np.ndarray | None, backend: str) -> OcrAnnotations:
    dims, sills, stamps, notes = interpret(items, ink)
    return OcrAnnotations(items=items, dimensions=dims, sill_heights=sills, room_stamps=stamps,
                          backend=backend, notes=notes)
