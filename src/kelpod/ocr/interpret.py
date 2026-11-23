"""Turn recognised text items into dimensions, sill heights and room stamps."""

from __future__ import annotations

import math

import numpy as np

from kelpod.contracts import DimensionLabel, OcrAnnotations, RoomStamp, SillHeight, TextItem
from kelpod.ocr.dimlines import DimensionLineFinder, pick_span
from kelpod.ocr.engines import quad_size
from kelpod.ocr.parse import (
    is_brh_keyword,
    is_opening_label,
    match_room_name,
    number_tokens,
    parse_area,
    parse_length_m,
    parse_number,
    parse_sill_height,
    parse_sill_value,
)


def split_merged_items(items: list[TextItem]) -> list[TextItem]:
    """Clean up number items that do not parse as a whole: ``0,6351,15`` (two labels read as
    one) or ``L 88,5 L`` (tick remnants read as letters). Each plausible number token becomes its
    own item, its quad cut from the original in proportion to its character span. Confidence is
    lowered because the split is a guess."""
    out: list[TextItem] = []
    for it in items:
        toks = None
        t = it.text
        if (parse_length_m(t) is None and parse_area(t) is None and parse_sill_height(t) is None
                and not is_brh_keyword(t) and not is_opening_label(t)
                and match_room_name(t) is None):
            toks = number_tokens(t)
        if not toks:
            out.append(it)
            continue
        q = np.asarray(it.quad, float)
        n = max(1, len(t))
        for start, end, tok in toks:
            t0, t1 = start / n, end / n
            tl = q[0] + (q[1] - q[0]) * t0
            tr = q[0] + (q[1] - q[0]) * t1
            bl = q[3] + (q[2] - q[3]) * t0
            br = q[3] + (q[2] - q[3]) * t1
            quad = [tuple(map(float, c)) for c in (tl, tr, br, bl)]
            out.append(it.model_copy(update={"text": tok, "quad": quad,
                                             "confidence": it.confidence * 0.7}))
    return out


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
    items = split_merged_items(items)
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
    cands: list[tuple[TextItem, float, float, float]] = []  # item, value, angle, confidence
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
        n = parse_number(it.text)
        conf = it.confidence * (0.8 if n is not None and n.fixed else 1.0)
        cands.append((it, v, _snap_angle(it.angle_deg), conf))
    lines = _link_lines(cands, ink, notes) if ink is not None else [None] * len(cands)
    for (it, v, angle, conf), line in zip(cands, lines, strict=True):
        dims.append(DimensionLabel(value_m=v, text=it.text, anchor=it.center, angle_deg=angle,
                                   line=line, confidence=conf))
    if skipped:
        notes.append(f"{len(skipped)} text item(s) not interpreted: {skipped[:20]}")
    return dims, sills, stamps, notes


def _link_lines(cands, ink: np.ndarray, notes: list[str]):
    """Find the measured span of every dimension text.

    Pass 1 takes the nearest tick on each side of the text. The page-wide pixels per metre is
    the median of span length / value over those spans (no plan scale assumed). Pass 2 checks
    every span against it and, where it disagrees or is missing, picks the tick pair whose
    length fits (short spans whose label overhangs the ticks, missed or extra ticks); spans that
    still do not fit within 8 % are dropped rather than passed on wrong.
    """
    finder = DimensionLineFinder(ink)
    found = [finder.ticks(it.quad, angle) for it, _, angle, _ in cands]
    first = []
    ratios = []
    for (_, v, _, _), f in zip(cands, found, strict=True):
        sp = pick_span(f[0]) if f is not None else None
        first.append(sp)
        if sp is not None:
            ratios.append((sp[1] - sp[0]) / v)
    if len(ratios) < 3:
        out = []
        for f, sp in zip(found, first, strict=True):
            out.append(None if sp is None else finder.to_image(sp[0], sp[1], f[0].y, f[1]))
        if ratios:
            notes.append("too few dimension lines to check spans against each other")
        return out
    r = float(np.median(ratios))
    out = []
    n_ok = n_refit = n_drop = 0
    for (_, v, _, _), f, sp in zip(cands, found, first, strict=True):
        if f is None:
            out.append(None)
            continue
        exp = v * r
        if sp is not None and abs((sp[1] - sp[0]) - exp) <= 0.08 * exp:
            n_ok += 1
        else:
            sp2 = pick_span(f[0], exp)
            if sp is not None or sp2 is not None:
                n_refit += sp2 is not None
                n_drop += sp2 is None
            sp = sp2
        out.append(None if sp is None else finder.to_image(sp[0], sp[1], f[0].y, f[1]))
    notes.append(
        f"{sum(o is not None for o in out)}/{len(cands)} dimensions linked to a dimension line "
        f"with ticks ({n_refit} re-picked, {n_drop} dropped as inconsistent; "
        f"{r:.1f} px/m from the spans)"
    )
    return out


def annotate(items: list[TextItem], ink: np.ndarray | None, backend: str) -> OcrAnnotations:
    items = split_merged_items(items)
    dims, sills, stamps, notes = interpret(items, ink)
    return OcrAnnotations(items=items, dimensions=dims, sill_heights=sills, room_stamps=stamps,
                          backend=backend, notes=notes)
