"""Classical (OpenCV) segmentation baseline for deskewed, rectilinear German plans.

Follows the drawing conventions in ``docs/ARCHITECTURE.md`` (DIN 1356-1 style): walls are closed
outlines, masonry is hatched (or solid), partitions are outline only, doors are wall gaps with a
leaf and a quarter arc, windows are wall gaps with two or three thin lines along the wall.
All size thresholds are given in metres and converted with the page scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kelpod.contracts import (
    OpeningKind,
    OpeningPx,
    RasterPage,
    Segmentation,
    WallKind,
    WallSegment,
)
from kelpod.segment import openings as op
from kelpod.segment import rooms as rm
from kelpod.segment import walls as wl


@dataclass
class ClassicalParams:
    min_wall_m: float = 0.09
    """Thinnest wall to keep (light partitions are 10 / 11.5 cm)."""
    max_wall_m: float = 0.60
    max_fill_radius_m: float = 0.20
    """Enclosed background with a smaller inscribed radius is treated as wall interior."""
    min_wall_len_m: float = 0.12
    min_component_m: float = 0.40
    """Wall-mask components whose bounding box is smaller than this are dropped (text blobs)."""
    min_opening_m: float = 0.45
    max_opening_m: float = 3.5
    min_exterior_m: float = 0.20
    """Hatched walls at least this thick may hold two-line windows."""
    exterior_min_th_m: float = 0.28
    drywall_max_th_m: float = 0.15
    min_room_m2: float = 0.6
    door_score: float = 0.6


class ClassicalSegmenter:
    name = "classical"

    def __init__(self, params: ClassicalParams | None = None) -> None:
        self.params = params or ClassicalParams()

    def segment(self, page: RasterPage) -> Segmentation:
        return segment_classical(page, self.params)


def segment_classical(page: RasterPage, p: ClassicalParams | None = None) -> Segmentation:
    p = p or ClassicalParams()
    ink = np.asarray(page.ink, dtype=bool)
    notes: list[str] = []
    m_per_px = page.m_per_px_from_scale
    if m_per_px is None:
        m_per_px = 0.0254 / page.dpi * 100.0
        notes.append("plan_scale unknown: assumed 1:100 for size thresholds")

    def px(m: float) -> float:
        return m / m_per_px

    # 1. wall bodies
    filled, body = wl.wall_body_mask(
        ink,
        max_radius_px=px(p.max_fill_radius_m),
        open_px=int(max(2, round(0.55 * px(p.min_wall_m)))),
        min_extent_px=px(p.min_component_m),
    )
    max_th = min(wl.estimate_max_thickness(body), px(p.max_wall_m))
    klen = int(np.ceil(1.25 * max(max_th, px(p.min_wall_m)))) + 2

    # 2. runs per orientation, windows, merged walls, junctions
    runs: list[wl.Run] = []
    for orient in (wl.H, wl.V):
        runs += wl.extract_runs(
            body,
            ink,
            orient,
            kernel_len=klen,
            min_th_px=0.8 * px(p.min_wall_m),
            min_len_px=px(p.min_wall_len_m),
        )
    wl.classify_windows(
        runs,
        min_w_px=px(p.min_opening_m),
        max_w_px=px(p.max_opening_m),
        min_ext_th_px=px(p.min_exterior_m),
    )
    window_runs = [r for r in runs if r.is_window]
    walls = wl.merge_walls(runs, max_gap_px=2.0)
    walls = [w for w in walls if w.length >= px(p.min_wall_len_m)]
    wl.snap_junctions(walls, tol_px=max(2.0, px(0.03)))
    # outlines are drawn centred on the wall faces: remove the pen width from thickness and
    # from free wall ends (jambs) so walls describe the faces, not the ink extent
    lws = [r.meta.get("face_lw", 0.0) for r in runs if not r.is_window]
    lws = [v for v in lws if v > 0]
    face_lw = float(np.median(lws)) if lws else 0.0
    face_lw = min(face_lw, 0.25 * min((w.th for w in walls), default=0.0))
    if face_lw > 0:
        for w in walls + window_runs:
            w.th = max(w.th - face_lw, 0.5 * w.th)
            if not w.meta.get("snap0"):
                w.a0 += face_lw / 2
            if not w.meta.get("snap1"):
                w.a1 -= face_lw / 2
    for r in window_runs:  # window edges = the neighbouring jambs
        _snap_window_ends(r, walls, tol=face_lw + 3)

    # 3. openings in gaps
    ink_d = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gaps = op.find_gaps(
        walls, min_w_px=px(p.min_opening_m), max_w_px=px(p.max_opening_m), blockers=window_runs
    )
    gaps = op.classify_gaps(gaps, ink, ink_d, door_thr=p.door_score)

    # 4. wall mask = wall bodies under the vectorised walls; barrier adds openings
    rect = np.zeros_like(body)
    for w in walls:
        rm.draw_rect(rect, w.orient, w.c, w.a0 - 2, w.a1 + 2, w.th + face_lw + 2)
    wall_mask = filled & rect & body
    barrier = np.zeros_like(body)
    for w in walls:
        rm.draw_rect(barrier, w.orient, w.c, w.a0, w.a1, w.th)
    for r in window_runs:
        rm.draw_rect(barrier, r.orient, r.c, r.a0 - 1, r.a1 + 1, r.th)
    for g in gaps:
        rm.draw_rect(barrier, g.orient, g.c, g.a0 - 1, g.a1 + 1, g.th)
    _, lab, outside = rm.free_space(barrier)
    rooms = rm.extract_rooms(lab, outside, min_area_px=px(1.0) ** 2 * p.min_room_m2)

    # 5. wall kinds
    segs: list[WallSegment] = []
    for i, w in enumerate(walls):
        ext = _touches_outside(w, lab, outside)
        th_m = w.th * m_per_px
        hatched = w.hatched_frac >= 0.5
        if ext and (hatched or th_m >= p.min_exterior_m):
            kind = WallKind.EXTERIOR
        elif hatched and th_m >= p.exterior_min_th_m and ext is None:
            kind = WallKind.EXTERIOR
        elif hatched or th_m > p.drywall_max_th_m:
            kind = WallKind.INTERIOR
        else:
            kind = WallKind.DRYWALL
        w.meta["id"] = i
        segs.append(
            WallSegment(
                id=i,
                p0=w.p0(),
                p1=w.p1(),
                thickness_px=float(w.th),
                kind=kind,
                confidence=0.9 if hatched else 0.7,
            )
        )

    # 6. openings -> contract
    out_openings: list[OpeningPx] = []
    for r in window_runs:
        wid = _nearest_wall(r, walls)
        c = walls[wid].c if wid is not None else r.c
        out_openings.append(
            OpeningPx(
                id=len(out_openings),
                kind=OpeningKind.WINDOW,
                p0=(r.a0, c) if r.orient == wl.H else (c, r.a0),
                p1=(r.a1, c) if r.orient == wl.H else (c, r.a1),
                wall_id=wid,
                confidence=0.85 if r.label == "L" else 0.7,
            )
        )
    for g in gaps:
        out_openings.append(
            OpeningPx(
                id=len(out_openings),
                kind=g.kind,
                p0=g.to_xy(g.a0, g.c),
                p1=g.to_xy(g.a1, g.c),
                wall_id=g.left.meta.get("id"),
                hinge=g.hinge,
                swing_into=g.swing_into,
                confidence=g.confidence,
            )
        )

    notes.append(
        f"classical: {len(segs)} walls, {len(out_openings)} openings, {len(rooms)} rooms; "
        f"max wall thickness ~{max_th * m_per_px:.3f} m; outline pen ~{face_lw:.1f} px"
    )
    return Segmentation(
        wall_mask=wall_mask,
        walls=segs,
        openings=out_openings,
        rooms=rooms,
        backend="classical",
        notes=notes,
    )


def _touches_outside(w: wl.Run, lab: np.ndarray, outside: np.ndarray) -> bool | None:
    """True if one side of the wall is the outside free space, False if neither is, None if
    the sides could not be sampled."""
    hgt, wid = lab.shape
    n = max(3, int(w.length / 10))
    a = np.linspace(w.a0 + 0.2 * w.length, w.a1 - 0.2 * w.length, n)
    votes = 0
    total = 0
    for side in (-1, 1):
        off = w.c + side * (w.th / 2 + 3)
        xs, ys = (a, np.full_like(a, off)) if w.orient == wl.H else (np.full_like(a, off), a)
        xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
        ok = (xi >= 0) & (xi < wid) & (yi >= 0) & (yi < hgt)
        if not ok.any():
            continue
        labs = lab[yi[ok], xi[ok]]
        total += 1
        if outside[labs].mean() > 0.5:
            votes += 1
    if total == 0:
        return None
    return votes > 0


def _nearest_wall(r: wl.Run, walls: list[wl.Run]) -> int | None:
    best = None
    for i, w in enumerate(walls):
        if w.orient != r.orient or abs(w.c - r.c) > 0.5 * max(w.th, r.th) + 2:
            continue
        d = min(abs(w.a1 - r.a0), abs(r.a1 - w.a0))
        if best is None or d < best[0]:
            best = (d, i)
    return None if best is None else best[1]


def _snap_window_ends(r: wl.Run, walls: list[wl.Run], tol: float) -> None:
    for w in walls:
        if w.orient != r.orient or abs(w.c - r.c) > 0.5 * max(w.th, r.th) + 2:
            continue
        if abs(w.a1 - r.a0) <= tol:
            r.a0 = w.a1
        if abs(w.a0 - r.a1) <= tol:
            r.a1 = w.a0
