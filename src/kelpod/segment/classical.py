"""Classical (OpenCV) segmentation baseline for deskewed, rectilinear German plans.

Follows the drawing conventions in ``docs/ARCHITECTURE.md`` (DIN 1356-1 style): walls are closed
outlines, masonry is hatched (or solid), partitions are outline only, doors are wall gaps with a
leaf and a quarter arc, windows are wall gaps with two or three thin lines along the wall.
All size thresholds are given in metres and converted with the page scale.
"""

from __future__ import annotations

import copy
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
    min_face_th_m: float = 0.09
    """Walls whose faces (pen width removed) are closer than this are single lines (fold marks)."""
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
    klen = int(np.ceil(1.1 * max(max_th, px(p.min_wall_m)))) + 3

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
    ink_d = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # windows sit in the building outline: a "window" with building interior on both sides is
    # a wall whose band holds a stray line (fold mark, text); rebuild once with it as a wall
    base_runs, demoted = runs, set()
    for attempt in range(2):
        runs = copy.deepcopy(base_runs)
        for r in runs:
            if _run_key(r) in demoted:
                r.is_window = False
        window_runs = [r for r in runs if r.is_window]
        walls = wl.merge_walls(runs, max_gap_px=2.0)
        walls = [w for w in walls if w.length >= px(p.min_wall_len_m)]
        # outlines are drawn centred on the wall faces: the pen width is removed below from
        # thickness and from free wall ends (jambs) so walls describe the faces, not the ink
        lws = [r.meta.get("face_lw", 0.0) for r in runs if r.label == "H" and not r.is_window]
        lws = [v for v in lws if v > 0] or [
            r.meta.get("face_lw", 0.0) for r in runs if r.meta.get("face_lw", 0.0) > 0
        ]
        face_lw = float(np.median(lws)) if lws else 0.0
        face_lw = min(face_lw, 0.25 * px(p.min_wall_m) * 1.5)
        # a single (blurred) line - fold mark, dimension line - is not a wall: walls have two faces
        # at least ~10 cm apart
        walls = [w for w in walls if _face_th(w.th, face_lw) >= px(p.min_face_th_m)]
        wl.snap_junctions(walls, tol_px=max(2.0, px(0.03)))
        walls = wl.drop_face_bumps(walls, px(p.min_wall_len_m))
        if face_lw > 0:
            for w in walls + window_runs:
                w.th = _face_th(w.th, face_lw)
                if not w.meta.get("snap0"):
                    w.a0 += face_lw / 2
                if not w.meta.get("snap1"):
                    w.a1 -= face_lw / 2
        for r in window_runs:  # window edges = the neighbouring jambs
            _snap_window_ends(r, walls, tol=face_lw + 3)

        # 3. openings in gaps

        def openings_in_gaps(walls=walls, window_runs=window_runs):
            gs = op.find_gaps(
                walls,
                min_w_px=px(p.min_opening_m),
                max_w_px=px(p.max_opening_m),
                blockers=window_runs,
            )
            return op.classify_gaps(
                gs, ink, ink_d, door_thr=p.door_score, min_window_th=px(p.min_exterior_m)
            )

        gaps = openings_in_gaps()
        # walls and windows outside the building footprint are dimension-chain artefacts
        fp = _footprint(
            walls, window_runs, gaps, ink.shape, face_lw, px(1.0) ** 2 * p.min_room_m2, m_per_px
        )
        if fp is not None:
            n_w, n_r = len(walls), len(window_runs)
            if _footprint_partial(fp, walls):
                # incomplete outline: keep what is wired to the footprint, drop loose pieces
                walls, window_runs = _connected_to_footprint(
                    fp, walls, window_runs, min_len=px(3.0)
                )
            else:
                walls = [w for w in walls if _inside(fp, w)]
                window_runs = [r for r in window_runs if _inside(fp, r)]
            if len(walls) != n_w or len(window_runs) != n_r:
                notes.append(
                    f"dropped {n_w - len(walls)} walls and {n_r - len(window_runs)} "
                    "windows outside the masonry footprint"
                )
                gaps = openings_in_gaps(walls, window_runs)
                gaps = [g for g in gaps if _inside(fp, g)]

        inner = [] if fp is None else [r for r in window_runs if _interior_both_sides(fp, r)]
        if not inner or attempt == 1:
            break
        demoted = {_run_key(r) for r in inner}
        notes.append(f"{len(inner)} window(s) inside the building treated as walls")

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


def _run_key(r: wl.Run) -> tuple:
    return (r.orient, r.comp, r.order)


def _interior_both_sides(interior: np.ndarray, r) -> bool:
    """True if the building interior lies on both sides of the run (at most positions)."""
    h, w = interior.shape
    hits = 0
    for t in (0.3, 0.5, 0.7):
        a = r.a0 + t * (r.a1 - r.a0)
        both = True
        for side in (-1, 1):
            n = r.c + side * (r.th / 2 + 6)
            x, y = (a, n) if r.orient == wl.H else (n, a)
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < h and 0 <= xi < w and interior[yi, xi]):
                both = False
        hits += both
    return hits >= 2


def _face_th(th: float, face_lw: float) -> float:
    """Face-to-face thickness of a run measured over the ink (outline pen width removed)."""
    return max(th - face_lw, 0.5 * th)


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


def _footprint(walls, window_runs, gaps, shape, face_lw, min_hole_px, mpp) -> np.ndarray | None:
    """Room-sized free space enclosed by all walls, windows and openings (the building interior);
    ``None`` if nothing is enclosed (open outline), in which case nothing is filtered."""
    if len(walls) < 3:
        return None
    m = np.zeros(shape, bool)
    for w in walls:
        rm.draw_rect(m, w.orient, w.c, w.a0 - 2, w.a1 + 2, w.th + face_lw + 2)
    for r in window_runs:
        rm.draw_rect(m, r.orient, r.c, r.a0 - 2, r.a1 + 2, r.th + face_lw + 2)

    def anchored(w) -> bool:  # part of the wall network, not an isolated blob
        return w.meta.get("snap0") or w.meta.get("snap1") or w.length * mpp >= 1.0

    for g in gaps:
        if g.right is not None or anchored(g.left):
            rm.draw_rect(m, g.orient, g.c, g.a0 - 2, g.a1 + 2, g.th + face_lw + 2)
    n, lab, outside = rm.free_space(m)
    areas = np.bincount(lab.ravel(), minlength=n)
    rooms = (~outside) & (areas >= min_hole_px)
    rooms[0] = False
    if not rooms.any():
        return None
    # the footprint is everything the outside cannot reach (walls and small enclosed slivers
    # included, e.g. between a wall and text touching it), restricted to the connected pieces
    # that enclose at least one room-sized space
    notout = ~outside
    notout[0] = True
    blob = notout[lab].astype(np.uint8)
    nb, blab = cv2.connectedComponents(blob, connectivity=4)
    hit = np.zeros(nb, bool)
    hit[np.unique(blab[rooms[lab]])] = True
    hit[0] = False
    return hit[blab]


def _footprint_partial(fp: np.ndarray, walls) -> bool:
    """True if the enclosed footprint touches under 65 % (by length) of the wall network: an
    incomplete outline (hatched walls with open ends) must not delete the walls outside it."""
    total = sum(w.length for w in walls)
    if total <= 0:
        return False
    frac = sum(w.length for w in walls if _inside(fp, w)) / total
    return frac < 0.65


def _seg_dist(a, b) -> float:
    """Distance between two axis-aligned runs (centre lines)."""
    (ax0, ay0), (ax1, ay1) = a.p0(), a.p1()
    (bx0, by0), (bx1, by1) = b.p0(), b.p1()
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _connected_to_footprint(fp: np.ndarray, walls, window_runs, min_len: float):
    """Walls and windows in networks (pieces touching within a few px) that border the
    footprint and are at least ``min_len`` long in total. Loose pieces - dimension ticks and
    extension lines, furniture - are dropped."""
    items = list(walls) + list(window_runs)
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _seg_dist(items[i], items[j]) <= 6.0:
                parent[find(i)] = find(j)
    total: dict[int, float] = {}
    seeded: set[int] = set()
    for i, r in enumerate(items):
        g = find(i)
        total[g] = total.get(g, 0.0) + r.length
        if _inside(fp, r):
            seeded.add(g)
    keep = {i for i in range(n) if find(i) in seeded and total[find(i)] >= min_len}
    k = len(walls)
    return ([w for i, w in enumerate(walls) if i in keep],
            [r for i, r in enumerate(window_runs) if i + k in keep])


def _inside(interior: np.ndarray, r) -> bool:
    """True if one side of the run borders the enclosed building interior."""
    h, w = interior.shape
    for t in (0.3, 0.5, 0.7):
        a = r.a0 + t * (r.a1 - r.a0)
        for side in (-1, 1):
            n = r.c + side * (r.th / 2 + 6)
            x, y = (a, n) if r.orient == wl.H else (n, a)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w and interior[yi, xi]:
                return True
    return False
