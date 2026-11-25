"""Close door gaps in solid (poché) walls, gated on evidence.

In historic plans a door is a plain gap in a solid wall: the segmentation finds the wall on both
sides and nothing in between, so the wall ends dangle and the rooms next to it stay open. A
dangling wall end is bridged here when

* the gap is door-sized (``door_gap_min_m`` .. ``door_gap_max_m``; up to ``max_bridge_m`` it is
  bridged as plain wall without an opening, a dropout of the segmentation),
* the bridge meets something: the collinear dangling end of another wall of the same thickness
  (thickness within ``thickness_compat_m``, centrelines within a small offset), or a
  perpendicular wall (so a gap towards open air, e.g. an entrance with nothing behind it, is left
  alone), and
* it makes the graph better: the set of bridges must raise the number of closed rooms, and each
  bridge kept must be needed for that (leave-one-out).

Each kept bridge is returned as a synthetic door opening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kelpod.contracts import OpeningKind, Point

from .graph import GraphBuilder, Seg, find_rooms
from .lines import OpeningM
from .params import GeometryParams


@dataclass
class _Cand:
    seg: Seg
    opening: OpeningM | None
    """None for a gap wider than a door: a dropout in the wall, bridged without an opening."""
    gap: float


def _dangling_ends(segs: list[Seg], tol: float) -> list[tuple[int, int]]:
    """(seg index, end 0/1) of ends that touch no other wall (end point or wall body)."""
    out = []
    for i, s in enumerate(segs):
        for e in (0, 1):
            p = s.p0 if e == 0 else s.p1
            touched = False
            for j, o in enumerate(segs):
                if j == i:
                    continue
                if _dist_point_seg(p, o.p0, o.p1) <= max(tol, 0.5 * o.t):
                    touched = True
                    break
            if not touched:
                out.append((i, e))
    return out


def _dist_point_seg(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _out_dir(s: Seg, e: int) -> tuple[Point, Point] | None:
    L = s.length
    if L < 1e-6:
        return None
    ux, uy = (s.p1[0] - s.p0[0]) / L, (s.p1[1] - s.p0[1]) / L
    return ((s.p0, (-ux, -uy)) if e == 0 else (s.p1, (ux, uy)))


def candidates(segs: list[Seg], prm: GeometryParams) -> list[_Cand]:
    ends = _dangling_ends(segs, prm.node_tol_m * 5)
    lo, hi = prm.door_gap_min_m, prm.max_bridge_m
    cands: list[_Cand] = []
    used: set[tuple[int, int]] = set()
    dang = set(ends)
    for i, e in sorted(ends, key=lambda ie: ie):
        if (i, e) in used:
            continue
        s = segs[i]
        r = _out_dir(s, e)
        if r is None or s.length < 0.3:
            continue
        (px, py), (ux, uy) = r
        best: _Cand | None = None
        # 1. collinear dangling end of another wall of the same thickness
        for j, e2 in dang:
            if j == i or (j, e2) in used:
                continue
            o = segs[j]
            r2 = _out_dir(o, e2)
            if r2 is None:
                continue
            (qx, qy), (vx, vy) = r2
            if ux * vx + uy * vy > -0.98:  # must point at each other
                continue
            along = (qx - px) * ux + (qy - py) * uy
            perp = abs((qx - px) * -uy + (qy - py) * ux)
            if not lo <= along <= hi or perp > max(0.08, 0.3 * min(s.t, o.t)):
                continue
            if abs(s.t - o.t) > prm.thickness_compat_m:
                continue
            if best is None or along < best.gap:
                a, b = (px, py), (qx, qy)
                best = _Cand(
                    Seg(a, b, (s.t + o.t) / 2, s.kind),
                    OpeningM(OpeningKind.DOOR, a, b, True) if along <= prm.door_gap_max_m else None,
                    along,
                )
                best_j = (j, e2)
        if best is not None:
            used.add((i, e))
            used.add(best_j)
            cands.append(best)
            continue
        # 2. a perpendicular wall behind the gap (open air behind it: not a door)
        hit = None
        for j, o in enumerate(segs):
            if j == i:
                continue
            dx, dy = o.p1[0] - o.p0[0], o.p1[1] - o.p0[1]
            L = math.hypot(dx, dy)
            if L < 1e-6 or abs((dx * ux + dy * uy) / L) > 0.2:
                continue
            den = ux * dy - uy * dx
            if abs(den) < 1e-9:
                continue
            t_ray = ((o.p0[0] - px) * dy - (o.p0[1] - py) * dx) / den
            t_seg = ((o.p0[0] - px) * uy - (o.p0[1] - py) * ux) / den
            if not (0.0 <= t_seg <= 1.0 or -0.1 / L <= t_seg <= 1.0 + 0.1 / L):
                continue
            width = t_ray - o.t / 2
            if not lo <= width <= hi:
                continue
            if hit is None or t_ray < hit[0]:
                hit = (t_ray, o, width)
        if hit is not None:
            t_ray, o, width = hit
            a = (px, py)
            b = (px + ux * t_ray, py + uy * t_ray)
            c = (px + ux * (t_ray - o.t / 2), py + uy * (t_ray - o.t / 2))
            cands.append(
                _Cand(
                    Seg(a, b, s.t, s.kind),
                    OpeningM(OpeningKind.DOOR, a, c, True) if width <= prm.door_gap_max_m else None,
                    width,
                )
            )
            used.add((i, e))
    return cands


def _n_rooms(segs: list[Seg], prm: GeometryParams) -> int:
    nodes, walls = GraphBuilder(segs, prm).finish()
    return len(find_rooms(nodes, walls, prm)[0])


def close_door_gaps(
    segs: list[Seg], prm: GeometryParams, issues: list[str]
) -> tuple[list[Seg], list[OpeningM]]:
    """Return ``segs`` plus the bridges that pass the evidence gate, and their door openings."""
    cands = candidates(segs, prm)[: prm.door_gap_max_candidates]
    if not cands:
        return segs, []
    base = _n_rooms(segs, prm)
    keep = list(cands)
    n_all = _n_rooms(segs + [c.seg for c in keep], prm)
    if n_all <= base:
        return segs, []
    # drop every bridge that is not needed for the gain (widest gaps first)
    for c in sorted(cands, key=lambda c: -c.gap):
        trial = [k for k in keep if k is not c]
        if _n_rooms(segs + [k.seg for k in trial], prm) >= n_all:
            keep = trial
    issues.append(
        f"Closed {len(keep)} door gap(s) in solid walls (room count {base} -> {n_all}); "
        "added as doors. Please check them."
    )
    return segs + [c.seg for c in keep], [c.opening for c in keep if c.opening is not None]


__all__ = ["close_door_gaps"]
