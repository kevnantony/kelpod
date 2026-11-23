"""Axis-parallel wall lines: snap angles, cluster coordinates, merge and bridge pieces, and
extend/trim wall ends to the perpendicular walls they meet.

Axis walls are kept as *pieces* on *lines*: a line is one shared coordinate (``y`` for horizontal,
``x`` for vertical walls), a piece is an interval along it. A piece end that meets a perpendicular
wall stores a reference to that wall's line, so when the dimension fit moves a line, every wall end
attached to it moves along and junctions stay closed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from kelpod.contracts import OpeningKind, Point, WallKind

from .params import STANDARD_THICKNESSES_M, GeometryParams


@dataclass
class WallM:
    """A wall in plan space before it is sorted into axis pieces or free walls."""

    p0: Point
    p1: Point
    t: float
    kind: WallKind


@dataclass
class Line:
    orient: str  # "h": coord is y; "v": coord is x
    coord: float


@dataclass
class Piece:
    orient: str
    c: float
    a: float
    b: float
    t: float
    kind: WallKind
    line: int = -1
    ref_a: int | None = None
    ref_b: int | None = None

    @property
    def length(self) -> float:
        return self.b - self.a


@dataclass
class OpeningM:
    kind: OpeningKind
    p0: Point
    p1: Point
    synthetic: bool = False

    @property
    def mid(self) -> Point:
        return ((self.p0[0] + self.p1[0]) / 2, (self.p0[1] + self.p1[1]) / 2)

    @property
    def width(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])


def snap_thickness(t: float, prm: GeometryParams) -> float:
    best = min(STANDARD_THICKNESSES_M, key=lambda s: abs(s - t))
    return best if abs(best - t) <= prm.thickness_snap_m else t


def classify(walls: list[WallM], prm: GeometryParams) -> tuple[list[Piece], list[WallM]]:
    """Split walls into axis-parallel pieces and free (diagonal / odd-angle) walls."""
    pieces: list[Piece] = []
    free: list[WallM] = []
    for w in walls:
        (x0, y0), (x1, y1) = w.p0, w.p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        if min(ang, 180.0 - ang) <= prm.snap_deg:
            pieces.append(Piece("h", (y0 + y1) / 2, min(x0, x1), max(x0, x1), w.t, w.kind))
        elif abs(ang - 90.0) <= prm.snap_deg:
            pieces.append(Piece("v", (x0 + x1) / 2, min(y0, y1), max(y0, y1), w.t, w.kind))
        else:
            for target in (45.0, 135.0):
                if abs(ang - target) <= prm.diag_deg:
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    r = math.radians(target)
                    hx, hy = math.cos(r) * length / 2, math.sin(r) * length / 2
                    w = WallM((mx - hx, my - hy), (mx + hx, my + hy), w.t, w.kind)
                    break
            free.append(w)
    return pieces, free


def cluster_lines(pieces: list[Piece], prm: GeometryParams) -> list[Line]:
    """Give parallel pieces with (almost) the same coordinate one shared, length-weighted coordinate."""
    lines: list[Line] = []
    for orient in ("h", "v"):
        ps = sorted((p for p in pieces if p.orient == orient), key=lambda p: p.c)
        groups: list[list[Piece]] = []
        for p in ps:
            g = groups[-1] if groups else None
            if g and p.c - g[-1].c <= prm.line_tol_m and p.c - g[0].c <= 2 * prm.line_tol_m:
                g.append(p)
            else:
                groups.append([p])
        for g in groups:
            wsum = sum(max(p.length, 1e-6) for p in g)
            coord = sum(p.c * max(p.length, 1e-6) for p in g) / wsum
            lid = len(lines)
            lines.append(Line(orient, coord))
            for p in g:
                p.line = lid
                p.c = coord
    return lines


def _compatible(p: Piece, q: Piece, prm: GeometryParams) -> bool:
    return abs(p.t - q.t) <= prm.thickness_compat_m


def _absorb(cur: Piece, p: Piece) -> None:
    la, lb = max(cur.length, 1e-6), max(p.length, 1e-6)
    cur.t = (cur.t * la + p.t * lb) / (la + lb)
    if lb > la:
        cur.kind = p.kind
    cur.b = max(cur.b, p.b)


def _merge_line(ps: list[Piece], prm: GeometryParams) -> list[Piece]:
    ps = sorted(ps, key=lambda p: p.a)
    out = [ps[0]]
    for p in ps[1:]:
        cur = out[-1]
        if p.a - cur.b > prm.merge_gap_m:
            out.append(p)
        elif _compatible(cur, p, prm):
            _absorb(cur, p)
        elif p.b <= cur.b:
            continue  # an incompatible piece inside another one: noise
        else:
            m = (cur.b + p.a) / 2
            cur.b, p.a = m, m
            out.append(p)
    return out


def end_tol(p: Piece, q: Piece, prm: GeometryParams) -> float:
    return max(prm.end_tol_min_m, prm.end_tol_factor * max(p.t, q.t))


def junction_at(p: Piece, e: float, pieces: list[Piece], prm: GeometryParams) -> Piece | None:
    """The nearest perpendicular piece that the end ``e`` of ``p`` should meet, if any."""
    best, best_d = None, math.inf
    for q in pieces:
        if q.orient == p.orient:
            continue
        tol = end_tol(p, q, prm)
        d = abs(q.c - e)
        if d <= tol and q.a - tol <= p.c <= q.b + tol and d < best_d:
            best, best_d = q, d
    return best


def _covering_opening(
    line: Line, lo: float, hi: float, t: float, openings: list[OpeningM]
) -> OpeningM | None:
    gap = hi - lo
    for o in openings:
        (x0, y0), (x1, y1) = o.p0, o.p1
        if line.orient == "h":
            along, perp, d_al, d_pp = (x0, x1), (y0 + y1) / 2, abs(x1 - x0), abs(y1 - y0)
        else:
            along, perp, d_al, d_pp = (y0, y1), (x0 + x1) / 2, abs(y1 - y0), abs(x1 - x0)
        if d_pp > d_al and d_al > 1e-6:
            continue  # opening runs across this line
        if abs(perp - line.coord) > max(0.25, 0.75 * t):
            continue
        oa, ob = min(along), max(along)
        if ob - oa < 1e-6:
            if lo - 0.05 <= oa <= hi + 0.05:
                return o
            continue
        overlap = min(ob, hi) - max(oa, lo)
        if overlap >= 0.5 * min(gap, ob - oa):
            return o
    return None


def merge_and_bridge(
    pieces: list[Piece],
    lines: list[Line],
    openings: list[OpeningM],
    prm: GeometryParams,
    issues: list[str],
) -> tuple[list[Piece], list[OpeningM]]:
    """Merge collinear pieces on each line, then bridge gaps that are openings or dropouts."""
    by_line: dict[int, list[Piece]] = defaultdict(list)
    for p in pieces:
        by_line[p.line].append(p)
    merged = {lid: _merge_line(ps, prm) for lid, ps in by_line.items()}
    flat = [p for ps in merged.values() for p in ps]

    synthetic: list[OpeningM] = []
    n_free_bridges = 0
    out: list[Piece] = []
    for lid, ps in merged.items():
        line = lines[lid]
        res = [ps[0]]
        for p in ps[1:]:
            cur = res[-1]
            lo, hi = cur.b, p.a
            gap = hi - lo
            if gap > prm.max_bridge_m:
                res.append(p)
                continue
            op = _covering_opening(line, lo, hi, max(cur.t, p.t), openings)
            free_ends = (
                junction_at(cur, lo, flat, prm) is None and junction_at(p, hi, flat, prm) is None
            )
            if op is None and gap > prm.small_gap_m and not free_ends:
                res.append(p)  # e.g. a corridor between two walls that end at cross walls
                continue
            if op is None and gap > prm.small_gap_m:
                n_free_bridges += 1
                if line.orient == "h":
                    a, b = (lo, line.coord), (hi, line.coord)
                else:
                    a, b = (line.coord, lo), (line.coord, hi)
                synthetic.append(OpeningM(OpeningKind.UNKNOWN, a, b, synthetic=True))
            if _compatible(cur, p, prm):
                _absorb(cur, p)
            else:
                m = (lo + hi) / 2
                cur.b, p.a = m, m
                res.append(p)
        out.extend(res)
    if n_free_bridges:
        issues.append(
            f"Closed {n_free_bridges} gap(s) in walls where no door or window was detected; "
            "they were added as openings of unknown type. Please check them."
        )
    for p in out:
        p.t = snap_thickness(p.t, prm)
    return out, synthetic


def snap_ends(pieces: list[Piece], prm: GeometryParams) -> list[Piece]:
    """Extend or trim each piece end to the nearest perpendicular wall it should meet."""
    updates = []
    for p in pieces:
        qa = junction_at(p, p.a, pieces, prm)
        qb = junction_at(p, p.b, pieces, prm)
        updates.append((qa, qb))
    out = []
    for p, (qa, qb) in zip(pieces, updates, strict=True):
        if qa is not None:
            p.a, p.ref_a = qa.c, qa.line
        if qb is not None:
            p.b, p.ref_b = qb.c, qb.line
        if p.b - p.a > 1e-6:
            out.append(p)
    return out
