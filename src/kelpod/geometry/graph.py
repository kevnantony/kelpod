"""Wall segments (metres) -> node/wall graph, rooms, openings and stamp labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from kelpod.contracts import Node, Opening, OpeningKind, Point, Room, RoomStamp, Wall, WallKind
from kelpod.roomnames import is_exterior

from .lines import OpeningM
from .params import GeometryParams


@dataclass
class Seg:
    p0: Point
    p1: Point
    t: float
    kind: WallKind

    @property
    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _lerp(a: float, b: float, t: float) -> float:
    """Exact at the ends and for constant coordinates (keeps axis walls exactly axis-parallel)."""
    if a == b or t == 0.0:
        return a
    if t == 1.0:
        return b
    return a + t * (b - a)


def snap_free(free: list[Seg], fixed: list[Seg], tol: float) -> list[Seg]:
    """Snap ends of non-axis walls to nearby wall ends, else onto nearby walls."""
    out = []
    ends = [p for s in fixed for p in (s.p0, s.p1)]
    for f in free:
        new = []
        for p in (f.p0, f.p1):
            best, best_d = p, tol
            for q in ends:
                d = math.hypot(q[0] - p[0], q[1] - p[1])
                if d < best_d:
                    best, best_d = q, d
            if best is p:
                for s in fixed:
                    r = (s.p1[0] - s.p0[0], s.p1[1] - s.p0[1])
                    L2 = r[0] ** 2 + r[1] ** 2
                    if L2 < 1e-12:
                        continue
                    u = ((p[0] - s.p0[0]) * r[0] + (p[1] - s.p0[1]) * r[1]) / L2
                    if 0 <= u <= 1:
                        q = (s.p0[0] + u * r[0], s.p0[1] + u * r[1])
                        d = math.hypot(q[0] - p[0], q[1] - p[1])
                        if d < best_d:
                            best, best_d = q, d
            new.append(best)
            ends.append(best)
        out.append(Seg(new[0], new[1], f.t, f.kind))
    return out


def node_segments(segs: list[Seg], eps: float) -> list[Seg]:
    """Split segments at T-junctions and crossings (planar noding)."""
    splits: list[list[tuple[float, Point]]] = [[] for _ in segs]
    for i, s in enumerate(segs):
        p, r = s.p0, (s.p1[0] - s.p0[0], s.p1[1] - s.p0[1])
        Lr = math.hypot(*r)
        if Lr < 1e-9:
            continue
        for j in range(i + 1, len(segs)):
            o = segs[j]
            q = o.p0
            sv = (o.p1[0] - q[0], o.p1[1] - q[1])
            Ls = math.hypot(*sv)
            if Ls < 1e-9:
                continue
            den = _cross(r, sv)
            if abs(den) < 1e-12 * Lr * Ls:
                continue  # parallel: collinear pieces only touch at shared ends
            qp = (q[0] - p[0], q[1] - p[1])
            t = _cross(qp, sv) / den
            u = _cross(qp, r) / den
            et, eu = eps / Lr, eps / Ls
            if not (-et <= t <= 1 + et and -eu <= u <= 1 + eu):
                continue
            # exact point for axis-on-axis junctions, else interpolated on segment i
            if r[1] == 0 and sv[0] == 0:
                x = (q[0], p[1])
            elif r[0] == 0 and sv[1] == 0:
                x = (p[0], q[1])
            else:
                x = (_lerp(p[0], s.p1[0], t), _lerp(p[1], s.p1[1], t))
            if et < t < 1 - et:
                splits[i].append((t, x))
            if eu < u < 1 - eu:
                splits[j].append((u, x))
    out = []
    for s, ts in zip(segs, splits, strict=True):
        pts = [s.p0, *(x for _, x in sorted(ts)), s.p1]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            if a != b:
                out.append(Seg(a, b, s.t, s.kind))
    return out


class GraphBuilder:
    """Mutable node/wall graph with merge, spur and component clean-up."""

    def __init__(self, segs: list[Seg], prm: GeometryParams):
        self.prm = prm
        self.pts: list[Point] = []
        self.walls: dict[int, tuple[int, int, float, WallKind]] = {}
        seen: dict[tuple[int, int], int] = {}
        for s in segs:
            a, b = self._node(s.p0), self._node(s.p1)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                wid = seen[key]
                _, _, t, k = self.walls[wid]
                if s.t > t:
                    self.walls[wid] = (*key, s.t, s.kind)
                continue
            wid = len(seen)
            seen[key] = wid
            self.walls[wid] = (a, b, s.t, s.kind)

    def _node(self, p: Point) -> int:
        tol = self.prm.node_tol_m
        for i, q in enumerate(self.pts):
            if abs(q[0] - p[0]) <= tol and abs(q[1] - p[1]) <= tol:
                return i
        self.pts.append((float(p[0]), float(p[1])))
        return len(self.pts) - 1

    def length(self, wid: int) -> float:
        a, b, _, _ = self.walls[wid]
        return math.dist(self.pts[a], self.pts[b])

    def graph(self) -> nx.MultiGraph:
        g = nx.MultiGraph()
        for wid, (a, b, _, _) in self.walls.items():
            g.add_edge(a, b, key=wid)
        return g

    def remove_spurs(self) -> int:
        """Drop short dangling stubs hanging off junctions (overshoots, hatching remnants)."""
        removed = 0
        while True:
            deg: dict[int, int] = {}
            for a, b, _, _ in self.walls.values():
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
            victims = [
                wid
                for wid, (a, b, _, _) in self.walls.items()
                if self.length(wid) < self.prm.spur_len_m
                and ((deg[a] == 1 and deg[b] >= 3) or (deg[b] == 1 and deg[a] >= 3))
            ]
            if not victims:
                return removed
            for wid in victims:
                del self.walls[wid]
            removed += len(victims)

    def prune_components(self) -> tuple[int, float]:
        """Drop small loose wall groups without a closed loop (text, dimension lines, noise)."""
        g = self.graph()
        comps = []
        for nodes in nx.connected_components(g):
            sub = g.subgraph(nodes)
            wids = [k for _, _, k in sub.edges(keys=True)]
            total = sum(self.length(w) for w in wids)
            comps.append((wids, total, sub.number_of_edges() >= sub.number_of_nodes()))
        if not comps:
            return 0, 0.0
        biggest = max(c[1] for c in comps)
        n, dropped = 0, 0.0
        for wids, total, cyclic in comps:
            if total == biggest or cyclic:
                continue
            if total < self.prm.min_component_share * biggest or total < 1.0:
                for w in wids:
                    del self.walls[w]
                n += len(wids)
                dropped += total
        return n, dropped

    def merge_collinear(self) -> None:
        """Re-join walls split at nodes that no longer have a branch (degree 2, straight)."""
        changed = True
        while changed:
            changed = False
            inc: dict[int, list[int]] = {}
            for wid, (a, b, _, _) in self.walls.items():
                inc.setdefault(a, []).append(wid)
                inc.setdefault(b, []).append(wid)
            for n, ws in inc.items():
                if len(ws) != 2 or ws[0] == ws[1]:
                    continue
                (a1, b1, t1, k1), (a2, b2, t2, k2) = self.walls[ws[0]], self.walls[ws[1]]
                if abs(t1 - t2) > 1e-6 or k1 != k2:
                    continue
                o1 = b1 if a1 == n else a1
                o2 = b2 if a2 == n else a2
                if o1 == o2:
                    continue
                v1 = np.subtract(self.pts[o1], self.pts[n])
                v2 = np.subtract(self.pts[o2], self.pts[n])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9 or np.dot(v1, v2) / (n1 * n2) > -0.9999:
                    continue
                del self.walls[ws[1]]
                self.walls[ws[0]] = (o1, o2, t1, k1)
                changed = True
                break

    def finish(self) -> tuple[list[Node], list[Wall]]:
        used = sorted({n for a, b, _, _ in self.walls.values() for n in (a, b)})
        remap = {old: new for new, old in enumerate(used)}
        nodes = [Node(id=remap[o], x=self.pts[o][0], y=self.pts[o][1]) for o in used]
        walls = [
            Wall(id=i, start=remap[a], end=remap[b], thickness_m=round(t, 4), kind=k)
            for i, (a, b, t, k) in enumerate(self.walls.values())
        ]
        return nodes, walls


# ------------------------------------------------------------------------------------------------
# Rooms
# ------------------------------------------------------------------------------------------------


def wall_polygons(nodes: list[Node], walls: list[Wall]) -> list[Polygon]:
    """Wall bodies; ends at junctions extend by half the thickest other wall there so that
    L-corners and T-junctions close without poking into rooms."""
    pos = {n.id: (n.x, n.y) for n in nodes}
    at: dict[int, list[Wall]] = {}
    for w in walls:
        at.setdefault(w.start, []).append(w)
        at.setdefault(w.end, []).append(w)
    polys = []
    for w in walls:
        p0, p1 = np.array(pos[w.start]), np.array(pos[w.end])
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]]) * (w.thickness_m / 2)
        e0 = max((o.thickness_m for o in at[w.start] if o.id != w.id), default=0.0) / 2
        e1 = max((o.thickness_m for o in at[w.end] if o.id != w.id), default=0.0) / 2
        a, b = p0 - u * e0, p1 + u * e1
        polys.append(Polygon([a + nrm, b + nrm, b - nrm, a - nrm]))
    # non-axis junctions: fill the wedge with a disc
    for nid, ws in at.items():
        if len(ws) < 2:
            continue
        diag = False
        for w in ws:
            dx = pos[w.end][0] - pos[w.start][0]
            dy = pos[w.end][1] - pos[w.start][1]
            if min(abs(dx), abs(dy)) > 1e-6:
                diag = True
        if diag:
            r = max(w.thickness_m for w in ws) / 2
            polys.append(SPoint(pos[nid]).buffer(r, quad_segs=8))
    return polys


def find_rooms(nodes: list[Node], walls: list[Wall], prm: GeometryParams) -> tuple[list[Room], int]:
    """Rooms = holes in the union of wall bodies (inner faces), counter-clockwise."""
    polys = wall_polygons(nodes, walls)
    if not polys:
        return [], 0
    union = unary_union(polys)
    geoms = list(getattr(union, "geoms", [union]))
    rooms: list[Room] = []
    n_small = 0
    for g in geoms:
        if not isinstance(g, Polygon):
            continue
        for ring in g.interiors:
            hole = Polygon(ring).simplify(1e-4)
            if not hole.is_valid:
                hole = hole.buffer(0)
            if not isinstance(hole, Polygon) or hole.area < prm.min_room_area_m2:
                n_small += 1
                continue
            hole = orient(hole, sign=1.0)
            coords = [(round(x, 6), round(y, 6)) for x, y in list(hole.exterior.coords)[:-1]]
            rooms.append(Room(id=len(rooms), polygon=coords, area_m2=round(hole.area, 4)))
    return rooms, n_small


def label_rooms(
    rooms: list[Room], stamps: list[tuple[RoomStamp, Point]], issues: list[str]
) -> None:
    polys = [Polygon(r.polygon) for r in rooms]
    chosen: dict[int, tuple[float, RoomStamp]] = {}
    lost = []
    for st, p in stamps:
        sp = SPoint(p)
        hit = next((i for i, poly in enumerate(polys) if poly.contains(sp)), None)
        if hit is None:
            lost.append(st.name)
            continue
        d = polys[hit].representative_point().distance(sp)
        score = d - st.confidence  # prefer confident, central stamps
        if hit in chosen:
            issues.append(
                f"Room stamps '{chosen[hit][1].name}' and '{st.name}' fall into the same room; "
                "a wall between them may be missing."
            )
            if score >= chosen[hit][0]:
                continue
        chosen[hit] = (score, st)
    for i, (_, st) in chosen.items():
        rooms[i].name = st.name
        rooms[i].exterior = is_exterior(st.name)
        rooms[i].stamp_area_m2 = st.area_m2
    if lost:
        issues.append(
            f"{len(lost)} room stamp(s) lie outside every closed room ({', '.join(lost[:5])}); "
            "those rooms are probably not closed."
        )


# ------------------------------------------------------------------------------------------------
# Openings
# ------------------------------------------------------------------------------------------------


def attach_openings(
    openings: list[OpeningM],
    nodes: list[Node],
    walls: list[Wall],
    sills: list[tuple[float, Point]],
    prm: GeometryParams,
    issues: list[str],
) -> list[Opening]:
    pos = {n.id: np.array((n.x, n.y)) for n in nodes}
    placed: list[Opening] = []
    spans: dict[int, list[tuple[float, float]]] = {}
    unplaced = 0
    for o in openings:
        mid = np.array(o.mid)
        dvec = np.subtract(o.p1, o.p0)
        dl = float(np.linalg.norm(dvec))
        best, best_d = None, math.inf
        for w in walls:
            a, b = pos[w.start], pos[w.end]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            u = (b - a) / L
            if dl > 1e-6 and abs(float(np.dot(dvec / dl, u))) < math.cos(math.radians(15)):
                continue
            s = float(np.dot(mid - a, u))
            dist = abs(float(_cross(u, mid - a)))
            if -0.05 <= s <= L + 0.05 and dist <= max(0.6 * w.thickness_m, 0.2) and dist < best_d:
                best, best_d = (w, a, u, L), dist
        if best is None:
            unplaced += 1
            continue
        w, a, u, L = best
        s0 = float(np.dot(np.asarray(o.p0) - a, u))
        s1 = float(np.dot(np.asarray(o.p1) - a, u))
        lo, hi = max(0.0, min(s0, s1)), min(L, max(s0, s1))
        if hi - lo < prm.min_opening_width_m:
            unplaced += 1
            continue
        if any(min(hi, b2) - max(lo, a2) > 0.5 * (hi - lo) for a2, b2 in spans.get(w.id, [])):
            continue  # duplicate of an opening already placed
        spans.setdefault(w.id, []).append((lo, hi))
        if o.kind == OpeningKind.DOOR:
            sill, height = 0.0, prm.door_height_m
        elif o.kind == OpeningKind.WINDOW:
            sill, height = prm.window_sill_m, prm.window_height_m
        else:
            sill, height = None, None
        placed.append(
            Opening(
                id=len(placed),
                kind=o.kind,
                wall_id=w.id,
                offset_m=round(lo, 4),
                width_m=round(hi - lo, 4),
                sill_m=sill,
                height_m=height,
            )
        )
    _apply_sills(placed, nodes, walls, sills, prm)
    if unplaced:
        issues.append(f"{unplaced} detected opening(s) could not be placed on any wall.")
    return placed


def opening_center(o: Opening, pos: dict[int, np.ndarray], walls: dict[int, Wall]) -> np.ndarray:
    w = walls[o.wall_id]
    a, b = pos[w.start], pos[w.end]
    u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
    return a + u * (o.offset_m + o.width_m / 2)


def _apply_sills(placed, nodes, walls, sills, prm) -> None:
    """Each BRH label sets the sill of the nearest window; the head height stays at the default
    (sill 0.90 + 1.385 = 2.285 m), so the window height follows the sill."""
    if not sills:
        return
    pos = {n.id: np.array((n.x, n.y)) for n in nodes}
    wmap = {w.id: w for w in walls}
    windows = [o for o in placed if o.kind == OpeningKind.WINDOW]
    if not windows:
        return
    centres = [opening_center(o, pos, wmap) for o in windows]
    best: dict[int, tuple[float, float]] = {}
    for value, p in sills:
        ds = [float(np.linalg.norm(c - np.asarray(p))) for c in centres]
        i = int(np.argmin(ds))
        if ds[i] <= prm.brh_radius_m and (i not in best or ds[i] < best[i][0]):
            best[i] = (ds[i], value)
    head = prm.window_sill_m + prm.window_height_m
    for i, (_, value) in best.items():
        windows[i].sill_m = round(value, 4)
        windows[i].height_m = round(max(head - value, 0.3), 4)
