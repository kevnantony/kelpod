"""Random one-storey rectilinear layouts in plan space (metres), before any drawing.

The layout is a guillotine partition of a rectangular footprint. All sizes follow the German
octametric system (*oktametrische Maßordnung*, DIN 4172): clear room widths and opening widths are
``n * 12.5 cm + 1 cm``, outer dimensions ``n * 12.5 cm - 1 cm``. Every wall line has one thickness,
so every room is a rectangle bounded by inner wall faces.

Internal coordinates: the inner faces of the exterior walls span ``[0, inner_w] x [0, inner_h]``;
:func:`make_layout` shifts everything by ``origin`` so the result is in plan space (y up, origin at
the bottom-left corner of the image).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kelpod.contracts import (
    Node,
    Opening,
    OpeningKind,
    PlanGraph,
    Room,
    Wall,
    WallKind,
)

EXTERIOR_T = 0.365
INTERIOR_T = (0.24, 0.175)
DRYWALL_T = (0.115, 0.10)
OKTA = 0.125
MIN_ROOM_W = 1.135
"""Smallest clear room width (a narrow WC or Abstellraum)."""
WINDOW_WIDTHS = (1.01, 1.135, 1.26, 1.385, 1.51, 1.76, 2.01)
SMALL_WINDOW_WIDTHS = (0.635, 0.76, 0.885)
DOOR_HEIGHT = 2.01
WINDOW_HEAD = 2.26
"""Window head height above the floor; window height = head - sill."""
EPS = 1e-6


@dataclass
class Line:
    """A straight wall line (centreline) in internal coordinates."""

    id: int
    axis: str  # "h" (constant y) or "v" (constant x)
    coord: float  # centreline x (for "v") or y (for "h")
    lo: float  # centreline extent along the line
    hi: float
    thickness: float
    kind: WallKind


@dataclass
class Cell:
    """A room rectangle (clear inner faces) and the wall lines on its four sides."""

    x0: float
    y0: float
    x1: float
    y1: float
    # side -> line id, sides "s" (y0), "n" (y1), "w" (x0), "e" (x1)
    sides: dict[str, int] = field(default_factory=dict)
    depth: int = 0

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.w * self.h

    def side_interval(self, side: str) -> tuple[float, float]:
        return (self.x0, self.x1) if side in "sn" else (self.y0, self.y1)


@dataclass
class OpeningSpec:
    """An opening on a wall line; ``s0 < s1`` are gap edges along the line (internal coords)."""

    kind: OpeningKind
    line_id: int
    s0: float
    s1: float
    sill: float
    height: float
    room: int  # cell index the opening belongs to (door: the room it swings into)
    other_room: int | None = None  # door: room on the other side (None = outside)
    hinge_at_s0: bool = True
    window_lines: int = 2


@dataclass
class Layout:
    """A complete layout in plan space plus the drawing-relevant bookkeeping."""

    graph: PlanGraph
    lines: list[Line]
    cells: list[Cell]
    names: list[str]
    stamp_areas: list[float]
    openings: list[OpeningSpec]
    origin: tuple[float, float]
    """Internal (0, 0) = inner face corner, in plan-space metres."""
    outer: tuple[float, float, float, float]
    """Outer faces of the exterior walls (x0, y0, x1, y1), plan space."""

    def to_plan(self, x: float, y: float) -> tuple[float, float]:
        return (x + self.origin[0], y + self.origin[1])


def okta_clear(n: int) -> float:
    """Clear width (*Öffnungsmaß*) of ``n`` octametric units: ``n * 12.5 cm + 1 cm``."""
    return round(n * OKTA + 0.01, 4)


def _plausible(w: float, h: float) -> bool:
    """Narrow rooms (WC, hall, store) must stay short: no 1.2 m x 10 m strips."""
    short, long_ = min(w, h), max(w, h)
    return short >= 1.8 or long_ <= 3.5 * short


def _split(cell: Cell, axis: str, rng: np.random.Generator, t: float) -> float | None:
    """Pick a clear width for the first part of a split, or None if no split gives two plausible
    rooms."""
    length = cell.w if axis == "v" else cell.h
    across = cell.h if axis == "v" else cell.w
    lo_n = math.ceil((MIN_ROOM_W - 0.01) / OKTA - EPS)
    hi_n = math.floor((length - t - MIN_ROOM_W - 0.01) / OKTA + EPS)
    ok = [
        n
        for n in range(lo_n, hi_n + 1)
        if _plausible(okta_clear(n), across) and _plausible(length - t - okta_clear(n), across)
    ]
    if not ok:
        return None
    return okta_clear(int(rng.choice(ok)))


def _partition(
    inner_w: float, inner_h: float, n_rooms: int, rng: np.random.Generator
) -> tuple[list[Line], list[Cell]]:
    te = EXTERIOR_T
    lines = [
        Line(0, "h", -te / 2, -te / 2, inner_w + te / 2, te, WallKind.EXTERIOR),
        Line(1, "h", inner_h + te / 2, -te / 2, inner_w + te / 2, te, WallKind.EXTERIOR),
        Line(2, "v", -te / 2, -te / 2, inner_h + te / 2, te, WallKind.EXTERIOR),
        Line(3, "v", inner_w + te / 2, -te / 2, inner_h + te / 2, te, WallKind.EXTERIOR),
    ]
    cells = [Cell(0.0, 0.0, inner_w, inner_h, {"s": 0, "n": 1, "w": 2, "e": 3})]
    attempts = 0
    while len(cells) < n_rooms and attempts < 200:
        attempts += 1
        weights = np.array([c.area for c in cells]) ** 1.5
        i = int(rng.choice(len(cells), p=weights / weights.sum()))
        cell = cells[i]
        # split across the longer side most of the time
        prefer_v = cell.w >= cell.h
        if rng.random() < 0.2:
            prefer_v = not prefer_v
        # early (top-level) splits are load-bearing, later ones often light partitions
        if cell.depth == 0 or rng.random() < 0.45:
            kind, t = WallKind.INTERIOR, float(rng.choice(INTERIOR_T))
        else:
            kind, t = WallKind.DRYWALL, float(rng.choice(DRYWALL_T))
        for axis in ("v", "h") if prefer_v else ("h", "v"):
            a = _split(cell, axis, rng, t)
            if a is None:
                continue
            lid = len(lines)
            if axis == "v":
                xc = cell.x0 + a + t / 2
                lo = cell.y0 - lines[cell.sides["s"]].thickness / 2
                hi = cell.y1 + lines[cell.sides["n"]].thickness / 2
                lines.append(Line(lid, "v", xc, lo, hi, t, kind))
                left = Cell(cell.x0, cell.y0, cell.x0 + a, cell.y1, {**cell.sides, "e": lid})
                right = Cell(cell.x0 + a + t, cell.y0, cell.x1, cell.y1, {**cell.sides, "w": lid})
                new = [left, right]
            else:
                yc = cell.y0 + a + t / 2
                lo = cell.x0 - lines[cell.sides["w"]].thickness / 2
                hi = cell.x1 + lines[cell.sides["e"]].thickness / 2
                lines.append(Line(lid, "h", yc, lo, hi, t, kind))
                bottom = Cell(cell.x0, cell.y0, cell.x1, cell.y0 + a, {**cell.sides, "n": lid})
                top = Cell(cell.x0, cell.y0 + a + t, cell.x1, cell.y1, {**cell.sides, "s": lid})
                new = [bottom, top]
            for c in new:
                c.depth = cell.depth + 1
            cells[i : i + 1] = new
            break
    return lines, cells


def _name_rooms(cells: list[Cell]) -> list[str]:
    n = len(cells)
    order = sorted(range(n), key=lambda i: cells[i].area)  # small -> large
    names: list[str | None] = [None] * n
    names[order[-1]] = "Wohnen"
    rest = order[:-1]
    if n >= 4:
        # the narrowest elongated room becomes the hall
        elong = [i for i in rest if max(cells[i].w, cells[i].h) / min(cells[i].w, cells[i].h) > 1.8]
        if elong:
            hall = min(elong, key=lambda i: min(cells[i].w, cells[i].h))
            names[hall] = "Flur"
            rest = [i for i in rest if i != hall]
    small = ["WC", "Abst."] if n >= 6 else []
    for name in small + ["Bad"]:
        if rest:
            names[rest.pop(0)] = name
    pool = ["Küche", "Schlafen", "Kind", "Essen", "Arbeiten", "Gast"]
    rest_sorted = sorted(rest, key=lambda i: -cells[i].area)
    for k, i in enumerate(rest_sorted):
        names[i] = pool[k] if k < len(pool) else f"Zimmer {k + 1}"
    return [str(x) for x in names]


def _shared(a: Cell, b: Cell) -> tuple[int, float, float] | None:
    """Line id and clear overlap interval if two cells face each other across one wall line."""
    for sa, sb in (("e", "w"), ("w", "e"), ("n", "s"), ("s", "n")):
        if a.sides[sa] == b.sides[sb]:
            ia, ib = a.side_interval(sa), b.side_interval(sb)
            lo, hi = max(ia[0], ib[0]), min(ia[1], ib[1])
            if hi - lo > EPS:
                return a.sides[sa], lo, hi
    return None


class _Occupancy:
    """Keeps openings on one line from overlapping (with clearance)."""

    def __init__(self) -> None:
        self.by_line: dict[int, list[tuple[float, float]]] = {}

    def free(self, line_id: int, s0: float, s1: float, clearance: float = 0.25) -> bool:
        return all(
            s1 + clearance <= a or s0 >= b + clearance for a, b in self.by_line.get(line_id, [])
        )

    def add(self, line_id: int, s0: float, s1: float) -> None:
        self.by_line.setdefault(line_id, []).append((s0, s1))


def _place(
    lo: float, hi: float, width: float, rng: np.random.Generator, inset: float, centred: bool
) -> float | None:
    """Start of an opening of ``width`` inside ``[lo + inset, hi - inset]``."""
    free = hi - lo - 2 * inset - width
    if free < -EPS:
        return None
    free = max(free, 0.0)
    if centred:
        s0 = lo + inset + free / 2
    else:
        choice = rng.random()
        if choice < 0.35:
            s0 = lo + inset
        elif choice < 0.7:
            s0 = hi - inset - width
        else:
            s0 = lo + inset + float(rng.uniform(0, free))
    return round(s0 * 200) / 200  # snap to 5 mm


def _sill_for(name: str, rng: np.random.Generator) -> float:
    if name in ("Bad", "WC"):
        return float(rng.choice([1.20, 1.40]))
    if name == "Küche":
        return float(rng.choice([0.90, 1.10, 1.20]))
    return float(rng.choice([0.80, 0.875, 0.90, 0.90, 0.90]))


def _openings(
    cells: list[Cell], lines: list[Line], names: list[str], rng: np.random.Generator
) -> list[OpeningSpec]:
    occ = _Occupancy()
    out: list[OpeningSpec] = []
    wet = {"WC", "Bad", "Abst."}

    # doors: random spanning tree over the room adjacency graph, plus a few extra doors
    edges = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            s = _shared(cells[i], cells[j])
            if s and s[2] - s[1] >= 0.76 + 0.3:
                edges.append((i, j, *s))
    rng.shuffle(edges)
    # prefer connecting rooms through the hall (realistic and gives Flur many doors)
    edges.sort(key=lambda e: 0 if "Flur" in (names[e[0]], names[e[1]]) else 1)
    parent = list(range(len(cells)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    chosen = []
    for e in edges:
        ri, rj = find(e[0]), find(e[1])
        if ri != rj:
            parent[ri] = rj
            chosen.append(e)
        elif rng.random() < 0.12:
            chosen.append(e)
    for i, j, lid, lo, hi in chosen:
        small = names[i] in wet or names[j] in wet
        width = 0.76 if small else float(rng.choice([0.885, 0.885, 1.01]))
        if hi - lo < width + 0.3:
            width = 0.76
        s0 = _place(lo, hi, width, rng, inset=0.115 if rng.random() < 0.6 else 0.2, centred=False)
        if s0 is None or not occ.free(lid, s0, s0 + width):
            continue
        occ.add(lid, s0, s0 + width)
        # doors swing into the room, not into the hall
        into, other = (j, i) if names[i] == "Flur" else (i, j)
        if names[j] != "Flur" and rng.random() < 0.5:
            into, other = other, into
        out.append(
            OpeningSpec(
                OpeningKind.DOOR,
                lid,
                s0,
                s0 + width,
                0.0,
                DOOR_HEIGHT,
                into,
                other,
                hinge_at_s0=bool(rng.random() < 0.5),
            )
        )

    # windows (and the entrance door) on exterior sides
    exterior = {ln.id for ln in lines if ln.kind == WallKind.EXTERIOR}

    # entrance: the hall if it has an outside wall, else the room with the longest outside side
    def ext_len(ci: int) -> float:
        c = cells[ci]
        sides = [c.side_interval(s) for s in "snwe" if c.sides[s] in exterior]
        return max((b - a for a, b in sides), default=0.0)

    candidates = sorted(
        range(len(cells)), key=lambda ci: (names[ci] != "Flur" or ext_len(ci) < 1.6, -ext_len(ci))
    )
    entrance_room = candidates[0]
    entrance_done = False
    window_lines = int(rng.choice([2, 3]))
    for ci, cell in enumerate(cells):
        ext_sides = [s for s in "snwe" if cell.sides[s] in exterior]
        rng.shuffle(ext_sides)
        if ci == entrance_room:
            # the longest outside side, so the door always fits
            side = max(ext_sides, key=lambda s: cell.side_interval(s)[1] - cell.side_interval(s)[0])
            lo, hi = cell.side_interval(side)
            s0 = _place(lo, hi, 1.01, rng, inset=0.25, centred=False)
            if s0 is not None:
                occ.add(cell.sides[side], s0, s0 + 1.01)
                out.append(
                    OpeningSpec(
                        OpeningKind.DOOR,
                        cell.sides[side],
                        s0,
                        s0 + 1.01,
                        0.0,
                        DOOR_HEIGHT,
                        ci,
                        None,
                        hinge_at_s0=bool(rng.random() < 0.5),
                    )
                )
                entrance_done = True
        for k, side in enumerate(ext_sides):
            name = names[ci]
            if name == "Abst." or (name == "Flur" and rng.random() < 0.5):
                continue
            if k > 0 and rng.random() < 0.45:
                continue  # not every outside wall gets a window
            lo, hi = cell.side_interval(side)
            widths = SMALL_WINDOW_WIDTHS if name in wet or name == "Flur" else WINDOW_WIDTHS
            fitting = [w for w in widths if w <= hi - lo - 0.6]
            if not fitting:
                continue
            width = float(rng.choice(fitting))
            s0 = _place(lo, hi, width, rng, inset=0.3, centred=rng.random() < 0.6)
            lid = cell.sides[side]
            if s0 is None or not occ.free(lid, s0, s0 + width):
                continue
            occ.add(lid, s0, s0 + width)
            sill = _sill_for(name, rng)
            out.append(
                OpeningSpec(
                    OpeningKind.WINDOW,
                    lid,
                    s0,
                    s0 + width,
                    sill,
                    round(WINDOW_HEAD - sill, 4),
                    ci,
                    None,
                    window_lines=window_lines,
                )
            )
    if not entrance_done:
        raise RuntimeError("no entrance door placed")
    return out


def _graph(
    lines: list[Line],
    cells: list[Cell],
    names: list[str],
    stamp_areas: list[float],
    openings: list[OpeningSpec],
    origin: tuple[float, float],
    m_per_px: float,
    image_h: int,
) -> PlanGraph:
    ox, oy = origin
    node_ids: dict[tuple[float, float], int] = {}
    nodes: list[Node] = []

    def node(x: float, y: float) -> int:
        key = (round(x + ox, 6), round(y + oy, 6))
        if key not in node_ids:
            node_ids[key] = len(nodes)
            nodes.append(Node(id=len(nodes), x=key[0], y=key[1]))
        return node_ids[key]

    walls: list[Wall] = []
    wall_span: list[tuple[int, float, float]] = []  # (line id, lo, hi) per wall
    for ln in lines:
        stops = {ln.lo, ln.hi}
        for other in lines:
            if other.axis == ln.axis:
                continue
            for end in (other.lo, other.hi):
                # other's endpoint lies on ln?
                if abs(end - ln.coord) < EPS and ln.lo - EPS <= other.coord <= ln.hi + EPS:
                    stops.add(other.coord)
        pts = sorted(stops)
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            if b - a < EPS:
                continue
            if ln.axis == "h":
                s, e = node(a, ln.coord), node(b, ln.coord)
            else:
                s, e = node(ln.coord, a), node(ln.coord, b)
            walls.append(
                Wall(id=len(walls), start=s, end=e, thickness_m=ln.thickness, kind=ln.kind)
            )
            wall_span.append((ln.id, a, b))

    ops: list[Opening] = []
    for op in openings:
        for wid, (lid, a, b) in enumerate(wall_span):
            if lid == op.line_id and a - EPS <= op.s0 and op.s1 <= b + EPS:
                ops.append(
                    Opening(
                        id=len(ops),
                        kind=op.kind,
                        wall_id=wid,
                        offset_m=round(op.s0 - a, 6),
                        width_m=round(op.s1 - op.s0, 6),
                        sill_m=op.sill,
                        height_m=op.height,
                    )
                )
                break
        else:  # pragma: no cover - by construction every opening lies within one wall
            raise RuntimeError(f"opening {op} not on a single wall")

    rooms = []
    for i, c in enumerate(cells):
        poly = [
            (c.x0 + ox, c.y0 + oy),
            (c.x1 + ox, c.y0 + oy),
            (c.x1 + ox, c.y1 + oy),
            (c.x0 + ox, c.y1 + oy),
        ]
        poly = [(round(x, 6), round(y, 6)) for x, y in poly]
        rooms.append(
            Room(
                id=i,
                polygon=poly,
                name=names[i],
                area_m2=round(c.area, 6),
                stamp_area_m2=stamp_areas[i],
            )
        )
    return PlanGraph(
        nodes=nodes,
        walls=walls,
        openings=ops,
        rooms=rooms,
        m_per_px=m_per_px,
        scale_source="ground_truth",
        image_height_px=image_h,
    )


def make_layout(
    rng: np.random.Generator, *, margin_m: float, m_per_px: float, n_rooms: int | None = None
) -> tuple[Layout, tuple[int, int]]:
    """Draw a random layout. Returns the layout and the image size ``(width_px, height_px)``."""
    if n_rooms is None:
        n_rooms = int(rng.integers(3, 9))
    if not 1 <= n_rooms <= 12:
        raise ValueError("n_rooms must be in 1..12")
    for _ in range(50):
        area = n_rooms * float(rng.uniform(11.0, 17.0))
        aspect = float(rng.uniform(1.0, 1.6))
        outer_w = math.sqrt(area * aspect) + 2 * EXTERIOR_T
        outer_h = area / (outer_w - 2 * EXTERIOR_T) + 2 * EXTERIOR_T
        if rng.random() < 0.5:
            outer_w, outer_h = outer_h, outer_w
        # outer dimensions n * 12.5 cm - 1 cm (Außenmaß)
        outer_w = round(round((outer_w + 0.01) / OKTA) * OKTA - 0.01, 4)
        outer_h = round(round((outer_h + 0.01) / OKTA) * OKTA - 0.01, 4)
        inner_w, inner_h = outer_w - 2 * EXTERIOR_T, outer_h - 2 * EXTERIOR_T
        lines, cells = _partition(inner_w, inner_h, n_rooms, rng)
        if len(cells) == n_rooms:
            break
    else:  # pragma: no cover
        raise RuntimeError("could not partition the footprint")

    names = _name_rooms(cells)
    stamp_areas = [round(c.area + 1e-9, 2) for c in cells]
    openings = _openings(cells, lines, names, rng)

    width_px = math.ceil((outer_w + 2 * margin_m) / m_per_px)
    height_px = math.ceil((outer_h + 2 * margin_m) / m_per_px)
    # origin: put the outer face corner at (margin, margin) in plan space
    origin = (margin_m + EXTERIOR_T, margin_m + EXTERIOR_T)
    graph = _graph(lines, cells, names, stamp_areas, openings, origin, m_per_px, height_px)
    outer = (margin_m, margin_m, margin_m + outer_w, margin_m + outer_h)
    layout = Layout(graph, lines, cells, names, stamp_areas, openings, origin, outer)
    return layout, (width_px, height_px)
