"""Plan graph (2D, metres) -> one-storey 3D building model.

Conventions (kept deliberately simple for the MVP):

* One storey ``EG`` at elevation 0. Every wall is extruded from the storey floor (z = 0) to the
  storey height, which is the *clear* room height (default 2.60 m). There is no ceiling slab yet,
  so wall height = clear height = space height.
* Openings without a sill or height get German defaults: doors sill 0.00 m, height 2.01 m
  (Rohbau-Richtmaß); windows sill 0.90 m, height 1.385 m. The head (sill + height) is clamped to
  the storey height minus a 5 cm lintel; widths are clamped to the wall length.
* One space per room, named after the room stamp or ``Raum n``.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from kelpod.contracts import (
    BuildingModel,
    Opening3D,
    OpeningKind,
    PlanGraph,
    Space,
    Storey,
    Wall3D,
)

DOOR_SILL_M = 0.0
DOOR_HEIGHT_M = 2.01
WINDOW_SILL_M = 0.90
WINDOW_HEIGHT_M = 1.385
MIN_LINTEL_M = 0.05
"""Minimum wall left above an opening head."""
MIN_OPENING_HEIGHT_M = 0.30


def _opening_heights(kind: OpeningKind, sill: float | None, height: float | None, storey_h: float):
    if kind == OpeningKind.DOOR:
        sill = DOOR_SILL_M if sill is None else sill
        height = DOOR_HEIGHT_M if height is None else height
    else:  # windows and unknown openings are treated as windows
        sill = WINDOW_SILL_M if sill is None else sill
        height = WINDOW_HEIGHT_M if height is None else height
    max_head = storey_h - MIN_LINTEL_M
    sill = min(max(sill, 0.0), max(max_head - MIN_OPENING_HEIGHT_M, 0.0))
    height = max(min(height, max_head - sill), min(MIN_OPENING_HEIGHT_M, max_head - sill))
    return sill, height


def build_model(
    graph: PlanGraph, *, storey_height_m: float = 2.60, name: str = "Kelpod building"
) -> BuildingModel:
    """Extrude the plan graph to a one-storey 3D model with default opening heights."""
    if storey_height_m <= MIN_LINTEL_M + MIN_OPENING_HEIGHT_M:
        raise ValueError(f"storey height {storey_height_m} m is too low")
    nodes = {n.id: (n.x, n.y) for n in graph.nodes}

    walls: list[Wall3D] = []
    lengths: dict[int, float] = {}
    for w in graph.walls:
        p0, p1 = nodes[w.start], nodes[w.end]
        length = math.dist(p0, p1)
        if length < 1e-6:
            continue  # degenerate wall, nothing to extrude
        lengths[w.id] = length
        walls.append(
            Wall3D(
                id=w.id,
                start=p0,
                end=p1,
                thickness_m=w.thickness_m,
                height_m=storey_height_m,
                kind=w.kind,
            )
        )

    openings: list[Opening3D] = []
    for o in graph.openings:
        if o.wall_id not in lengths:
            continue
        length = lengths[o.wall_id]
        width = min(max(o.width_m, 0.0), length)
        if width < 0.05:
            continue
        offset = min(max(o.offset_m, 0.0), length - width)
        sill, height = _opening_heights(o.kind, o.sill_m, o.height_m, storey_height_m)
        openings.append(
            Opening3D(
                id=o.id,
                kind=o.kind,
                wall_id=o.wall_id,
                offset_m=offset,
                width_m=width,
                sill_m=sill,
                height_m=height,
            )
        )

    spaces: list[Space] = []
    for i, r in enumerate(graph.rooms, start=1):
        if len(r.polygon) < 3:
            continue
        poly = Polygon(r.polygon)
        pts = list(r.polygon)
        if not poly.exterior.is_ccw:
            pts.reverse()
        spaces.append(
            Space(
                id=r.id,
                name=r.name or f"Raum {i}",
                polygon=pts,
                height_m=storey_height_m,
                area_m2=abs(poly.area),
            )
        )

    storey = Storey(
        name="EG",
        elevation_m=0.0,
        height_m=storey_height_m,
        walls=walls,
        openings=openings,
        spaces=spaces,
    )
    return BuildingModel(name=name, storeys=[storey])
