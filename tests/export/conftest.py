"""Hand-built plan graphs for the export tests (other modules are built in parallel)."""

from __future__ import annotations

import pytest

from kelpod.contracts import Node, Opening, OpeningKind, PlanGraph, Room, Wall, WallKind

EXT = 0.365
INT = 0.24


def two_room_house() -> PlanGraph:
    """10 x 8 m (centrelines) house, one interior wall at x = 6 split into two rooms.

    The top and bottom exterior walls are split at the T-junctions with the interior wall, like
    the geometry step does. One door in the interior wall, two windows in exterior walls.
    """
    pts = {1: (0, 0), 2: (6, 0), 3: (10, 0), 4: (10, 8), 5: (6, 8), 6: (0, 8)}
    nodes = [Node(id=i, x=x, y=y) for i, (x, y) in pts.items()]
    ext = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 5, 6), (6, 6, 1)]
    walls = [Wall(id=i, start=a, end=b, thickness_m=EXT, kind=WallKind.EXTERIOR) for i, a, b in ext]
    walls.append(Wall(id=7, start=2, end=5, thickness_m=INT, kind=WallKind.INTERIOR))
    e, i = EXT / 2, INT / 2
    left = [(e, e), (6 - i, e), (6 - i, 8 - e), (e, 8 - e)]
    right = [(6 + i, e), (10 - e, e), (10 - e, 8 - e), (6 + i, 8 - e)]

    def area(p):
        return (p[1][0] - p[0][0]) * (p[2][1] - p[1][1])

    rooms = [
        Room(id=1, polygon=left, name="Wohnen", area_m2=area(left), stamp_area_m2=43.0),
        Room(id=2, polygon=right, name="Küche", area_m2=area(right)),
    ]
    openings = [
        Opening(id=1, kind=OpeningKind.DOOR, wall_id=7, offset_m=3.0, width_m=0.885),
        Opening(id=2, kind=OpeningKind.WINDOW, wall_id=1, offset_m=2.0, width_m=1.51),
        Opening(id=3, kind=OpeningKind.WINDOW, wall_id=3, offset_m=3.0, width_m=1.26, sill_m=0.9),
    ]
    return PlanGraph(
        nodes=nodes,
        walls=walls,
        openings=openings,
        rooms=rooms,
        m_per_px=0.0127,
        scale_source="ground_truth",
        image_height_px=1000,
    )


@pytest.fixture
def house() -> PlanGraph:
    return two_room_house()
