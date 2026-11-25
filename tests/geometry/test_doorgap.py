"""Door gaps in solid walls: bridged only with evidence (see geometry/doorgap.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from geo_fixtures import INT, Drawer, ocr, page, seg, two_room_house  # noqa: E402

from kelpod import geometry  # noqa: E402
from kelpod.contracts import OpeningKind, WallKind  # noqa: E402
from kelpod.geometry.doorgap import candidates, close_door_gaps  # noqa: E402
from kelpod.geometry.graph import Seg  # noqa: E402
from kelpod.geometry.params import GeometryParams  # noqa: E402


def _house(end_x: float, dr: Drawer):
    walls, rooms = two_room_house(dr)
    # a cross wall in the right half that stops short of the east wall (a door beside the corner)
    walls.append(dr.wall((4, 3), (end_x, 3), INT, WallKind.INTERIOR))
    return walls, rooms


def test_door_gap_beside_cross_wall_closes_room_and_adds_door():
    dr = Drawer(seed=1)
    walls, rooms = _house(7.0, dr)  # 1.0 m centreline gap to the east wall
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert len(g.rooms) == 3, g.report.issues
    assert any(o.kind == OpeningKind.DOOR for o in g.openings)
    assert any("door gap" in i for i in g.report.issues)


def test_no_bridge_when_nothing_is_within_reach():
    dr = Drawer(seed=1)
    walls, rooms = _house(5.0, dr)  # 3 m to the east wall: not a door, not a dropout
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert len(g.rooms) == 2
    assert not any("door gap" in i for i in g.report.issues)


def test_collinear_ends_of_different_thickness_are_not_candidates():
    prm = GeometryParams()
    k = WallKind.INTERIOR
    same = [Seg((0, 0), (2, 0), 0.24, k), Seg((3.0, 0), (5, 0), 0.24, k)]
    other = [Seg((0, 0), (2, 0), 0.24, k), Seg((3.0, 0), (5, 0), 0.49, k)]
    assert len(candidates(same, prm)) == 1
    assert candidates(other, prm) == []


def test_bridge_that_closes_no_room_is_dropped():
    prm = GeometryParams()
    k = WallKind.INTERIOR
    segs = [Seg((0, 0), (2, 0), 0.24, k), Seg((3.0, 0), (5, 0), 0.24, k)]
    issues: list[str] = []
    out, doors = close_door_gaps(segs, prm, issues)
    assert out == segs and doors == [] and issues == []
