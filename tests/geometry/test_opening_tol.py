"""An opening a few dm beside its (dimension-fitted) wall must still be placed on that wall."""

from kelpod.contracts import Node, OpeningKind, WallKind
from kelpod.geometry.graph import attach_openings
from kelpod.geometry.lines import OpeningM
from kelpod.geometry.params import GeometryParams


def _wall_graph():
    from kelpod.contracts import Wall

    nodes = [Node(id=0, x=8.92, y=4.18), Node(id=1, x=8.92, y=9.83)]
    walls = [Wall(id=0, start=0, end=1, thickness_m=0.115, kind=WallKind.INTERIOR)]
    return nodes, walls


def test_door_22cm_off_thin_wall_is_placed():
    nodes, walls = _wall_graph()
    door = OpeningM(OpeningKind.DOOR, (8.70, 4.56), (8.70, 5.43), False)
    issues: list[str] = []
    out = attach_openings([door], nodes, walls, [], GeometryParams(), issues)
    assert len(out) == 1 and out[0].kind == OpeningKind.DOOR and out[0].wall_id == 0


def test_opening_far_from_every_wall_is_not_placed():
    nodes, walls = _wall_graph()
    door = OpeningM(OpeningKind.DOOR, (8.40, 4.56), (8.40, 5.43), False)
    assert attach_openings([door], nodes, walls, [], GeometryParams(), []) == []
