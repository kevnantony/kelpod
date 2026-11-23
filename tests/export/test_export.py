from __future__ import annotations

import math

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.shape
import numpy as np
import pytest
from conftest import EXT, INT, two_room_house
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from kelpod.contracts import (
    BuildingModel,
    Node,
    Opening,
    OpeningKind,
    PlanGraph,
    Room,
    Storey,
    Wall,
    Wall3D,
)
from kelpod.export import build_model, corner_extensions, path_connections, write_ifc
from kelpod.export.ifc import schema_problems

SCHEMAS = ["IFC4", "IFC2X3"]


# ------------------------------------------------------------------------------------------
# build_model
# ------------------------------------------------------------------------------------------


def test_build_model_defaults(house):
    model = build_model(house, name="Haus")
    assert model.name == "Haus"
    (storey,) = model.storeys
    assert storey.name == "EG" and storey.elevation_m == 0.0 and storey.height_m == 2.60
    assert len(storey.walls) == 7 and all(w.height_m == 2.60 for w in storey.walls)
    by_id = {o.id: o for o in storey.openings}
    assert (by_id[1].sill_m, by_id[1].height_m) == (0.0, 2.01)  # door default
    assert (by_id[2].sill_m, by_id[2].height_m) == (0.90, 1.385)  # window default
    assert [s.name for s in storey.spaces] == ["Wohnen", "Küche"]
    assert storey.spaces[0].area_m2 == pytest.approx(house.rooms[0].area_m2)


def test_build_model_clamps_and_cleans():
    g = PlanGraph(
        nodes=[Node(id=1, x=0, y=0), Node(id=2, x=2, y=0), Node(id=3, x=2, y=0)],
        walls=[
            Wall(id=1, start=1, end=2, thickness_m=0.24),
            Wall(id=2, start=2, end=3, thickness_m=0.24),  # zero length
        ],
        openings=[
            Opening(
                id=1,
                kind=OpeningKind.WINDOW,
                wall_id=1,
                offset_m=1.5,
                width_m=1.0,
                sill_m=1.8,
                height_m=1.5,
            ),
            Opening(id=2, kind=OpeningKind.DOOR, wall_id=2, offset_m=0, width_m=0.9),
            Opening(id=3, kind=OpeningKind.DOOR, wall_id=1, offset_m=-1, width_m=5.0),
        ],
        rooms=[Room(id=1, polygon=[(0, 0), (0, 1), (1, 1), (1, 0)], area_m2=1.0)],  # clockwise
        m_per_px=0.01,
        scale_source="default",
        image_height_px=100,
    )
    (storey,) = build_model(g, storey_height_m=2.5).storeys
    assert [w.id for w in storey.walls] == [1]
    win, door = storey.openings
    assert win.offset_m + win.width_m <= 2.0 + 1e-9  # pulled back onto the wall
    assert win.sill_m + win.height_m <= 2.5 - 0.05 + 1e-9  # head below the top of the wall
    assert door.offset_m == 0 and door.width_m == 2.0
    space = storey.spaces[0]
    assert space.name == "Raum 1"
    assert _ccw(space.polygon)


def _ccw(pts):
    return (
        sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True))
        > 0
    )


def test_build_model_rejects_tiny_storey(house):
    with pytest.raises(ValueError):
        build_model(house, storey_height_m=0.2)


# ------------------------------------------------------------------------------------------
# Corners and connections (pure geometry)
# ------------------------------------------------------------------------------------------


def _w(i, a, b, t=0.3):
    return Wall3D(id=i, start=a, end=b, thickness_m=t, height_m=2.6)


def test_l_corner_butt_joint():
    walls = [_w(1, (0, 0), (5, 0), 0.365), _w(2, (5, 0), (5, 3), 0.24)]
    ext = corner_extensions(walls)
    assert ext[1] == (0.0, pytest.approx(0.12))  # wall 1 is longer: extend by 0.24 / 2
    assert ext[2] == (pytest.approx(-0.1825), 0.0)  # wall 2 trimmed to wall 1's face


def test_t_junction_and_straight_joint_not_extended():
    walls = [_w(1, (0, 0), (3, 0)), _w(2, (3, 0), (6, 0)), _w(3, (3, 0), (3, 4))]
    ext = corner_extensions(walls)
    assert ext[3][0] == 0.0 and ext[1][1] == 0.0 and ext[2][0] == 0.0


def test_path_connections_types():
    walls = [_w(1, (0, 0), (6, 0)), _w(2, (3, 0), (3, 4)), _w(3, (6, 0), (6, 4))]
    conns = {(a, b): (ta, tb) for a, ta, b, tb in path_connections(walls)}
    assert conns[(1, 3)] == ("ATEND", "ATSTART")
    assert conns[(1, 2)] == ("ATPATH", "ATSTART")  # wall 2 ends on unsplit wall 1


# ------------------------------------------------------------------------------------------
# IFC export
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=SCHEMAS)
def exported(request, tmp_path_factory):
    schema = request.param
    path = tmp_path_factory.mktemp("ifc") / f"house_{schema}.ifc"
    model = build_model(two_room_house(), name="Testhaus")
    result = write_ifc(model, path, schema=schema, floor_slab=True)
    return schema, result, ifcopenshell.open(str(path))


def test_export_counts_and_clean(exported):
    schema, result, f = exported
    assert result.reopened_ok and result.schema_ == schema
    assert result.warnings == []
    wall_cls = "IfcWallStandardCase" if schema == "IFC2X3" else "IfcWall"
    expected = {
        "IfcProject": 1,
        "IfcSite": 1,
        "IfcBuilding": 1,
        "IfcBuildingStorey": 1,
        wall_cls: 7,
        "IfcOpeningElement": 3,
        "IfcDoor": 1,
        "IfcWindow": 2,
        "IfcSpace": 2,
        "IfcSlab": 1,
        "IfcRelVoidsElement": 3,
        "IfcRelFillsElement": 3,
        "IfcRelConnectsPathElements": 10,
    }
    for cls, n in expected.items():
        assert result.entity_counts.get(cls) == n, cls
    assert f.schema == schema


def test_export_express_rules_clean(exported):
    _, result, f = exported
    assert schema_problems(f, express_rules=True) == []


def test_export_spatial_structure_and_units(exported):
    _, _, f = exported
    (storey,) = f.by_type("IfcBuildingStorey")
    assert storey.Name == "EG"
    assert ifcopenshell.util.element.get_aggregate(storey).is_a("IfcBuilding")
    for el in f.by_type("IfcWall") + f.by_type("IfcDoor") + f.by_type("IfcWindow"):
        assert ifcopenshell.util.element.get_container(el) == storey
    for sp in f.by_type("IfcSpace"):
        assert ifcopenshell.util.element.get_aggregate(sp) == storey
    length = [u for u in f.by_type("IfcSIUnit") if u.UnitType == "LENGTHUNIT"]
    assert len(length) == 1 and length[0].Prefix is None and length[0].Name == "METRE"
    ids = {c.ContextIdentifier for c in f.by_type("IfcGeometricRepresentationSubContext")}
    assert {"Body", "Axis"} <= ids


def test_export_psets_and_spaces(exported):
    schema, _, f = exported
    external = {}
    for w in f.by_type("IfcWall"):
        psets = ifcopenshell.util.element.get_psets(w)
        external[w.Tag] = psets["Pset_WallCommon"]["IsExternal"]
    assert external == {
        "1": True,
        "2": True,
        "3": True,
        "4": True,
        "5": True,
        "6": True,
        "7": False,
    }
    names = {s.LongName: s for s in f.by_type("IfcSpace")}
    assert set(names) == {"Wohnen", "Küche"}
    qto = "BaseQuantities" if schema == "IFC2X3" else "Qto_SpaceBaseQuantities"
    q = ifcopenshell.util.element.get_psets(names["Wohnen"])[qto]
    assert q["NetFloorArea"] == pytest.approx((6 - INT / 2 - EXT / 2) * (8 - EXT))
    door = f.by_type("IfcDoor")[0]
    assert ifcopenshell.util.element.get_psets(door)["Pset_DoorCommon"]["IsExternal"] is False
    for win in f.by_type("IfcWindow"):
        assert ifcopenshell.util.element.get_psets(win)["Pset_WindowCommon"]["IsExternal"] is True


def _shapes(f, cls):
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    out = {}
    for el in f.by_type(cls):
        shape = ifcopenshell.geom.create_shape(settings, el)
        out[el] = (np.array(shape.geometry.verts).reshape(-1, 3), shape.geometry)
    return out


def test_wall_bodies_close_the_envelope(exported):
    """Union of wall footprints = outer rectangle minus the two rooms (closed corners)."""
    _, _, f = exported
    feet = {
        el.Tag: MultiPoint(v[:, :2].tolist()).convex_hull
        for el, (v, _) in _shapes(f, "IfcWall").items()
    }
    # Tiny buffer: float noise at butt joints must not open slivers between bodies.
    union = unary_union([p.buffer(1e-6, join_style="mitre") for p in feet.values()])
    assert union.geom_type == "Polygon"
    outer = (10 + EXT) * (8 + EXT)
    rooms = sum(r.area_m2 for r in two_room_house().rooms)
    assert union.area == pytest.approx(outer - rooms, abs=2e-3)
    assert len(union.interiors) == 2
    minx, miny, maxx, maxy = union.bounds
    assert (minx, miny, maxx, maxy) == pytest.approx(
        (-EXT / 2, -EXT / 2, 10 + EXT / 2, 8 + EXT / 2), abs=1e-5
    )
    # Butt joints: exterior corners are filled once, not twice.
    exterior = [p for tag, p in feet.items() if tag != "7"]
    assert sum(p.area for p in exterior) == pytest.approx(unary_union(exterior).area, abs=1e-3)


def test_openings_cut_walls_and_fillings_sit_in_them(exported):
    _, _, f = exported
    shapes = _shapes(f, "IfcWall")
    by_tag = {el.Tag: (v, g) for el, (v, g) in shapes.items()}
    # Interior wall 7: 8 m centreline, 2.60 m high, door 0.885 x 2.01 cut out.
    vol = ifcopenshell.util.shape.get_volume(by_tag["7"][1])
    assert vol == pytest.approx(INT * 2.6 * (8 - 0.885 * 2.01 / 2.6), rel=1e-3)
    # Window 3 on wall 3 (x = 10): inside the wall thickness and at sill 0.90.
    win = next(w for w in f.by_type("IfcWindow") if w.Tag == "3")
    v = ifcopenshell.geom.create_shape(_world(), win).geometry.verts
    v = np.array(v).reshape(-1, 3)
    assert v[:, 0].min() >= 10 - EXT / 2 - 1e-6 and v[:, 0].max() <= 10 + EXT / 2 + 1e-6
    assert v[:, 2].min() == pytest.approx(0.90, abs=1e-6)
    assert v[:, 1].min() == pytest.approx(3.0, abs=1e-6)
    # Door 1 on wall 7 (x = 6): within the wall thickness (door leaf may stand 2 cm proud).
    door = f.by_type("IfcDoor")[0]
    v = np.array(ifcopenshell.geom.create_shape(_world(), door).geometry.verts).reshape(-1, 3)
    assert v[:, 0].min() >= 6 - INT / 2 - 0.02 and v[:, 0].max() <= 6 + INT / 2 + 0.02
    assert v[:, 1].min() == pytest.approx(3.0, abs=1e-6)
    assert v[:, 1].max() == pytest.approx(3.885, abs=1e-6)
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-6)


def _world():
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    return s


def test_export_without_slab_and_messy_model(tmp_path):
    """Skewed wall, free end, unknown kinds: still writes, problems become warnings."""
    storey = Storey(
        walls=[
            _w(1, (0, 0), (4, 0)),
            _w(2, (4, 0), (6, 2)),  # 45 degree corner
            _w(3, (6, 2), (6, 5)),  # free end at (6, 5)
        ],
        openings=[],
    )
    model = BuildingModel(name="Schief", storeys=[storey])
    for schema in SCHEMAS:
        result = write_ifc(model, tmp_path / f"m_{schema}.ifc", schema=schema)
        assert result.reopened_ok
        assert result.warnings == []
        assert "IfcSlab" not in result.entity_counts


def test_unsupported_schema(tmp_path):
    with pytest.raises(ValueError):
        write_ifc(BuildingModel(), tmp_path / "x.ifc", schema="IFC4X3")  # type: ignore[arg-type]


def test_pipeline_signature_compatible(tmp_path, house):
    """The pipeline calls ``build_model(graph, storey_height_m=…, name=…)`` and
    ``write_ifc(model, path, schema=…)``."""
    model = build_model(house, storey_height_m=2.75, name="plan")
    res = write_ifc(model, tmp_path / "out.ifc", schema="IFC4")
    assert res.reopened_ok and math.isclose(model.storeys[0].height_m, 2.75)
