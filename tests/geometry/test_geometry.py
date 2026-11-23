from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from geo_fixtures import (  # noqa: E402
    EXT,
    INT,
    M_PER_PX,
    Drawer,
    face_dims,
    ocr,
    page,
    seg,
    sill,
    stamp,
    two_room_house,
)
from shapely.geometry import Polygon  # noqa: E402

from kelpod import geometry  # noqa: E402
from kelpod.contracts import (  # noqa: E402
    DimensionLabel,
    Node,
    OpeningKind,
    PlanGraph,
    Room,
    Wall,
    WallKind,
    WallSegment,
)

ROOM_AREA = (4 - EXT / 2 - INT / 2) * (6 - EXT)  # 3.6975 * 5.635


def wall_len(g: PlanGraph, w: Wall) -> float:
    a, b = g.node(w.start), g.node(w.end)
    return math.hypot(b.x - a.x, b.y - a.y)


def degrees(g: PlanGraph) -> dict[int, int]:
    deg = {n.id: 0 for n in g.nodes}
    for w in g.walls:
        deg[w.start] += 1
        deg[w.end] += 1
    return deg


def xs_of_vertical_walls(g: PlanGraph) -> list[float]:
    out = set()
    for w in g.walls:
        a, b = g.node(w.start), g.node(w.end)
        if abs(a.x - b.x) < 1e-9:
            out.add(round(a.x, 6))
    return sorted(out)


def ys_of_horizontal_walls(g: PlanGraph) -> list[float]:
    out = set()
    for w in g.walls:
        a, b = g.node(w.start), g.node(w.end)
        if abs(a.y - b.y) < 1e-9:
            out.add(round(a.y, 6))
    return sorted(out)


# ------------------------------------------------------------------------------------------------
# closure and topology
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shrink", [0.15, -0.1])
def test_noisy_rectangle_house_closes_exactly(seed, shrink):
    """Noise of ~2 px, wall ends stopping short or overshooting, no OCR at all."""
    dr = Drawer(noise_px=2.0, seed=seed)
    walls, rooms = two_room_house(dr, shrink=shrink)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    r = g.report
    assert r.is_closed, r.issues
    assert (r.n_nodes, r.n_walls, r.n_rooms, r.n_components) == (6, 7, 2, 1)
    assert r.node_closure == 1.0
    for room in g.rooms:
        assert room.area_m2 == pytest.approx(ROOM_AREA, rel=0.03)
    # every wall is exactly axis-parallel
    for w in g.walls:
        a, b = g.node(w.start), g.node(w.end)
        assert a.x == b.x or a.y == b.y
    assert {w.thickness_m for w in g.walls} == {EXT, INT}


def test_rooms_are_ccw_inner_faces():
    dr = Drawer()
    walls, rooms = two_room_house(dr)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    for room in g.rooms:
        poly = Polygon(room.polygon)
        assert poly.exterior.is_ccw
        assert len(room.polygon) == 4
        assert room.area_m2 == pytest.approx(ROOM_AREA, abs=1e-3)


def test_t_junction_split():
    """A stem stopping 10 cm short of a bar splits the bar in two."""
    dr = Drawer()
    walls = [
        dr.wall((0, 0), (6, 0), EXT),
        dr.wall((3, 0.1), (3, 3), INT, WallKind.INTERIOR),
    ]
    g = geometry.run(page(), seg(walls), ocr())
    assert len(g.walls) == 3
    deg = degrees(g)
    assert sorted(deg.values()) == [1, 1, 1, 3]
    hub = next(n for n in g.nodes if deg[n.id] == 3)
    assert (hub.x, hub.y) == pytest.approx((2.0 + 3.0, 2.0), abs=0.02)
    assert set(g.report.dangling_nodes) == {n.id for n in g.nodes if deg[n.id] == 1}


def test_crossing_walls_are_split():
    dr = Drawer()
    walls = [dr.wall((0, 2), (6, 2), INT), dr.wall((3, 0), (3, 4), INT)]
    g = geometry.run(page(), seg(walls), ocr())
    assert len(g.walls) == 4
    assert max(degrees(g).values()) == 4


def test_collinear_pieces_are_merged():
    dr = Drawer(noise_px=1.0, seed=3)
    walls, rooms = two_room_house(dr)
    # cut the bottom wall into three overlapping/touching pieces
    walls[0:1] = [
        dr.wall((0, 0), (2.5, 0), EXT),
        dr.wall((2.45, 0), (5.0, 0), EXT),
        dr.wall((5.03, 0), (8, 0), EXT),
    ]
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert g.report.is_closed, g.report.issues
    assert len(g.walls) == 7


# ------------------------------------------------------------------------------------------------
# gaps and openings
# ------------------------------------------------------------------------------------------------


def _house_with_gap(dr, lo, hi):
    walls, rooms = two_room_house(dr)
    walls[0:1] = [dr.wall((0, 0), (lo, 0), EXT), dr.wall((hi, 0), (8, 0), EXT)]
    return walls, rooms


def test_door_gap_is_bridged_and_door_attached():
    dr = Drawer(noise_px=1.0, seed=7)
    walls, rooms = _house_with_gap(dr, 1.0, 2.01)
    door = dr.opening(OpeningKind.DOOR, (1.0, 0), (2.01, 0))
    win = dr.opening(OpeningKind.WINDOW, (5.0, 6), (6.5, 6))
    g = geometry.run(page(), seg(walls, [door, win], rooms), ocr())
    assert g.report.is_closed, g.report.issues
    kinds = {o.kind: o for o in g.openings}
    assert set(kinds) == {OpeningKind.DOOR, OpeningKind.WINDOW}
    d = kinds[OpeningKind.DOOR]
    wall = next(w for w in g.walls if w.id == d.wall_id)
    a, b = g.node(wall.start), g.node(wall.end)
    assert a.y == b.y == pytest.approx(2.0, abs=0.02)  # bottom wall
    sign = 1.0 if b.x > a.x else -1.0
    edges = sorted((a.x + sign * d.offset_m, a.x + sign * (d.offset_m + d.width_m)))
    assert edges == pytest.approx([2.0 + 1.0, 2.0 + 2.01], abs=0.03)
    assert (d.sill_m, d.height_m) == (0.0, 2.01)
    w = kinds[OpeningKind.WINDOW]
    assert (w.sill_m, w.height_m) == (0.90, 1.385)
    assert w.width_m == pytest.approx(1.5, abs=0.03)


def test_gap_without_opening_bridged_as_unknown_opening():
    dr = Drawer(seed=2)
    walls, rooms = _house_with_gap(dr, 1.0, 1.9)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert g.report.is_closed, g.report.issues
    assert [o.kind for o in g.openings] == [OpeningKind.UNKNOWN]
    assert g.openings[0].width_m == pytest.approx(0.9, abs=0.03)
    assert any("unknown type" in i for i in g.report.issues)


def test_corridor_gap_between_cross_walls_is_not_bridged():
    """Two walls on one line that each end at a cross wall: the gap is a corridor, keep it open."""
    dr = Drawer()
    walls = [
        dr.wall((0, 3), (3, 3), INT),
        dr.wall((4.2, 3), (8, 3), INT),
        dr.wall((3, 0), (3, 6), INT),
        dr.wall((4.2, 0), (4.2, 6), INT),
    ]
    g = geometry.run(page(), seg(walls), ocr())
    assert not g.openings
    ys = [g.node(w.start).y for w in g.walls if g.node(w.start).y == g.node(w.end).y]
    # no horizontal wall spans the corridor between x = 3 and x = 4.2
    for w in g.walls:
        a, b = g.node(w.start), g.node(w.end)
        if a.y == b.y:
            lo, hi = sorted((a.x - 2, b.x - 2))
            assert not (lo < 3.6 < hi)
    assert ys


# ------------------------------------------------------------------------------------------------
# scale and dimension fitting
# ------------------------------------------------------------------------------------------------


def test_scale_from_dimensions_overrides_plan_scale():
    """The scan was printed at 95 %: px are bigger than 1:100 at 200 dpi suggests."""
    true_m = M_PER_PX / 0.95
    dr = Drawer(m_per_px=true_m, noise_px=1.0, seed=4)
    walls, rooms = two_room_house(dr)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr(face_dims(dr)))
    assert g.scale_source == "dimensions"
    assert g.m_per_px == pytest.approx(true_m, rel=0.005)
    for room in g.rooms:
        assert room.area_m2 == pytest.approx(ROOM_AREA, abs=0.02)


def test_scale_falls_back_to_plan_scale_and_default():
    dr = Drawer()
    walls, rooms = two_room_house(dr)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert g.scale_source == "plan_scale" and g.m_per_px == pytest.approx(M_PER_PX)
    g = geometry.run(page(plan_scale=None), seg(walls, rooms=rooms), ocr())
    assert g.scale_source == "default" and g.m_per_px == pytest.approx(M_PER_PX)
    assert any("assumed 1:100" in i for i in g.report.issues)


def test_three_percent_length_error_fixed_to_one_cm():
    """The left room was drawn 3 % too wide; the written dimensions are right."""
    dr = Drawer(noise_px=1.0, seed=5)
    stretch = 1.03

    def sx(x):  # drawn x of a true x
        return x * stretch if x <= 4 else x + 4 * (stretch - 1)

    dr.xform = lambda x, y: (sx(x), y)
    walls, rooms = two_room_house(dr)
    g0 = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    xs0 = xs_of_vertical_walls(g0)
    assert xs0[1] - xs0[0] == pytest.approx(4.12, abs=0.04)  # as drawn: wrong by ~12 cm

    g = geometry.run(page(), seg(walls, rooms=rooms), ocr(face_dims(dr)))
    xs, ys = xs_of_vertical_walls(g), ys_of_horizontal_walls(g)
    assert xs[1] - xs[0] == pytest.approx(4.0, abs=0.01)
    assert xs[2] - xs[1] == pytest.approx(4.0, abs=0.01)
    assert ys[1] - ys[0] == pytest.approx(6.0, abs=0.01)
    assert g.report.dimension_residual_m < 0.01
    assert g.report.is_closed
    for room in g.rooms:
        assert room.area_m2 == pytest.approx(ROOM_AREA, abs=0.06)


def test_centreline_dimensions_are_recognised():
    dr = Drawer(seed=6)
    dr.xform = lambda x, y: (x * 1.02, y)
    walls, rooms = two_room_house(dr)
    dims = [
        dr.dim(4.0, (0, -1), (4, -1)),
        dr.dim(4.0, (4, -1), (8, -1)),
        dr.dim(6.0, (-1, 0), (-1, 6)),
    ]
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr(dims))
    xs = xs_of_vertical_walls(g)
    assert xs[1] - xs[0] == pytest.approx(4.0, abs=0.01)
    assert xs[2] - xs[1] == pytest.approx(4.0, abs=0.01)
    assert any("centrelines" in i for i in g.report.issues)


def test_misread_dimension_is_ignored():
    dr = Drawer(seed=8)
    walls, rooms = two_room_house(dr)
    dims = face_dims(dr)
    dims[1] = dims[1].model_copy(update={"value_m": 3.75})  # OCR read 3,75 instead of 3,6975
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr(dims))
    xs = xs_of_vertical_walls(g)
    assert xs[1] - xs[0] == pytest.approx(4.0, abs=0.01)
    assert g.report.dimension_residual_m < 0.01
    assert any("contradict" in i for i in g.report.issues)


# ------------------------------------------------------------------------------------------------
# stamps, BRH, odd angles, skew
# ------------------------------------------------------------------------------------------------


def test_room_names_areas_and_brh():
    dr = Drawer(seed=9)
    walls, rooms = two_room_house(dr)
    win = dr.opening(OpeningKind.WINDOW, (1.0, 6), (2.5, 6))
    stamps = [stamp(dr, "Wohnen", 20.84, (2, 3)), stamp(dr, "Küche", 30.0, (6, 3))]
    g = geometry.run(
        page(), seg(walls, [win], rooms), ocr(stamps=stamps, sills=[sill(dr, 0.6, (1.7, 6.6))])
    )
    names = {r.name: r for r in g.rooms}
    assert set(names) == {"Wohnen", "Küche"}
    assert names["Wohnen"].stamp_area_m2 == 20.84
    assert any("Küche" in i and "differs" in i for i in g.report.issues)
    assert not any("Wohnen" in i and "differs" in i for i in g.report.issues)
    (o,) = g.openings
    assert o.sill_m == 0.6
    assert o.height_m == pytest.approx(0.90 + 1.385 - 0.6)


def test_diagonal_corner_closes():
    dr = Drawer(noise_px=1.0, seed=10)
    walls = [
        dr.wall((0, 0), (8, 0), EXT),
        dr.wall((8, 0), (8, 4), EXT),
        dr.wall((8, 4), (6, 6), EXT),
        dr.wall((6, 6), (0, 6), EXT),
        dr.wall((0, 6), (0, 0), EXT),
    ]
    g = geometry.run(page(), seg(walls), ocr())
    assert g.report.is_closed, g.report.issues
    assert len(g.rooms) == 1
    (diag,) = [
        w
        for w in g.walls
        if abs(g.node(w.start).x - g.node(w.end).x) > 1e-6
        and abs(g.node(w.start).y - g.node(w.end).y) > 1e-6
    ]
    a, b = g.node(diag.start), g.node(diag.end)
    assert abs(abs(b.x - a.x) - abs(b.y - a.y)) < 0.02  # 45 degrees


def test_residual_skew_is_removed():
    dr = Drawer(rot_deg=1.5, noise_px=1.0, seed=11)
    walls, rooms = two_room_house(dr)
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    assert g.report.is_closed
    for w in g.walls:
        a, b = g.node(w.start), g.node(w.end)
        assert a.x == b.x or a.y == b.y
    for room in g.rooms:
        assert room.area_m2 == pytest.approx(ROOM_AREA, rel=0.02)


# ------------------------------------------------------------------------------------------------
# robustness
# ------------------------------------------------------------------------------------------------


def test_empty_input_returns_graph_with_issues():
    g = geometry.run(page(), seg([]), ocr())
    assert isinstance(g, PlanGraph)
    assert g.report is not None and not g.walls
    assert any("No walls" in i for i in g.report.issues)
    g = geometry.run(page(), seg([]), None)
    assert g.report is not None


def test_spurious_pieces_and_garbage_ocr():
    dr = Drawer(noise_px=2.0, seed=12)
    walls, rooms = two_room_house(dr, shrink=0.1)
    junk = [
        dr.wall((10, -1), (10.4, -1), 0.1),  # isolated scrap
        dr.wall((-1.5, 1), (-1.5, 5), 0.06),  # a dimension line taken for a wall
        dr.wall((1.0, 3.0), (1.6, 3.4), 0.1),  # text stroke inside a room
        dr.wall((5.5, 0), (5.5, 0.25), INT),  # short stub on the bottom wall
    ]
    n = len(walls)
    junk += [
        WallSegment(id=n + 10, p0=(100.0, 100.0), p1=(100.0, 100.0), thickness_px=20),
        WallSegment(id=n + 11, p0=(float("nan"), 1.0), p1=(5.0, 5.0), thickness_px=20),
        WallSegment(id=n + 12, p0=(1.0, 1.0), p1=(50.0, 1.0), thickness_px=-3),
    ]
    dims = face_dims(dr) + [
        DimensionLabel(value_m=123.0, text="123", anchor=(5, 5), line=((5, 5), (60, 5))),
        DimensionLabel(value_m=float("nan"), text="?", anchor=(5, 5), line=((5, 5), (60, 5))),
        DimensionLabel(value_m=2.0, text="2,00", anchor=(5, 5), line=None),
    ]
    g = geometry.run(page(), seg(walls + junk, rooms=rooms), ocr(dims))
    r = g.report
    assert r.n_components == 1
    assert r.n_rooms == 2 and r.room_closure == 1.0
    assert not r.dangling_nodes, r.issues
    assert r.is_closed


def test_open_house_reports_instead_of_failing():
    dr = Drawer(seed=13)
    walls, rooms = two_room_house(dr)
    del walls[1]  # right exterior wall missing
    g = geometry.run(page(), seg(walls, rooms=rooms), ocr())
    r = g.report
    assert not r.is_closed
    assert r.dangling_nodes
    assert r.room_closure == 0.5
    assert any("did not become closed" in i for i in r.issues)


# ------------------------------------------------------------------------------------------------
# check()
# ------------------------------------------------------------------------------------------------


def _graph(nodes, walls, rooms=()):
    return PlanGraph(
        nodes=[Node(id=i, x=x, y=y) for i, (x, y) in enumerate(nodes)],
        walls=[Wall(id=i, start=a, end=b, thickness_m=0.24) for i, (a, b) in enumerate(walls)],
        rooms=list(rooms),
        m_per_px=0.0127,
        scale_source="plan_scale",
        image_height_px=100,
    )


def test_check_closed_square():
    g = _graph(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        [Room(id=0, polygon=[(0.1, 0.1), (3.9, 0.1), (3.9, 3.9), (0.1, 3.9)], area_m2=14.44)],
    )
    r = geometry.check(g)
    assert r.is_closed and r.node_closure == 1.0 and r.n_components == 1 and not r.issues


def test_check_flags_dangling_walls_and_second_component():
    g = _graph(
        [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (10, 0), (12, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (5, 6)],
        [Room(id=0, polygon=[(0, 0), (0, 4), (4, 4), (4, 0)], area_m2=16.0, stamp_area_m2=12.0)],
    )
    r = geometry.check(g)
    assert r.n_components == 2
    assert set(r.dangling_nodes) == {4, 5, 6}
    assert r.node_closure == pytest.approx(4 / 7)
    assert r.room_closure == 0.0  # clockwise polygon
    assert not r.is_closed
    text = " ".join(r.issues)
    assert "2 separate groups" in text
    assert "do not meet another wall" in text
    assert "counter-clockwise" in text
    assert "differs from the stamp" in text


def test_check_empty_graph():
    r = geometry.check(_graph([], []))
    assert r.n_components == 0 and r.node_closure == 0.0 and not r.is_closed


def test_runtime_is_small():
    import time

    dr = Drawer(noise_px=2.0, seed=14)
    walls = []
    for i in range(6):  # a 6 x 6 grid of rooms, 84 walls in pieces
        walls.append(dr.wall((0, i * 3.0), (15, i * 3.0), INT))
        walls.append(dr.wall((i * 3.0, 0), (i * 3.0, 15), INT))
    t0 = time.perf_counter()
    g = geometry.run(page(h=2400, w=2400), seg(walls), ocr())
    assert time.perf_counter() - t0 < 3.0
    assert len(g.rooms) == 25 and g.report.is_closed
    assert np.isclose(sum(r.area_m2 for r in g.rooms), 25 * (3 - INT) ** 2, rtol=0.02)


# ------------------------------------------------------------------------------------------------
# synthetic plans (kelpod.evaluation.synth ground truth -> simulated segment/ocr output)
# ------------------------------------------------------------------------------------------------


def _synth(seed):
    synth = pytest.importorskip("kelpod.evaluation.synth")
    try:
        return synth.generate(seed, degrade=0.0, dpi=200)
    except Exception as exc:  # the generator is built in parallel; do not fail on its bugs
        pytest.skip(f"synth generator failed: {exc}")


def _area_errors(gt: PlanGraph, pred: PlanGraph) -> list[float]:
    from shapely.affinity import translate

    def lo(ps):
        return np.min([np.array(p.exterior.coords).min(0) for p in ps], 0)

    gp = [Polygon(r.polygon) for r in gt.rooms]
    pp = [Polygon(r.polygon) for r in pred.rooms]
    d = lo(gp) - lo(pp)
    pp = [translate(p, *d) for p in pp]
    out = []
    for a in gp:
        b = max(pp, key=lambda q: a.intersection(q).area)
        assert a.intersection(b).area / a.union(b).area > 0.9
        out.append(abs(b.area - a.area) / a.area)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_synth_plans_close_and_match_ground_truth(seed):
    from geo_fixtures import inputs_from_synth

    plan = _synth(seed)
    page_, seg_, ocr_ = inputs_from_synth(plan, np.random.default_rng(seed))
    g = geometry.run(page_, seg_, ocr_)
    assert g.report.is_closed, g.report.issues
    assert len(g.rooms) == len(plan.graph.rooms)
    assert sorted(o.kind for o in g.openings) == sorted(o.kind for o in plan.graph.openings)
    assert g.scale_source == "dimensions"
    assert max(_area_errors(plan.graph, g)) < 0.02


@pytest.mark.parametrize("seed", range(3))
def test_synth_distorted_drawing_fixed_by_dimensions(seed):
    """The left/top half of the drawing is 3 % too large; the labels are right."""
    from geo_fixtures import inputs_from_synth

    plan = _synth(seed)
    h, w = plan.image.shape

    def distort(q):
        x, y = q
        x = x * 1.03 if x < w / 2 else x + w / 2 * 0.03
        y = y * 1.03 if y < h / 2 else y + h / 2 * 0.03
        return (x, y)

    rng = np.random.default_rng(seed)
    page_, seg_, ocr_ = inputs_from_synth(plan, rng, distort=distort)
    g = geometry.run(page_, seg_, ocr_)
    assert g.report.is_closed
    assert g.report.dimension_residual_m < 0.01
    assert np.mean(_area_errors(plan.graph, g)) < 0.01
    no_dims = geometry.run(page_, seg_, ocr_.model_copy(update={"dimensions": []}))
    assert np.mean(_area_errors(plan.graph, no_dims)) > 0.02  # the distortion is real
