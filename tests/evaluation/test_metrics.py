from __future__ import annotations

import pytest

from kelpod.contracts import Node, OpeningKind, PlanGraph, Room, Wall
from kelpod.evaluation.metrics import closure_metrics, compare_to_ground_truth
from kelpod.evaluation.synth import generate


@pytest.fixture(scope="module")
def gt():
    return generate(4, degrade=0.0).graph


def _copy(g: PlanGraph, **update) -> PlanGraph:
    return g.model_copy(deep=True, update=update)


def test_gt_vs_gt_is_perfect(gt):
    c = compare_to_ground_truth(gt, gt)
    assert c.room_iou_mean == pytest.approx(1.0)
    assert c.room_precision == c.room_recall == 1.0
    assert c.wall_precision == c.wall_recall == c.wall_f1 == pytest.approx(1.0)
    assert c.opening_recall["door"] == c.opening_recall["window"] == c.opening_recall["any"] == 1
    assert c.opening_precision == 1.0
    assert c.room_area_error == pytest.approx(0.0) and c.total_area_error == pytest.approx(0.0)
    assert c.room_name_accuracy == 1.0


def test_translation_is_aligned(gt):
    rooms = [
        r.model_copy(update={"polygon": [(x + 3.0, y - 1.5) for x, y in r.polygon]})
        for r in gt.rooms
    ]
    moved = _copy(
        gt, nodes=[Node(id=n.id, x=n.x + 3.0, y=n.y - 1.5) for n in gt.nodes], rooms=rooms
    )
    c = compare_to_ground_truth(moved, gt)
    assert c.offset_m == pytest.approx((-3.0, 1.5))
    assert c.wall_f1 == pytest.approx(1.0) and c.room_iou_mean == pytest.approx(1.0)
    assert compare_to_ground_truth(moved, gt, align="none").wall_f1 < 0.5


def test_perturbed_is_worse(gt):
    # stretch x by 5 %, drop one room and all windows, rename rooms
    sx = min(n.x for n in gt.nodes)
    pert = _copy(
        gt,
        nodes=[Node(id=n.id, x=sx + (n.x - sx) * 1.05, y=n.y) for n in gt.nodes],
        rooms=[
            r.model_copy(
                update={
                    "polygon": [(sx + (x - sx) * 1.05, y) for x, y in r.polygon],
                    "name": "Raum",
                }
            )
            for r in gt.rooms[1:]
        ],
        openings=[o for o in gt.openings if o.kind == OpeningKind.DOOR],
    )
    c = compare_to_ground_truth(pert, gt)
    perfect = compare_to_ground_truth(gt, gt)
    assert c.wall_f1 < perfect.wall_f1
    assert c.room_iou_mean < perfect.room_iou_mean
    assert c.room_recall < 1.0
    assert c.opening_recall["window"] == 0.0
    assert c.room_name_accuracy == 0.0
    assert c.total_area_error > 0.0


def test_empty_prediction(gt):
    empty = PlanGraph(
        m_per_px=gt.m_per_px, scale_source="default", image_height_px=gt.image_height_px
    )
    c = compare_to_ground_truth(empty, gt)
    assert c.wall_f1 == 0.0 and c.room_recall == 0.0 and c.room_iou_mean == 0.0
    assert c.opening_recall["any"] == 0.0
    r = closure_metrics(empty)
    assert not r.is_closed and r.node_closure == 0.0 and "no walls" in r.issues


def _square(with_spur: bool = False, split: bool = False) -> PlanGraph:
    pts = [(0, 0), (4, 0), (4, 3), (0, 3)]
    nodes = [Node(id=i, x=x, y=y) for i, (x, y) in enumerate(pts)]
    walls = [Wall(id=i, start=i, end=(i + 1) % 4, thickness_m=0.3) for i in range(4)]
    if with_spur:
        nodes.append(Node(id=4, x=6, y=0))
        walls.append(Wall(id=4, start=1, end=4, thickness_m=0.3))
    if split:
        nodes += [Node(id=5, x=10, y=0), Node(id=6, x=12, y=0)]
        walls.append(Wall(id=5, start=5, end=6, thickness_m=0.3))
    room = Room(
        id=0,
        polygon=[(0.15, 0.15), (3.85, 0.15), (3.85, 2.85), (0.15, 2.85)],
        area_m2=9.99,
        stamp_area_m2=12.0,
    )
    return PlanGraph(
        nodes=nodes,
        walls=walls,
        rooms=[room],
        m_per_px=0.01,
        scale_source="default",
        image_height_px=1000,
    )


def test_closure_metrics():
    r = closure_metrics(_square())
    assert r.is_closed and r.node_closure == 1.0 and r.n_components == 1
    assert any("stamp" in i for i in r.issues)  # 9.99 vs 12.0 m²
    r = closure_metrics(_square(with_spur=True))
    assert not r.is_closed and r.dangling_nodes == [4] and r.node_closure == pytest.approx(0.8)
    r = closure_metrics(_square(split=True))
    assert r.n_components == 2 and not r.is_closed
    bad = _square()
    bad.rooms.append(Room(id=1, polygon=[(0, 0), (1, 1), (1, 0), (0, 1)], area_m2=0))  # bow-tie
    r = closure_metrics(bad)
    assert r.room_closure == 0.5 and not r.is_closed
