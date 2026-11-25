"""Vector-plan renderer / GT converter, scale-free panel, family pooling in rule 4."""

from __future__ import annotations

import hashlib

import numpy as np
from shapely.geometry import Polygon, box

from kelpod.contracts import Node, OpeningKind, PlanGraph, Room, Wall, WallKind
from kelpod.evaluation.panel import METRICS, SCALEFREE_NA, scalefree_panel, similarity_align
from kelpod.evaluation.stats import Delta, decide
from kelpod.evaluation.vectorplans import VectorPlan, decompose_rectilinear, render_plan


def _house() -> VectorPlan:
    """8 x 6 m footprint, 0.3 m outer walls, one 0.12 m partition at x = 4, a door and a window."""
    t = 0.3
    walls = [
        box(0, 0, 8, t), box(0, 6 - t, 8, 6), box(0, 0, t, 6), box(8 - t, 0, 8, 6),
        box(4 - 0.06, t, 4 + 0.06, 6 - t),
    ]  # fmt: skip
    return VectorPlan(
        walls=walls,
        doors=[box(3.94 - 0.0, 2.0, 4.06, 2.9)],
        windows=[box(1.5, 0.0, 2.7, t)],
        rooms=[(box(t, t, 3.94, 6 - t), "Wohnen"), (box(4.06, t, 8 - t, 6 - t), "Bad")],
    )


def test_render_plan_writes_gt_in_metres_and_is_deterministic():
    a = render_plan(_house(), seed=7, degrade=0.3)
    b = render_plan(_house(), seed=7, degrade=0.3)
    assert a is not None and b is not None
    assert hashlib.sha256(a.image.tobytes()).digest() == hashlib.sha256(b.image.tobytes()).digest()
    g = a.graph
    assert g.scale_source == "ground_truth"
    assert len(g.rooms) == 2 and {r.name for r in g.rooms} == {"Wohnen", "Bad"}
    assert abs(sum(r.area_m2 for r in g.rooms) - (3.64 * 5.4 + 3.64 * 5.4)) < 0.5
    kinds = {o.kind for o in g.openings}
    assert kinds == {OpeningKind.DOOR, OpeningKind.WINDOW}
    assert all(o.sill_m is None for o in g.openings if o.kind == OpeningKind.WINDOW)
    assert WallKind.EXTERIOR in {w.kind for w in g.walls}
    assert WallKind.DRYWALL in {w.kind for w in g.walls}  # the 12 cm partition
    assert g.report is not None and g.report.node_closure == 1.0
    assert a.image.shape == a.clean.shape and a.image.dtype == np.uint8


def test_unusable_plans_are_refused():
    tiny = VectorPlan(walls=[box(0, 0, 1, 0.2)], rooms=[(box(0, 0, 1, 0.2), "X")])
    assert render_plan(tiny, seed=1, degrade=0.0) is None


def test_decompose_l_shape_into_two_strips():
    ell = Polygon([(0, 0), (4, 0), (4, 0.3), (0.3, 0.3), (0.3, 3), (0, 3)])
    rects = decompose_rectilinear(ell)
    assert rects is not None and len(rects) == 2
    assert sorted(round(r.t, 2) for r in rects) == [0.3, 0.3]
    assert decompose_rectilinear(Polygon([(0, 0), (4, 0), (0, 3)])) is None  # diagonal edge


def _graph(scale: float, shift: float, rooms: list[tuple[float, float, float, float]]):
    nodes = [Node(id=i, x=scale * x + shift, y=scale * y + shift) for i, (x, y) in
             enumerate([(0, 0), (6, 0), (6, 4), (0, 4)])]  # fmt: skip
    walls = [Wall(id=i, start=i, end=(i + 1) % 4, thickness_m=0.2, kind=WallKind.EXTERIOR)
             for i in range(4)]  # fmt: skip
    rs = []
    for k, (x0, y0, x1, y1) in enumerate(rooms):
        poly = [(scale * x + shift, scale * y + shift) for x, y in
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]  # fmt: skip
        rs.append(Room(id=k, polygon=poly, name=None, area_m2=scale**2 * (x1 - x0) * (y1 - y0)))
    return PlanGraph(nodes=nodes, walls=walls, rooms=rs, m_per_px=0.01,
                     scale_source="default", image_height_px=100)  # fmt: skip


def test_scalefree_panel_ignores_scale_and_offset_and_marks_na():
    rooms = [(0, 0, 3, 4), (3, 0, 6, 4)]
    gt = _graph(1.0, 0.0, rooms)
    pred = _graph(2.7, 5.0, rooms)  # wrong scale and offset: same plan
    out = scalefree_panel(pred, gt, runtime_s=1.0, ifc_ok=True)
    assert set(out) == set(METRICS)
    assert out["room_iou"] > 0.99 and out["correct_closed"] == 1.0
    assert out["room_count_error"] == 0.0
    assert all(out[k] is None for k in SCALEFREE_NA)
    al = similarity_align(pred, gt)
    assert al is not None and abs(al.rooms[0].polygon[0][0]) < 0.05
    # a merged room (open plan that the pipeline cannot separate) is not correct
    merged = scalefree_panel(_graph(1.0, 0.0, [(0, 0, 6, 4)]), gt, ifc_ok=True)
    assert merged["correct_closed"] == 0.0 and merged["room_count_error"] == 0.5
    crash = scalefree_panel(None, gt)
    assert crash["crash"] == 1.0 and crash["correct_closed"] == 0.0
    assert crash["wall_f1"] is None


def _d(delta: float) -> Delta:
    return Delta("correct_closed", 100, 0.5, 0.5 + delta, delta, delta - 0.02, delta + 0.02)


def test_rule_4_counts_a_family_once_on_its_pooled_delta():
    metrics = {"correct_closed": (+1, "rate")}
    split_deltas = {
        "primary": {"correct_closed": _d(0.05)},
        "test-ood-swiss": {"correct_closed": _d(0.10)},  # wins on one split ...
        "test-ood-resplan": {"correct_closed": _d(-0.02)},
        "test-synth": {"correct_closed": _d(-0.001)},
        "fresh": {"correct_closed": _d(0.04)},
    }
    fams = {"test-ood-swiss": "test-ood", "test-ood-resplan": "test-ood",
            "test-synth": "test-synth", "fresh": "fresh"}  # fmt: skip
    kinds = {k: metrics for k in split_deltas}
    kw = dict(primary="primary", p95_cand={}, families=fams)
    old = decide(split_deltas, kinds, **kw)
    new = decide(
        split_deltas, kinds, family_deltas={
            "test-ood": {"correct_closed": _d(-0.001)},  # ... but the pooled family does not gain
            "test-synth": {"correct_closed": _d(-0.001)},
            "fresh": {"correct_closed": _d(0.04)},
        }, **kw,
    )  # fmt: skip
    r4_old = next(r for r in old.rules if r.rule.startswith("4"))
    r4_new = next(r for r in new.rules if r.rule.startswith("4"))
    assert r4_old.passed  # per-split counting let test-ood win on its best split
    assert not r4_new.passed and "1 of 3" in r4_new.detail
