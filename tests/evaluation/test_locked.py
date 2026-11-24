"""Panel metrics, paired bootstrap, promotion rule, manifests and leak checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kelpod.contracts import PlanGraph, WallKind
from kelpod.evaluation import panel, splits, stats
from kelpod.evaluation.synth import generate, save


@pytest.fixture(scope="module")
def gt() -> PlanGraph:
    return generate(4, degrade=0.0).graph


def _copy(g: PlanGraph) -> PlanGraph:
    return g.model_copy(deep=True)


# ---------------------------------------------------------------- panel


def test_gt_vs_gt_is_correct(gt):
    p = panel.plan_panel(_copy(gt), gt, runtime_s=1.0, ifc_ok=True)
    assert p["correct_closed"] == 1.0
    assert p["room_iou"] == pytest.approx(1.0)
    assert p["wall_f1"] == pytest.approx(1.0)
    assert p["thickness_class_acc"] == pytest.approx(1.0)
    assert p["edit_cost"] == 0.0
    assert p["door_precision"] == p["door_recall"] == 1.0
    assert p["window_recall"] == 1.0
    assert p["sill_error_m"] == pytest.approx(0.0)
    for k in ("ext_wall_length_error", "window_facade_error", "gross_volume_error"):
        assert p[k] == pytest.approx(0.0, abs=1e-9)
    assert p["crash"] == 0.0 and p["ifc_fail"] == 0.0


def test_translation_is_free(gt):
    g = _copy(gt)
    for n in g.nodes:
        n.x += 0.7
        n.y -= 0.4
    for r in g.rooms:
        r.polygon = [(x + 0.7, y - 0.4) for x, y in r.polygon]
    p = panel.plan_panel(g, gt, ifc_ok=True)
    assert p["correct_closed"] == 1.0 and p["edit_cost"] == 0.0


def test_one_big_box_is_closed_but_not_correct(gt):
    """The protocol's example: closure alone does not count."""
    xs = [n.x for n in gt.nodes]
    ys = [n.y for n in gt.nodes]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    from kelpod.contracts import Node, Room, Wall

    nodes = [
        Node(id=i, x=x, y=y) for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    ]
    walls = [Wall(id=i, start=i, end=(i + 1) % 4, thickness_m=0.365, kind=WallKind.EXTERIOR)
             for i in range(4)]  # fmt: skip
    t = 0.1825
    room = Room(id=0, polygon=[(x0 + t, y0 + t), (x1 - t, y0 + t), (x1 - t, y1 - t),
                               (x0 + t, y1 - t)], area_m2=1.0)  # fmt: skip
    box = PlanGraph(nodes=nodes, walls=walls, rooms=[room], m_per_px=gt.m_per_px,
                    scale_source="dimensions", image_height_px=gt.image_height_px)  # fmt: skip
    p = panel.plan_panel(box, gt, ifc_ok=True)
    assert p["closed"] == 1.0
    assert p["correct_closed"] == 0.0
    assert p["edit_cost"] > 0


def test_missing_room_breaks_headline(gt):
    g = _copy(gt)
    g.rooms = g.rooms[1:]
    p = panel.plan_panel(g, gt, ifc_ok=True)
    assert p["correct_closed"] == 0.0
    assert p["room_recall"] < 1.0


def test_wrong_thickness_class(gt):
    g = _copy(gt)
    for w in g.walls:
        w.kind = WallKind.INTERIOR
    p = panel.plan_panel(g, gt, ifc_ok=True)
    assert p["thickness_class_acc"] < 1.0
    assert p["ext_wall_length_error"] == pytest.approx(1.0)


def test_dropped_opening_costs_an_edit(gt):
    g = _copy(gt)
    g.openings = g.openings[1:]
    assert panel.edit_cost(g, gt) == 1.0


def test_crash_panel(gt):
    p = panel.plan_panel(None, gt)
    assert p["crash"] == 1.0 and p["correct_closed"] == 0.0
    assert p["edit_cost"] == panel.edit_cost(
        PlanGraph(m_per_px=1, scale_source="default", image_height_px=1), gt
    )
    assert p["runtime_s"] is None
    assert set(p) == set(panel.METRICS)


def test_panel_keys_match_metrics(gt):
    assert set(panel.plan_panel(_copy(gt), gt)) == set(panel.METRICS)
    assert set(panel.proxy_panel(_copy(gt))) == set(panel.PROXY_METRICS)
    assert set(panel.proxy_panel(None)) == set(panel.PROXY_METRICS)


# ---------------------------------------------------------------- stats


def test_bootstrap_identical_is_zero():
    x = list(np.random.default_rng(0).random(100))
    d = stats.paired_bootstrap(x, x, n_resamples=2000)
    assert d.delta == 0 and d.lo == 0 and d.hi == 0


def test_bootstrap_detects_shift_and_is_deterministic():
    rng = np.random.default_rng(1)
    b = list((rng.random(300) < 0.5).astype(float))
    c = [1.0 if v == 0 and rng.random() < 0.3 else v for v in b]
    d1 = stats.paired_bootstrap(c, b, seed=7)
    d2 = stats.paired_bootstrap(c, b, seed=7)
    assert d1 == d2
    assert d1.lo > 0 and d1.delta > 0.1


def test_bootstrap_drops_none_pairs():
    d = stats.paired_bootstrap([1.0, None, 0.0], [0.0, 1.0, None], n_resamples=100)
    assert d.n == 1


def test_relative_and_median():
    d = stats.compare_metric("runtime_s", -1, "runtime", [2.0] * 10, [1.0] * 10, n_resamples=100)
    assert d.stat == "median" and d.relative and d.delta == pytest.approx(1.0)
    assert stats.regression(d, -1, "runtime")
    d = stats.compare_metric("edit_cost", -1, "error", [1.0] * 10, [1.0] * 10, n_resamples=100)
    assert d.delta == 0 and not stats.regression(d, -1, "error")


def test_regression_tolerances():
    D = stats.Delta
    assert stats.regression(D("wall_f1", 9, 0.9, 0.85, -0.05, -0.06, -0.04), +1, "score")
    assert not stats.regression(D("wall_f1", 9, 0.9, 0.895, -0.005, -0.02, 0.0), +1, "score")
    assert stats.regression(D("crash", 9, 0, 0.1, 0.1, 0.05, 0.15), -1, "rate")
    assert not stats.regression(D("x", 9, 1, 1.03, 0.03, 0.01, 0.06, relative=True), -1, "error")
    assert stats.regression(D("x", 9, 1, 1.1, 0.1, 0.06, 0.14, relative=True), -1, "error")


def _deltas(cc: float, lo: float, iou: float = 0.0):
    D = stats.Delta
    return {
        "correct_closed": D("correct_closed", 100, 0.5, 0.5 + cc, cc, lo, cc + 0.05),
        "room_iou": D("room_iou", 100, 0.9, 0.9 + iou, iou, iou - 0.01, iou + 0.01),
        "edit_cost": D("edit_cost", 100, 5, 5, 0.0, -0.02, 0.02, relative=True),
    }


def _kinds():
    return {
        "correct_closed": (1, "rate"),
        "room_iou": (1, "score"),
        "edit_cost": (-1, "error"),
    }


@pytest.mark.parametrize(
    ("cc", "lo", "synth_gain", "expect"),
    [
        (0.05, 0.01, True, True),  # clear gain, generalises
        (0.02, 0.01, True, False),  # under +3 pp
        (0.05, -0.01, True, False),  # CI touches 0
        (0.05, 0.01, False, False),  # gain only on fresh, not on test-synth
    ],
)
def test_decide(cc, lo, synth_gain, expect):
    deltas = {
        "primary": _deltas(cc, lo),
        "fresh": _deltas(cc, lo),
        "test-synth": _deltas(0.02 if synth_gain else -0.005, -0.03),
    }
    kinds = {k: _kinds() for k in deltas}
    dec = stats.decide(
        deltas, kinds, primary="primary", p95_cand={"fresh": 10.0, "test-synth": 10.0},
        families={"fresh": "fresh", "test-synth": "test-synth"},
    )  # fmt: skip
    assert dec.passed is expect, dec.to_dict()


def test_decide_p95_limit():
    deltas = {"primary": _deltas(0.05, 0.01), "fresh": _deltas(0.05, 0.01),
              "test-synth": _deltas(0.05, 0.01)}  # fmt: skip
    dec = stats.decide(
        deltas, {k: _kinds() for k in deltas}, primary="primary",
        p95_cand={"fresh": 200.0}, families={"fresh": "fresh", "test-synth": "test-synth"},
    )  # fmt: skip
    assert not dec.passed


# ---------------------------------------------------------------- splits, manifests, leaks


def test_fresh_params_deterministic_and_reserved():
    a = splits.fresh_params("c" * 40, "m" * 40, 5)
    assert a == splits.fresh_params("c" * 40, "m" * 40, 5)
    assert a["seeds"] != splits.fresh_params("d" * 40, "m" * 40, 5)["seeds"]
    assert all(s >= splits.FRESH_SEED_BASE for s in a["seeds"])
    assert all(0.0 <= g <= 1.2 for g in a["degrades"])


def test_split_seed_ranges_apart():
    ranges = [
        range(c["first_seed"], c["first_seed"] + c["n"]) for c in splits.SYNTH_SPLITS.values()
    ]
    used = [range(1_000_000, 10**9), range(100_000, 100_050), range(0, 10_000)]
    for r in ranges:
        for u in used + [x for x in ranges if x is not r]:
            assert not set(r) & set(range(max(r.start, u.start), min(r.stop, u.stop)))


def test_phash_separates_plans_and_matches_rerenders():
    a, a2, b = generate(7, 0.0), generate(7, 0.6), generate(8, 0.0)
    ha, ha2, hb = splits.phash(a.image), splits.phash(a2.image), splits.phash(b.image)
    assert splits.hamming(ha, ha2) < splits.hamming(ha, hb)
    assert (
        splits.near_duplicates(
            {"x": ha2}, {"a": ha, "b": hb}, max_distance=splits.hamming(ha, ha2)
        )[0][1]
        == "a"
    )


def _mini_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("KELPOD_EVAL_DATA", str(tmp_path / "data"))
    items = splits.generate_synth([300000], [0.3], "heldout", tmp_path / "data" / "test-synth", 1)
    return {
        "split": "test-synth", "status": "frozen", "kind": "synth", "family": "test-synth",
        "data_dir": str(tmp_path / "data" / "test-synth"), "items": items,
        "input_file": "plan.pdf", "gt_file": "ground_truth.json",
    }  # fmt: skip


def test_verify_manifest_detects_tampering(tmp_path, monkeypatch):
    m = _mini_manifest(tmp_path, monkeypatch)
    assert splits.verify_manifest(m) == []
    gt_path = tmp_path / "data" / "test-synth" / "s300000" / "ground_truth.json"
    gt_path.write_text(gt_path.read_text() + " ")
    assert any("mismatch" in p for p in splits.verify_manifest(m))
    gt_path.unlink()
    assert any("missing" in p for p in splits.verify_manifest(m))


def test_training_overlap(tmp_path, monkeypatch):
    m = _mini_manifest(tmp_path, monkeypatch)
    swiss = {"split": "test-ood-swiss", "family": "test-ood", "status": "pending",
             "items": [], "heldout_ids": {"swiss_site_id": [11, 12]}}  # fmt: skip
    manifests = {"test-synth": m, "test-ood-swiss": swiss}
    pdf_sha = m["items"][0]["files"]["plan.pdf"]
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"files": [{"path": "a.png", "sha256": "0" * 64, "site_id": 13,
                                            "style": "train"}]}))  # fmt: skip
    assert splits.check_training_overlap([clean], manifests) == []
    for entry, word in [
        ({"path": "x.pdf", "sha256": pdf_sha}, "hash of test file"),
        ({"path": "y.png", "sha256": "1" * 64, "site_id": 12}, "held-out swiss_site_id"),
        ({"path": "z.png", "sha256": "2" * 64, "style": {"split": "heldout"}}, "held-out test"),
        ({"path": "w.png", "sha256": "3" * 64, "phash": m["items"][0]["phash"]}, "near duplicate"),
    ]:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"files": [entry]}))
        problems = splits.check_training_overlap([bad], manifests)
        assert problems and word in problems[0], (entry, problems)


def test_swiss_rule_matches_training_converter():
    import zlib

    for site in range(200):
        h = zlib.crc32(str(site).encode()) % 10
        assert splits.swiss_split_of(site) == ("test" if h == 0 else "val" if h == 1 else "train")


def test_save_roundtrip_for_runner(tmp_path):
    """The locked runner reads plan.pdf only; ground_truth.json carries the GT graph."""
    d = save(generate(2, 0.2, style="heldout"), tmp_path / "p")
    data = json.loads((d / "ground_truth.json").read_text())
    assert PlanGraph.model_validate(data["graph"]).scale_source == "ground_truth"
