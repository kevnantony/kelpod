from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from drawing import Op, Plan, W, house, page_from, render  # noqa: E402

from kelpod import segment  # noqa: E402
from kelpod.contracts import OpeningKind, RasterPage, Segmentation, WallKind  # noqa: E402
from kelpod.segment.walls import fill_narrow_holes  # noqa: E402


def _seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


@pytest.fixture(scope="module")
def house_result():
    img, gt = render(house())
    return segment.run(page_from(img)), gt


def test_registry_and_learned_stub(monkeypatch):
    assert {"classical", "learned"} <= set(segment.available())
    img = np.full((50, 50), 255, np.uint8)
    # without a checkpoint the learned backend fails with a clear message
    monkeypatch.delenv("KELPOD_SEG_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="KELPOD_SEG_CHECKPOINT"):
        segment.run(page_from(img), backend="learned")
    with pytest.raises(ValueError):
        segment.run(page_from(img), backend="nope")
    assert isinstance(segment.get_segmenter("classical"), segment.Segmenter)


def test_blank_page():
    seg = segment.run(page_from(np.full((300, 400), 255, np.uint8)))
    assert isinstance(seg, Segmentation)
    assert seg.walls == [] and seg.openings == [] and seg.rooms == []
    assert seg.wall_mask.shape == (300, 400) and seg.wall_mask.dtype == bool


def test_fill_narrow_holes_keeps_rooms_open():
    ink = np.zeros((200, 200), bool)
    ink[20, 20:180] = ink[30, 20:180] = True  # narrow cell between two lines, closed at the ends
    ink[20:31, 20] = ink[20:31, 179] = True
    ink[100, 20:180] = ink[180, 20:180] = True  # wide room
    ink[100:181, 20] = ink[100:181, 179] = True
    f = fill_narrow_holes(ink, max_radius_px=10)
    assert f[25, 100] and not f[140, 100]


def test_house_walls(house_result):
    seg, gt = house_result
    mpp = gt["m_per_px"]
    for a, b, th, _hatched in gt["walls"]:
        # every GT wall has a parallel predicted wall on (near) its centreline
        horiz = abs(b[1] - a[1]) < abs(b[0] - a[0])
        near = [
            w
            for w in seg.walls
            if _seg_dist(((w.p0[0] + w.p1[0]) / 2, (w.p0[1] + w.p1[1]) / 2), a, b) <= 0.05 / mpp
            and (abs(w.p1[1] - w.p0[1]) < abs(w.p1[0] - w.p0[0])) == horiz
        ]
        assert near, f"no wall near {a}-{b}"
        assert abs(near[0].thickness_px - th) * mpp < 0.035
    # all predicted walls lie on a GT wall (no dimension chains / text / door leaves)
    for w in seg.walls:
        mid = ((w.p0[0] + w.p1[0]) / 2, (w.p0[1] + w.p1[1]) / 2)
        assert min(_seg_dist(mid, a, b) for a, b, *_ in gt["walls"]) <= 0.05 / mpp
    kinds = {w.kind for w in seg.walls}
    assert WallKind.EXTERIOR in kinds and WallKind.INTERIOR in kinds and WallKind.DRYWALL in kinds
    assert seg.wall_mask.any()


def test_house_openings(house_result):
    seg, gt = house_result
    mpp = gt["m_per_px"]
    assert len(seg.openings) == len(gt["openings"])
    for kind, a, b, _ in gt["openings"]:
        c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        o = min(
            seg.openings,
            key=lambda o: np.hypot((o.p0[0] + o.p1[0]) / 2 - c[0], (o.p0[1] + o.p1[1]) / 2 - c[1]),
        )
        oc = ((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2)
        assert np.hypot(oc[0] - c[0], oc[1] - c[1]) * mpp < 0.06
        want = OpeningKind.DOOR if kind == "door" else OpeningKind.WINDOW
        assert o.kind == want
        width = np.hypot(o.p1[0] - o.p0[0], o.p1[1] - o.p0[1])
        assert abs(width - np.hypot(b[0] - a[0], b[1] - a[1])) * mpp < 0.06
        assert o.wall_id is not None
        if want == OpeningKind.DOOR:
            assert o.hinge is not None and o.swing_into is not None


def test_door_hinge_side(house_result):
    seg, gt = house_result
    # entrance door in the bottom wall: hinge at the right jamb (x ~ 315 px), swings up (inside)
    doors = [o for o in seg.openings if o.kind == OpeningKind.DOOR]
    entrance = max(doors, key=lambda o: o.p0[1])
    assert abs(entrance.hinge[0] - 315) < 4
    assert entrance.swing_into[1] < entrance.hinge[1]


def test_house_rooms(house_result):
    seg, gt = house_result
    mpp = gt["m_per_px"]
    areas = sorted(r.area_px * mpp**2 for r in seg.rooms)
    # inner faces: Wohnen 4.1975 x 5.635, Kind / Bad 3.1825 x ~2.70
    expected = sorted([4.1975 * 5.635, 3.1825 * (3.0 - 0.1825 - 0.0575), 3.1825 * 2.76])
    assert len(areas) == 3
    for got, want in zip(areas, expected, strict=True):
        assert abs(got - want) / want < 0.05
    for r in seg.rooms:
        x, y = int(r.seed[0]), int(r.seed[1])
        assert not seg.wall_mask[y, x]


def test_unhatched_page_and_no_scale():
    plan = Plan(
        walls=[
            W(0, 0, 5, 0, 0.24, hatched=False),
            W(5, 0, 5, 4, 0.24, hatched=False),
            W(5, 4, 0, 4, 0.24, hatched=False),
            W(0, 4, 0, 0, 0.24, hatched=False),
        ],
        openings=[Op(0, 1.0, 1.01, "door")],
        dims=False,
    )
    img, gt = render(plan, size_m=(6, 5))
    page = RasterPage(image=img, ink=img < 128, dpi=200, source="x", plan_scale=None)
    seg = segment.run(page)
    assert len(seg.walls) >= 4
    assert any(o.kind == OpeningKind.DOOR for o in seg.openings)
    assert len(seg.rooms) == 1
    assert any("plan_scale unknown" in n for n in seg.notes)


def test_speed_a3():
    # A3 at 200 dpi with the house tiled 4 times: must stay well under a few seconds
    import time

    img, _ = render(house())
    big = np.full((2339, 3307), 255, np.uint8)
    h, w = img.shape
    for i in range(2):
        for j in range(2):
            big[i * h : (i + 1) * h, j * w : (j + 1) * w] = img
    t = time.perf_counter()
    seg = segment.run(page_from(big))
    assert time.perf_counter() - t < 5.0
    assert len(seg.rooms) == 12


@pytest.mark.slow
@pytest.mark.parametrize("seed,degrade", [(0, 0.0), (5, 0.5), (6, 0.5)])
def test_synth(seed, degrade):
    synth = pytest.importorskip("kelpod.evaluation.synth")
    pytest.importorskip("kelpod.clean.core")
    from synth_eval import evaluate, make_page

    try:
        sp = synth.generate(seed, degrade=degrade)
    except RuntimeError as e:  # generator under development
        pytest.skip(f"synth failed: {e}")
    page, to_page = make_page(sp)
    seg = segment.run(page)
    r = evaluate(sp, seg, to_page, page)
    assert r["wall_recall"] >= 0.9 and r["wall_precision"] >= 0.9
    assert r["room_recall"] >= 0.8 and r["room_iou"] >= 0.85
    assert (r["door_recall"] or 1.0) >= 0.8 and (r["window_recall"] or 1.0) >= 0.8
