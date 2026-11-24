"""Regression tests for segmentation failures that left synthetic plans unclosed.

Each test reproduces one failure mode seen on ``kelpod.evaluation.synth`` plans (seeds 2, 9 at
degrade 0.5 and seeds 0-9 at degrade 1.0) on a small hand-built fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from drawing import W, house, page_from, render  # noqa: E402

from kelpod import segment  # noqa: E402
from kelpod.contracts import OpeningKind  # noqa: E402
from kelpod.segment import openings as op  # noqa: E402
from kelpod.segment import walls as wl  # noqa: E402

MARGIN = 1.5


def _px(x_m: float, y_m: float, mpp: float) -> tuple[int, int]:
    return int(round((x_m + MARGIN) / mpp)), int(round((y_m + MARGIN) / mpp))


def _seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def _all_walls_on_gt(seg, gt, tol_m=0.05):
    mpp = gt["m_per_px"]
    for w in seg.walls:
        mid = ((w.p0[0] + w.p1[0]) / 2, (w.p0[1] + w.p1[1]) / 2)
        d = min(_seg_dist(mid, a, b) for a, b, *_ in gt["walls"])
        assert d <= tol_m / mpp, f"wall {w.p0}-{w.p1} is not on any drawn wall"


# ------------------------------------------------------------------------------------------------
# band texture: face outline peeling
# ------------------------------------------------------------------------------------------------


def _hatched_band(rows: int, cols: int, outline: int) -> np.ndarray:
    band = np.zeros((rows, cols), bool)
    band[:outline] = band[rows - outline :] = True
    for k in range(-rows, cols, 10):  # 45 degree hatching
        for r in range(rows):
            if 0 <= k + r < cols:
                band[r, k + r : k + r + 3] = True
    return band


def test_partly_inked_edge_row_is_not_glazing():
    # seed 2: the outermost row of the band is only partly inked (the face steps where a
    # thinner window joins); the outline below it was read as a glazing centre line
    band = _hatched_band(33, 100, 4)
    band[0] = False
    band[0, :55] = True  # ~0.55 of the row
    labels, lw = wl._column_labels(band, 33)
    assert set(labels) == {"H"}
    assert lw == pytest.approx(4.0, abs=0.6)


def test_thin_partition_with_thick_outline_is_empty():
    # seed 9: 11 px partition, 4-row outline at the top, 3 at the bottom, empty inside
    band = np.zeros((11, 150), bool)
    band[:4] = band[8:] = True
    labels, _ = wl._column_labels(band, 11)
    assert set(labels) == {"E"}


def test_peel_outline():
    assert wl._peel_outline(np.array([0.9, 0.9, 0.1, 0.1])) == (2, 2)
    assert wl._peel_outline(np.array([0.5, 0.9, 0.9, 0.1])) == (3, 2)
    assert wl._peel_outline(np.array([0.1, 0.1, 0.1, 0.1])) == (0, 0)


# ------------------------------------------------------------------------------------------------
# unit checks
# ------------------------------------------------------------------------------------------------


def test_drop_face_bumps():
    wall = wl.Run("h", 100.0, 0.0, 300.0, 30.0, "H")
    bump = wl.Run("v", 150.0, 77.0, 118.0, 15.0, "H")  # 8 px above the face at y = 85
    stub = wl.Run("v", 250.0, 40.0, 118.0, 15.0, "H")  # 45 px: a real wall stub
    walls = [wall, bump, stub]
    wl.snap_junctions(walls, tol_px=3.0)
    kept = wl.drop_face_bumps(walls, min_len_px=9.4)
    assert wall in kept and stub in kept and bump not in kept


def _door_fixture(clutter: bool):
    ink = np.zeros((200, 200), bool)
    left = wl.Run("h", 150.0, 0.0, 60.0, 20.0, "H")
    right = wl.Run("h", 150.0, 140.0, 200.0, 20.0, "H")
    gap = op.Gap("h", 150.0, 60.0, 140.0, 20.0, left, right)
    img = np.zeros((200, 200), np.uint8)
    # hinge at (60, 140), leaf up, arc of radius 80 to the far jamb
    cv2.line(img, (60, 140), (60, 60), 1, 2)
    cv2.ellipse(img, (60, 140), (80, 80), 0, -90, 0, 1, 1)
    if clutter:  # text filling the swing area
        img[75:135, 65:130] = 0
        for y in range(78, 135, 6):
            img[y : y + 3, 66:128] = 1
    ink |= img.astype(bool)
    ink_d = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return gap, ink, ink_d


def test_door_arc_needs_empty_swing():
    gap, _, ink_d = _door_fixture(clutter=False)
    score = op.door_arc_score(gap, ink_d)
    assert score is not None and score[0] >= 0.75
    assert op.door_arc_score(gap, ink_d, max_swing_fill=1.0)[0] >= 0.75
    gap, _, ink_d = _door_fixture(clutter=True)
    score = op.door_arc_score(gap, ink_d)
    assert score is None or score[0] < 0.6


def test_line_window_needs_exterior_thickness():
    ink = np.zeros((100, 200), bool)
    ink[45, 60:140] = ink[54, 60:140] = True  # two lines across a thin gap (e.g. text strokes)
    stub = wl.Run("h", 50.0, 0.0, 60.0, 10.0, "H")
    gap = op.Gap("h", 50.0, 60.0, 140.0, 10.0, stub, None)
    ink_d = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert op.classify_gaps([gap], ink, ink_d, min_window_th=16.0) == []
    gap = op.Gap("h", 50.0, 60.0, 140.0, 10.0, stub, None)
    kept = op.classify_gaps([gap], ink, ink_d, min_window_th=0.0)
    assert kept and kept[0].kind == OpeningKind.WINDOW


# ------------------------------------------------------------------------------------------------
# whole-page fixtures
# ------------------------------------------------------------------------------------------------


def test_fold_line_is_not_a_wall():
    # degrade 1.0: a fold mark (blurred single line, ~5 px) across the page became thin
    # "drywall" pieces with loose ends
    img, gt = render(house())
    mpp = gt["m_per_px"]
    h, w = img.shape
    y0 = _px(0, 2.0, mpp)[1]
    cv2.line(img, (0, y0), (w - 1, y0 + int(0.004 * w)), 0, 5)
    seg = segment.run(page_from(img))
    _all_walls_on_gt(seg, gt)
    assert len(seg.rooms) == 3


def test_stray_line_in_partition_is_not_a_window():
    # degrade 1.0: a fold line running inside a partition reads as a glazing centre line; windows
    # inside the building (interior on both sides) are walls. Here: an unhatched 20 cm partition
    # with a line along its centre, which looks exactly like a three-line window
    plan = house()
    plan.walls[5] = W(4.5, 3.0, 8, 3.0, 0.20, hatched=False)
    img, gt = render(plan)
    mpp = gt["m_per_px"]
    a, b = _px(4.5, 3.0, mpp), _px(8.0, 3.0, mpp)
    cv2.line(img, a, b, 0, 1)
    seg = segment.run(page_from(img))
    windows = [o for o in seg.openings if o.kind == OpeningKind.WINDOW]
    for o in windows:
        assert abs((o.p0[1] + o.p1[1]) / 2 - a[1]) * mpp > 0.3, "window on the partition"
    assert len(windows) == 3
    assert len(seg.rooms) == 3


def test_bump_on_a_wall_face_adds_no_wall():
    # seed 9: text touching a wall face survived the line opening together with the wall body
    # and became a perpendicular stub sticking out ~8 px (10 cm) of the face
    img, gt = render(house())
    mpp = gt["m_per_px"]
    face = _px(6.0, 6.0 - 0.365 / 2, mpp)  # inner face of the bottom wall, Bad room
    cv2.rectangle(img, (face[0] - 7, face[1] - 8), (face[0] + 7, face[1]), 0, -1)
    seg = segment.run(page_from(img))
    _all_walls_on_gt(seg, gt)
    assert len(seg.rooms) == 3


def test_footprint_includes_small_enclosed_slivers():
    # seed 9: a sliver between an exterior wall and a stub was too small to be a room, so the
    # exterior wall beside it looked "outside the footprint" and was dropped with its window
    from kelpod.segment.classical import _footprint, _inside

    def run(orient, c, a0, a1, th=20.0):
        return wl.Run(orient, c, a0, a1, th, "H", meta={"snap0": True, "snap1": True})

    walls = [
        run("h", 50, 50, 350),
        run("h", 350, 50, 350),
        run("v", 50, 50, 350),
        run("v", 350, 50, 350),
        run("v", 320, 200, 350, th=8.0),  # stub 30 px left of the right wall: a sliver
        run("h", 200, 320, 350, th=8.0),
    ]
    fp = _footprint(walls, [], [], (400, 400), 0.0, 2000.0, 0.0127)
    assert fp is not None
    assert fp[200, 200] and fp[300, 335]  # the room and the sliver
    assert not fp[10, 10]
    assert _inside(fp, walls[3])
