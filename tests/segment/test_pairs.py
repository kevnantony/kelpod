"""Thin double-line walls and low-resolution upscaling (synthetic cv2 fixtures)."""

from __future__ import annotations

import cv2
import numpy as np

from kelpod.clean.io import effective_dpi, lowres_factor, upscale_sharpen
from kelpod.segment.pairs import detect_pair_walls, segment_pairs


def _double_line_house(th=14, size=420, diag=False):
    img = np.full((size, size), 255, np.uint8)
    a, b = 60, size - 60
    for lo, hi in ((a, a + th), (b - th, b)):  # outer box, each face a thin line
        cv2.rectangle(img, (lo, lo), (hi, hi), 255, -1)
    cv2.rectangle(img, (a, a), (b, b), 0, 1)
    cv2.rectangle(img, (a + th, a + th), (b - th, b - th), 0, 1)
    mid = size // 2  # partition wall as two lines, with a door gap
    for y in (mid - th // 2, mid + th // 2):
        cv2.line(img, (a + th, y), (mid - 40, y), 0, 1)
        cv2.line(img, (mid + 40, y), (b - th, y), 0, 1)
    if diag:
        cv2.line(img, (a, b - 120), (a + 120, b), 0, 1)
    return img < 128


def test_double_line_walls_are_found_and_rooms_close():
    ink = _double_line_house()
    walls, info = detect_pair_walls(ink)
    assert info["th_mode_px"] is not None and abs(info["th_mode_px"] - 14) <= 2
    assert len([w for w in walls if w.angle == 0]) >= 3
    seg, _ = segment_pairs(ink)
    assert seg is not None and len(seg.rooms) >= 1


def test_clutter_that_is_not_paired_is_ignored():
    ink = _double_line_house()
    img = ink.astype(np.uint8)
    cv2.rectangle(img, (150, 100), (200, 130), 1, 1)  # a furniture box: no wall spacing pair
    seg, _ = segment_pairs(img.astype(bool))  # isolated pieces fall out of the wall network
    assert seg is not None
    assert all(not (140 < w.p0[0] < 210 and 90 < w.p0[1] < 140) for w in seg.walls)


def test_blank_page_has_no_pair_walls():
    seg, notes = segment_pairs(np.zeros((200, 200), bool))
    assert seg is None and notes


def test_effective_dpi_and_upscale():
    img = np.full((300, 300), 255, np.uint8)
    for i in range(12):  # glyph-like boxes 5 px high -> about 51 dpi
        for j in range(8):
            cv2.rectangle(img, (10 + 30 * j, 10 + 20 * i), (13 + 30 * j, 15 + 20 * i), 0, -1)
    eff = effective_dpi(img)
    assert eff is not None and eff < 130
    f = lowres_factor(img)
    assert 1.0 < f <= 3.0
    big = upscale_sharpen(img, f)
    assert big.shape[0] == round(300 * f) and big.dtype == np.uint8


def test_high_resolution_is_left_alone():
    img = np.full((600, 600), 255, np.uint8)
    for i in range(15):
        for j in range(15):
            cv2.rectangle(img, (10 + 38 * j, 10 + 38 * i), (22 + 38 * j, 40 + 38 * i), 0, 1)
    assert lowres_factor(img) == 1.0


def test_partial_footprint_drops_loose_pieces_keeps_network():
    """Incomplete outline: dimension stubs far from the wall network go, the network stays."""
    import numpy as np

    from kelpod.segment import walls as wl
    from kelpod.segment.classical import _connected_to_footprint

    def run(orient, c, a0, a1):
        return wl.Run(orient=orient, c=c, a0=a0, a1=a1, th=20.0, label="H")

    fp = np.zeros((600, 600), bool)
    fp[100:300, 100:300] = True
    net = [run(wl.H, 90, 80, 320), run(wl.V, 90, 80, 340), run(wl.H, 330, 80, 320)]
    loose = [run(wl.V, 500, 50, 120), run(wl.H, 520, 400, 440)]
    walls, windows = _connected_to_footprint(fp, net + loose, [], min_len=200.0)
    assert walls == net and windows == []
    # a network below the minimum total length is dropped even if it borders the footprint
    walls, _ = _connected_to_footprint(fp, [run(wl.H, 90, 100, 150)], [], min_len=200.0)
    assert walls == []
