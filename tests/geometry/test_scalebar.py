"""Scale-bar detection on synthetic bars drawn with cv2 (not the user's images)."""

from __future__ import annotations

import cv2
import numpy as np

from kelpod.contracts import TextItem
from kelpod.geometry.scalebar import (
    assume_tick_value,
    bar_scale,
    detect_scale_bar,
    read_bar_numbers,
)


def _page(tick_px=18, n=16, y=300, big=False):
    ink = np.zeros((400, 800), np.uint8)
    x0 = 100
    cv2.line(ink, (x0, y), (x0 + tick_px * n, y), 1, 2)
    for k in range(n + 1):
        cv2.line(ink, (x0 + tick_px * k, y - 6), (x0 + tick_px * k, y), 1, 2)
    # a wall-like blob and text-like noise that must not be taken for a bar
    cv2.rectangle(ink, (50, 20), (700, 28), 1, -1)
    return ink.astype(bool)


def test_detects_regular_ticks():
    bar = detect_scale_bar(_page())
    assert bar is not None
    assert len(bar.ticks_x) >= 16
    assert abs(bar.tick_px - 18) < 0.6


def test_plain_line_without_ticks_is_not_a_bar():
    ink = np.zeros((200, 600), np.uint8)
    cv2.line(ink, (50, 100), (400, 100), 1, 2)
    assert detect_scale_bar(ink.astype(bool)) is None


def test_irregular_ticks_rejected():
    ink = np.zeros((200, 600), np.uint8)
    cv2.line(ink, (50, 100), (450, 100), 1, 2)
    for x in (50, 70, 130, 140, 220, 300, 310, 420):
        cv2.line(ink, (x, 94), (x, 100), 1, 2)
    assert detect_scale_bar(ink.astype(bool)) is None


def _item(text, x, y):
    quad = [(x - 5, y - 4), (x + 5, y - 4), (x + 5, y + 4), (x - 5, y + 4)]
    return TextItem(text=text, quad=quad)


def test_numbers_give_scale_in_metres():
    bar = detect_scale_bar(_page(tick_px=20, n=10))
    items = [_item(str(5 * k), 100 + 100 * k, 320) for k in range(3)]  # 0,5,10 every 100 px
    a = read_bar_numbers(bar, items)
    assert a is not None and abs(a - 0.05) < 0.002


def test_numbers_far_from_bar_are_ignored():
    bar = detect_scale_bar(_page())
    assert read_bar_numbers(bar, [_item("5", 120, 50), _item("10", 200, 50), _item("15", 300, 50)]) is None


def test_assumed_tick_value_from_wall_thickness():
    bar = detect_scale_bar(_page(tick_px=18))
    assert assume_tick_value(bar, 7.0) == 1.0  # 7 px -> 0.39 m
    assert assume_tick_value(bar, 3.5) == 2.0
    assert assume_tick_value(bar, None) is None
    assert assume_tick_value(bar, 0.2) is None


def test_bar_scale_sources():
    ink = _page()
    r = bar_scale(ink, [], 7.0)
    assert r is not None and r[1] == "scale_bar_assumed" and abs(r[0] - 1 / 18) < 0.003
    assert bar_scale(np.zeros((50, 50), bool), [], 7.0) is None


def test_scale_from_loosely_agreeing_interior_dimensions():
    from kelpod.contracts import DimensionLabel, RasterPage
    from kelpod.geometry.params import GeometryParams
    from kelpod.geometry.scale import estimate_scale

    page = RasterPage(image=np.full((50, 50), 255, np.uint8), ink=np.zeros((50, 50), bool), dpi=200.0, source="x")
    # spans of 78 / 76 / 86 px for 3.0 / 3.0 / 3.3 m (+-7 %) and one misread value
    spans = [(3.0, 78), (3.0, 70), (3.3, 86), (8.0, 42)]
    dims = [
        DimensionLabel(value_m=v, text=str(v), anchor=(0, 0), line=((0, 0), (px, 0))) for v, px in spans
    ]
    m, src, _ = estimate_scale(page, dims, GeometryParams())
    assert src == "dimensions" and abs(m - 0.0385) < 0.004


def test_typical_wall_is_length_weighted():
    from kelpod.geometry.scalebar import typical_wall_px

    assert typical_wall_px([(100, 7.0), (90, 7.0), (5, 30.0), (4, 28.0)]) == 7.0
    assert typical_wall_px([]) is None


def test_ink_thickness_ignores_pen_lines():
    from kelpod.geometry.scalebar import ink_thickness_px

    ink = np.zeros((300, 300), np.uint8)
    cv2.rectangle(ink, (20, 20), (280, 280), 1, 12)  # poche walls, 12 px
    for y in range(40, 260, 5):
        cv2.line(ink, (40, y), (260, y), 1, 2)  # thin pen lines
    t = ink_thickness_px(ink.astype(bool))
    assert t is not None and 9 <= t <= 14
    assert ink_thickness_px(np.zeros((50, 50), bool)) is None
