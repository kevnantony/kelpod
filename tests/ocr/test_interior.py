"""Interior (dotted) dimension lines, drawn synthetically with cv2."""

from __future__ import annotations

import cv2
import numpy as np

from kelpod.ocr.interior import interior_span


def _room(w=400, h=200):
    ink = np.zeros((h, w), np.uint8)
    cv2.rectangle(ink, (40, 30), (360, 170), 1, 8)  # thick walls
    return ink


def _dotted(ink, y, xa, xb):
    for x in range(xa, xb, 6):
        ink[y, x : x + 3] = 1
    for x in (xa, xb):  # arrowhead barbs
        cv2.line(ink, (x, y), (x + (6 if x == xa else -6), y - 4), 1, 1)
        cv2.line(ink, (x, y), (x + (6 if x == xa else -6), y + 4), 1, 1)


def test_horizontal_span_runs_wall_to_wall():
    ink = _room()
    _dotted(ink, 100, 44, 356)
    ink[94:107, 170:230] = 0  # the value sits on the line (text omitted, box cleared)
    quad = [(170, 94), (230, 94), (230, 107), (170, 107)]
    sp = interior_span(ink.astype(bool), quad, 0.0)
    assert sp is not None
    assert abs(sp[0][0] - 44) <= 8 and abs(sp[1][0] - 356) <= 8
    assert sp[0][1] == sp[1][1] == 100.0


def test_vertical_text_gives_vertical_span():
    ink = np.zeros((300, 200), np.uint8)
    cv2.rectangle(ink, (20, 20), (180, 280), 1, 8)
    for y in range(26, 274, 6):
        ink[y : y + 3, 100] = 1
    ink[110:150, 94:107] = 0
    quad = [(94, 110), (107, 110), (107, 150), (94, 150)]
    sp = interior_span(ink.astype(bool), quad, 90.0)
    assert sp is not None and sp[0][0] == sp[1][0] == 100.0
    assert sp[1][1] - sp[0][1] > 200


def test_no_line_no_span():
    ink = _room()
    quad = [(170, 94), (230, 94), (230, 107), (170, 107)]
    assert interior_span(ink.astype(bool), quad, 0.0) is None


def test_slanted_text_is_ignored():
    ink = _room()
    _dotted(ink, 100, 44, 356)
    quad = [(170, 94), (230, 94), (230, 107), (170, 107)]
    assert interior_span(ink.astype(bool), quad, 40.0) is None
