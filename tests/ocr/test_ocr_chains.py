"""Joint dimension-chain decoding (``kelpod.ocr.chains``) on drawn chains with fake readings.

A two-level chain at 100 px/m: parts 0,365 | 2,51 | 1,01 | 1,115 on the inner line, the overall
5,00 on the outer line; 45° ticks and extension lines running down towards the "wall".
"""

import math

import cv2
import numpy as np
import pytest

from kelpod.contracts import TextItem
from kelpod.ocr.chains import digits_compatible, reading_values
from kelpod.ocr.interpret import interpret
from kelpod.ocr.nbest import ctc_nbest
from kelpod.ocr.parse import parse_area

X0 = 100.0
PARTS = [0.365, 2.51, 1.01, 1.115]
Y_IN, Y_OUT = 200, 140
TEXT_H = 20


def _ticks():
    xs = [X0]
    for v in PARTS:
        xs.append(xs[-1] + v * 100)
    return xs


def _ink() -> np.ndarray:
    img = np.zeros((400, 800), np.uint8)
    xs = _ticks()
    for y, ticks in ((Y_IN, xs), (Y_OUT, [xs[0], xs[-1]])):
        cv2.line(img, (int(ticks[0]) - 12, y), (int(round(ticks[-1])) + 12, y), 1, 1)
        for x in ticks:
            xi = int(round(x))
            cv2.line(img, (xi - 7, y + 7), (xi + 7, y - 7), 1, 2)  # 45° tick
            cv2.line(img, (xi, y - 10), (xi, 330), 1, 1)  # extension line to the wall
    return img.astype(bool)


def _item(text, x0, x1, y_line, alts=None, conf=0.99):
    cx = (x0 + x1) / 2
    w = max(14.0, 9.0 * len(text))
    y1 = y_line - 4
    q = [(cx - w / 2, y1 - TEXT_H), (cx + w / 2, y1 - TEXT_H), (cx + w / 2, y1), (cx - w / 2, y1)]
    return TextItem(text=text, quad=q, confidence=conf, angle_deg=0.0,
                    alternatives=alts if alts is not None else [(text, 0.95)])


def _labels(overrides=None, skip=()):
    xs = _ticks()
    overrides = overrides or {}
    texts = ["0,365", "2,51", "1,01", "1,115"]
    items = []
    for k, t in enumerate(texts):
        if k in skip:
            continue
        text, alts = overrides.get(k, (t, None))
        items.append(_item(text, xs[k], xs[k + 1], Y_IN, alts))
    items.append(_item("5,00", xs[0], xs[-1], Y_OUT))
    return items


def _dims(items):
    dims, _, _, _ = interpret(items, _ink(), decode="chain")
    return sorted(dims, key=lambda d: (d.anchor[1], d.anchor[0]))


def _inner(dims):
    return [d for d in dims if d.line is not None and abs(d.line[0][1] - Y_IN) < 3]


def test_clean_chain_unchanged():
    dims = _dims(_labels())
    assert [d.value_m for d in _inner(dims)] == pytest.approx(PARTS)
    assert all(d.origin == "read" for d in dims)


def test_chain_sum_repairs_misread():
    # 2,51 read as 2,57 (6 px off at 100 px/m: within the span tolerance); n-best has 2,51
    dims = _dims(_labels({1: ("2,57", [("2,57", 0.6), ("2,51", 0.35)])}))
    fixed = [d for d in dims if d.text == "2,57"]
    assert len(fixed) == 1 and fixed[0].value_m == pytest.approx(2.51)
    assert fixed[0].origin == "corrected"


def test_pixel_span_resolves_1_vs_7():
    dims = _dims(_labels({2: ("7,01", [("7,01", 0.7), ("1,01", 0.25)])}))
    d = [d for d in dims if d.text == "7,01"]
    assert len(d) == 1 and d[0].value_m == pytest.approx(1.01)


def test_missing_label_inferred_from_overall():
    dims = _dims(_labels(skip=(3,)))
    inf = [d for d in dims if d.origin == "inferred"]
    assert len(inf) == 1 and inf[0].value_m == pytest.approx(1.115)
    (a, b) = inf[0].line
    assert abs(min(a[0], b[0]) - _ticks()[3]) < 3 and abs(max(a[0], b[0]) - _ticks()[4]) < 3


def test_lost_superscript_repaired():
    # 36⁵ read as "36" with a trace of the raised 5 in the n-best list
    dims = _dims(_labels({0: ("36", [("36", 0.8), ("36²", 0.1)])}))
    d = [d for d in dims if d.text == "36"]
    assert len(d) == 1 and d[0].value_m == pytest.approx(0.365)


def test_reading_values_variants():
    vals = dict(reading_values(TextItem(text="1245", quad=[(0, 0)] * 4,
                                        alternatives=[("1245", 0.9)])))
    assert 12.45 in vals and 1.245 in vals  # 1245 cm vs 124⁵ cm; the span decides
    assert vals[12.45] < vals[1.245]
    vals = dict(reading_values(TextItem(text="0,3651", quad=[(0, 0)] * 4)))
    assert 0.365 in vals and 0.3651 not in vals  # whole millimetres only


def test_reading_values_zero_probabilities():
    it = TextItem(text="2,51", quad=[(0, 0)] * 4, confidence=0.0, alternatives=[("2,51", 0.0)])
    assert dict(reading_values(it))[2.51] == 0.0


def test_digits_compatible():
    assert digits_compatible("36", 0.365)
    assert digits_compatible("605", 1.605)
    assert digits_compatible("15", 0.175)
    assert not digits_compatible("4,45", 4.625)


def test_parse_area_lost_comma():
    assert parse_area("2091mF")[0] == pytest.approx(20.91)
    assert parse_area("2142m1")[0] == pytest.approx(21.42)
    assert parse_area("14,52 m²")[0] == pytest.approx(14.52)
    assert parse_area("7,9 qm")[0] == pytest.approx(7.9)


def test_ctc_nbest_sums_alignments():
    charset = ["blank", "1", "7", ",", "0"]
    # frames: 1/7 ambiguous, blank, comma, 0, blank, 1
    p = np.full((6, 5), 1e-4)
    p[0, 1], p[0, 2] = 0.55, 0.45
    p[1, 0] = 0.99
    p[2, 3] = 0.99
    p[3, 4] = 0.99
    p[4, 0] = 0.99
    p[5, 1] = 0.99
    p /= p.sum(axis=1, keepdims=True)
    out = ctc_nbest(p, charset, k=3)
    assert out[0][0] == "1,01" and out[1][0] == "7,01"
    assert out[0][1] == pytest.approx(0.55 * 0.99**5, rel=0.05)
    assert math.isclose(sum(q for _, q in out), 1.0, rel_tol=0.1)
