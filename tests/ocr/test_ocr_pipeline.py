"""Fast tests: rotation handling with a fake engine, tick association, interpretation, ``run``."""

import math

import numpy as np
import pytest
from ocr_render import make_fixture

import kelpod.ocr as ocr
from kelpod.contracts import OcrAnnotations, RasterPage, TextItem
from kelpod.ocr.dimlines import DimensionLineFinder, find_dimension_span
from kelpod.ocr.engines import (
    cw_to_original,
    dedupe,
    get_engine,
    original_to_cw,
    quad_angle_deg,
    read_page,
    register_engine,
    rotate_cw,
)
from kelpod.ocr.interpret import interpret


@pytest.fixture(scope="module")
def fx():
    return make_fixture()


def _item(text, quad, conf=0.99, angle=None):
    return TextItem(text=text, quad=quad, confidence=conf,
                    angle_deg=quad_angle_deg(quad) if angle is None else angle)


def _hquad(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2)]


# ---------------------------------------------------------------- rotation / engines


def test_rotation_mapping_roundtrip():
    img = np.zeros((40, 70), np.uint8)
    img[5, 60] = 255
    rot = rotate_cw(img)
    assert rot.shape == (70, 40)
    ys, xs = np.nonzero(rot)
    # pixel centres: original (60.5, 5.5) <-> rotated
    p = original_to_cw((60.5, 5.5), 40)
    assert (xs[0] + 0.5, ys[0] + 0.5) == pytest.approx(p)
    assert cw_to_original(p, 40) == pytest.approx((60.5, 5.5))


def test_quad_angle():
    assert quad_angle_deg(_hquad(0, 0, 40, 10)) == pytest.approx(0)
    # bottom-to-top text: top-left as read is the box's bottom-left
    q = [(0, 40), (0, 0), (10, 0), (10, 40)]
    assert quad_angle_deg(q) == pytest.approx(90)


class _FakeEngine:
    """Returns one upright item in the unrotated pass and one in the rotated pass."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def read(self, gray):
        self.calls += 1
        if self.calls == 1:  # original: a horizontal word and a tall box (vertical text)
            return [_item("3,51", _hquad(100, 50, 60, 20)), _item("x", [(10, 10), (30, 10),
                                                                     (30, 90), (10, 90)])]
        # rotated pass: the vertical text, now upright; plus the horizontal word seen tall
        return [_item("2,76", _hquad(150, 20, 80, 20), conf=0.9),
                _item("3,5", [(30, 70), (50, 70), (50, 130), (30, 130)])]


def test_read_page_two_passes():
    items = read_page(np.full((200, 300), 255, np.uint8), _FakeEngine())
    texts = {it.text: it for it in items}
    assert set(texts) == {"3,51", "2,76"}
    assert texts["3,51"].angle_deg == pytest.approx(0)
    v = texts["2,76"]
    assert v.angle_deg == pytest.approx(90)
    # rotated frame centre (150, 20) -> original (20, 200 - 150)
    assert v.center == pytest.approx((20, 50))


def test_dedupe_keeps_confident():
    a = _item("3,51", _hquad(100, 50, 60, 20), conf=0.95)
    b = _item("3,5l", _hquad(101, 51, 58, 20), conf=0.7)
    c = _item("4,26", _hquad(300, 50, 60, 20), conf=0.8)
    assert [it.text for it in dedupe([b, a, c])] == ["3,51", "4,26"]


def test_registry():
    assert "rapidocr" in ocr.available_engines()
    assert "none" in ocr.available_engines()
    with pytest.raises(ValueError):
        get_engine("nope")
    register_engine("fake-test", _FakeEngine)
    assert get_engine("fake-test").name == "fake"


def test_run_none_backend(fx):
    ann = ocr.run(fx.page(), backend="none")
    assert isinstance(ann, OcrAnnotations)
    assert ann.backend == "none" and not ann.items


# ---------------------------------------------------------------- tick association


def test_dimension_spans_from_drawn_ticks(fx):
    finder = DimensionLineFinder(fx.image < 128)
    for ch in fx.chains:
        angle = 90.0 if ch.vertical else 0.0
        found = finder.find(ch.quad, angle)
        assert found is not None, ch.text
        (p0, p1), score = found
        got = sorted([p0, p1])
        exp = sorted(ch.ticks)
        for g, e in zip(got, exp, strict=True):
            assert math.dist(g, e) <= 3.0, (ch.text, got, exp)
        assert score > 0.6


def test_dimension_span_none_without_line():
    img = np.full((200, 400), 255, np.uint8)
    img[90:110, 180:220] = 0  # a blob, no line
    assert find_dimension_span(img < 128, _hquad(200, 100, 40, 20), 0.0) is None


def test_dimension_span_rejects_missing_tick(fx):
    img = fx.image.copy()
    ch = fx.chains[0]  # 3,51: remove its right tick (and extension line)
    (xa, y), (xb, _) = ch.ticks
    img[int(y) - 30 : int(y) + 20, int(xb) - 10 : int(xb) + 11] = 255
    img[int(y) - 1 : int(y) + 2, int(xb) - 10 : int(xb) + 11] = 0  # keep the line itself
    found = find_dimension_span(img < 128, ch.quad, 0.0)
    assert found is None  # next tick is far away: asymmetric -> rejected


# ---------------------------------------------------------------- interpretation


def test_interpret_classifies(fx):
    items = [_item(ch.text, ch.quad) for ch in fx.chains]
    items += [
        _item("Wohnen", _hquad(600, 600, 80, 22)),
        _item("14,52 m2", _hquad(600, 635, 100, 22)),
        _item("Kuche", _hquad(1100, 600, 70, 22)),
        _item("8,10 m²", _hquad(1100, 635, 90, 22)),
        _item("Bad", _hquad(850, 950, 40, 22)),  # no area
        _item("BRH 0,90", _hquad(1350, 420, 100, 22)),
        _item("BRH", _hquad(1330, 1100, 50, 22)),
        _item("87,5", _hquad(1390, 1100, 50, 22)),
        _item("88,5/2,01", _hquad(900, 400, 110, 22)),
        _item("Grundriss EG", _hquad(900, 1350, 200, 22)),
    ]
    dims, sills, stamps, notes = interpret(items, fx.image < 128)
    assert sorted(d.text for d in dims) == sorted(ch.text for ch in fx.chains)
    by = {d.text: d for d in dims}
    for ch in fx.chains:
        d = by[ch.text]
        assert d.value_m == pytest.approx(ch.value_m)
        assert d.angle_deg == (90.0 if ch.vertical else 0.0)
        assert d.line is not None
    assert sorted(s.value_m for s in sills) == pytest.approx([0.875, 0.90])
    st = {s.name: s for s in stamps}
    assert st["Wohnen"].area_m2 == pytest.approx(14.52)
    assert st["Küche"].area_m2 == pytest.approx(8.10)
    assert st["Bad"].area_m2 is None
    assert any("not interpreted" in n for n in notes)


def test_interpret_without_ink():
    dims, *_ = interpret([_item("3,51", _hquad(10, 10, 40, 12))], None)
    assert dims[0].line is None and dims[0].value_m == pytest.approx(3.51)


def test_run_with_injected_engine(fx):
    """``run`` end to end with an engine that 'reads' the drawn labels perfectly."""

    class Perfect:
        name = "perfect"

        def __init__(self):
            self.calls = 0

        def read(self, gray):
            self.calls += 1
            if self.calls == 1:
                return [_item(c.text, c.quad) for c in fx.chains if not c.vertical]
            H = fx.image.shape[0]
            return [_item(c.text, [original_to_cw(p, H) for p in c.quad])
                    for c in fx.chains if c.vertical]

    register_engine("perfect-test", Perfect)
    page = RasterPage(image=fx.image, ink=fx.image < 128, dpi=200, source="x")
    ann = ocr.run(page, backend="perfect-test")
    assert ann.backend == "perfect"
    assert len(ann.dimensions) == len(fx.chains)
    assert all(d.line is not None for d in ann.dimensions)
    assert {d.angle_deg for d in ann.dimensions} == {0.0, 90.0}
