from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from kelpod.contracts import OpeningKind, WallKind, m_to_px
from kelpod.evaluation.synth import (
    format_area,
    format_length,
    format_sill,
    generate,
    load,
    save,
)

SEEDS = (0, 3, 7)


@pytest.fixture(scope="module")
def plans():
    return {s: generate(s, degrade=0.5) for s in SEEDS}


def test_deterministic():
    a, b = generate(11, degrade=0.6), generate(11, degrade=0.6)
    assert np.array_equal(a.image, b.image)
    assert a.graph.model_dump() == b.graph.model_dump()
    assert [t.model_dump() for t in a.annotations] == [t.model_dump() for t in b.annotations]
    assert not np.array_equal(a.image, generate(12, degrade=0.6).image)


def test_degrade_does_not_change_layout():
    clean, dirty = generate(5, degrade=0.0), generate(5, degrade=1.0)
    assert clean.graph.model_dump() == dirty.graph.model_dump()
    assert np.array_equal(clean.clean_image, dirty.clean_image)
    assert np.array_equal(clean.image, clean.clean_image)
    assert abs(dirty.degradation.rotation_deg) <= 3.0
    assert not np.array_equal(dirty.image, dirty.clean_image)


def test_ground_truth_closed_and_consistent(plans):
    for p in plans.values():
        g = p.graph
        assert g.scale_source == "ground_truth"
        assert math.isclose(g.m_per_px, 0.0254 / 300 * 100)
        assert g.image_height_px == p.image.shape[0]
        assert g.report is not None and g.report.is_closed, g.report.issues
        assert g.report.issues == []
        assert 3 <= len(g.rooms) <= 8
        pos = {n.id: (n.x, n.y) for n in g.nodes}
        for w in g.walls:  # rectilinear
            (x0, y0), (x1, y1) = pos[w.start], pos[w.end]
            assert abs(x0 - x1) < 1e-9 or abs(y0 - y1) < 1e-9
        ext = [w for w in g.walls if w.kind == WallKind.EXTERIOR]
        assert ext and all(w.thickness_m == 0.365 for w in ext)
        polys = [Polygon(r.polygon) for r in g.rooms]
        for r, poly in zip(g.rooms, polys, strict=True):
            assert poly.is_valid and poly.exterior.is_ccw
            assert math.isclose(poly.area, r.area_m2, abs_tol=1e-6)
            assert abs(r.stamp_area_m2 - r.area_m2) <= 0.005 + 1e-9  # stamp = rounded area
        assert unary_union(polys).area == pytest.approx(sum(p.area for p in polys))  # no overlap
        walls = {w.id: w for w in g.walls}
        kinds = {o.kind for o in g.openings}
        assert OpeningKind.DOOR in kinds and OpeningKind.WINDOW in kinds
        for o in g.openings:
            w = walls[o.wall_id]
            length = math.dist(pos[w.start], pos[w.end])
            assert 0 <= o.offset_m and o.offset_m + o.width_m <= length + 1e-9
            if o.kind == OpeningKind.WINDOW:
                assert w.kind == WallKind.EXTERIOR and o.sill_m > 0
            else:
                assert o.sill_m == 0 and o.height_m == pytest.approx(2.01)


def test_stamps_match_rooms(plans):
    for p in plans.values():
        g = p.graph
        names = {a.room_id: a for a in p.annotations if a.kind == "room_name"}
        areas = {a.room_id: a for a in p.annotations if a.kind == "room_area"}
        assert set(names) == set(areas) == {r.id for r in g.rooms}
        for r in g.rooms:
            assert names[r.id].text == r.name
            assert areas[r.id].text == format_area(r.stamp_area_m2)
            # the stamp is drawn inside its room
            poly_px = Polygon([m_to_px(q, g.m_per_px, g.image_height_px) for q in r.polygon])
            assert poly_px.contains(Polygon(names[r.id].quad).centroid)


def test_dimensions_drawn_and_consistent(plans):
    for p in plans.values():
        g = p.graph
        dims = [a for a in p.annotations if a.kind == "dimension"]
        assert {a.angle_deg for a in dims} == {0.0, 90.0}  # horizontal and rotated chains
        xs = [n.x for n in g.nodes]
        ys = [n.y for n in g.nodes]
        overall_w = max(xs) - min(xs) + 0.365
        overall_h = max(ys) - min(ys) + 0.365
        values = [a.value for a in dims]
        assert sum(math.isclose(v, overall_w, abs_tol=1e-6) for v in values) >= 2
        assert sum(math.isclose(v, overall_h, abs_tol=1e-6) for v in values) >= 2
        for a in dims:
            (x0, y0), (x1, y1) = a.line
            span_m = math.hypot(x1 - x0, y1 - y0) * g.m_per_px
            assert span_m == pytest.approx(a.value, abs=1e-6)
            assert a.text == format_length(a.value, p.length_style)
        sills = [a for a in p.annotations if a.kind == "sill"]
        assert sills
        for a in sills:
            o = g.openings[a.opening_id]
            assert o.kind == OpeningKind.WINDOW and a.value == o.sill_m


def test_drawing_matches_ground_truth(plans):
    p = plans[0]
    g = p.graph
    # wall centrelines lie on wall pixels, room seeds on free paper
    pos = {n.id: (n.x, n.y) for n in g.nodes}
    for w in g.walls:
        a, b = pos[w.start], pos[w.end]
        mid = m_to_px(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), g.m_per_px, g.image_height_px)
        covered = any(  # the midpoint may sit in a door/window gap; check a few points
            p.wall_mask[int(round(m[1])), int(round(m[0]))]
            for t in (0.1, 0.3, 0.5, 0.7, 0.9)
            for m in [
                m_to_px(
                    (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])),
                    g.m_per_px,
                    g.image_height_px,
                )
            ]
        )
        assert covered, (w, mid)
    for r in g.rooms:
        c = Polygon(r.polygon).representative_point()
        x, y = m_to_px((c.x, c.y), g.m_per_px, g.image_height_px)
        assert not p.wall_mask[int(y), int(x)]
    # ink at wall outlines in the clean image
    assert (p.clean_image[p.wall_mask] < 128).mean() > 0.2


def test_room_count_option():
    for n in (3, 8):
        assert len(generate(1, degrade=0.0, n_rooms=n).graph.rooms) == n


def test_formats():
    assert format_length(3.51) == "3,51"
    assert format_length(11.24, "m") == "11,24"
    assert format_length(0.885, "cm") == "88,5"
    assert format_length(0.885, "sup") == "88⁵"
    assert format_length(0.885, "m") == "0,885"
    assert format_length(0.76) == "76"
    assert format_length(1.015, "sup") == "1,01⁵"
    assert format_length(1.015, "cm") == "1,015"
    assert format_area(14.52) == "14,52 m²"
    assert format_sill(0.9) == "BRH 0,90"
    assert format_sill(0.875, "cm") == "BRH 87,5"
    assert format_sill(0.9, "cm") == "BRH 90"


def test_save_and_load(tmp_path, plans):
    p = plans[3]
    d = save(p, tmp_path / "s3", clean=True)
    for f in ("plan.png", "plan.pdf", "ground_truth.json", "plan_clean.png", "wall_mask.png"):
        assert (d / f).is_file()
    gt = load(d, image=True)
    assert gt.graph.model_dump() == p.graph.model_dump()
    assert len(gt.annotations) == len(p.annotations)
    assert np.array_equal(gt.image, p.image)
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(d / "plan.pdf"))
    w_pt, h_pt = pdf[0].get_size()
    assert w_pt == pytest.approx(p.image.shape[1] / p.dpi * 72, rel=1e-3)
    assert h_pt == pytest.approx(p.image.shape[0] / p.dpi * 72, rel=1e-3)
    pdf.close()


@pytest.fixture(params=["truetype", "hershey"])
def font_mode(request, monkeypatch):
    from kelpod.evaluation.synth import text

    if request.param == "hershey":
        monkeypatch.setenv("KELPOD_SYNTH_FONT", "hershey")
    else:
        monkeypatch.delenv("KELPOD_SYNTH_FONT", raising=False)
        if text.font_path.__wrapped__() is None:
            pytest.skip("no TrueType font installed")
    text.font_path.cache_clear()
    yield request.param
    monkeypatch.undo()
    text.font_path.cache_clear()


def test_hershey_fallback(monkeypatch):
    from kelpod.evaluation.synth import text

    monkeypatch.setenv("KELPOD_SYNTH_FONT", "hershey")
    text.font_path.cache_clear()
    try:
        p = generate(2, degrade=0.0)
        texts = [a.text for a in p.annotations]
        assert all("ü" not in t for t in texts)  # umlauts are spelled out
        assert any(t.endswith(" m²") for t in texts)  # superscripts stay semantic
    finally:
        monkeypatch.delenv("KELPOD_SYNTH_FONT")
        text.font_path.cache_clear()


def _ink_box(mask: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    ys, xs = np.nonzero(mask > thresh)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation of two ink crops after resizing ``a`` to ``b``'s shape."""
    import cv2

    a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@pytest.mark.parametrize(
    "text_in, base, sup", [("36⁵", "36", "5"), ("1,01⁵", "1,01", "5"), ("14,52 m²", "14,52 m", "2")]
)
def test_superscript_digits_are_drawn_as_raised_digits(font_mode, text_in, base, sup):
    from kelpod.evaluation.synth.text import SUP_SCALE, render_text

    digit_h = 30.0
    full, drawn = render_text(text_in, digit_h)
    assert drawn == text_in  # recorded text keeps the superscript
    plain, _ = render_text(base, digit_h)
    # the superscript is whatever ink lies right of the normal part
    region = full[:, plain.shape[1] :]
    assert (region > 0.3).sum() > 10, "superscript not drawn"
    glyph = _ink_box(region)
    ref = _ink_box(render_text(sup, digit_h * SUP_SCALE)[0])
    tofu = np.zeros_like(ref)
    tofu[[0, -1], :] = tofu[:, [0, -1]] = 1.0  # an empty box
    sim = _similarity(glyph, ref)
    assert sim > 0.7, sim
    assert sim > _similarity(glyph, tofu) + 0.2
    # smaller than a normal digit and raised: its ink ends well above the baseline
    assert glyph.shape[0] < 0.8 * _ink_box(plain).shape[0]
    rows = np.nonzero((region > 0.3).any(axis=1))[0]
    base_rows = np.nonzero((full[:, : plain.shape[1]] > 0.3).any(axis=1))[0]
    assert rows.max() < base_rows.max() - 0.25 * digit_h


def test_superscript_rendering_is_deterministic():
    from kelpod.evaluation.synth.text import render_text

    a, _ = render_text("1,01⁵", 30.0)
    b, _ = render_text("1,01⁵", 30.0)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_many_seeds_closed():
    for s in range(60):
        p = generate(s, degrade=0.3)
        assert p.graph.report.is_closed, (s, p.graph.report.issues)
