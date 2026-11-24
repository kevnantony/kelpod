"""RapidOCR on rendered pages (slow: loads the ONNX models, a few seconds per page)."""

import math
import os
import time

import numpy as np
import pytest
from ocr_render import make_fixture

import kelpod.ocr as ocr
from kelpod.contracts import RasterPage

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fx():
    return make_fixture()


@pytest.fixture(scope="module")
def ann(fx):
    return ocr.run(fx.page(), backend="rapidocr")


def test_reads_horizontal_and_vertical_dimensions(fx, ann):
    got = {(d.text, d.angle_deg): d for d in ann.dimensions}
    for ch in fx.chains:
        angle = 90.0 if ch.vertical else 0.0
        d = got.get((ch.text, angle))
        assert d is not None, (ch.text, sorted(got))
        assert d.value_m == pytest.approx(ch.value_m)
        assert d.line is not None, ch.text
        for g, e in zip(sorted(d.line), sorted(ch.ticks), strict=True):
            assert math.dist(g, e) <= 3.0


def test_no_spurious_dimensions(fx, ann):
    assert len(ann.dimensions) == len(fx.chains)


def test_sills_and_stamps(fx, ann):
    assert sorted(s.value_m for s in ann.sill_heights) == pytest.approx(sorted(fx.sills))
    stamps = {s.name: s.area_m2 for s in ann.room_stamps}
    for name, area in fx.stamps:
        assert stamps.get(name) == pytest.approx(area), (name, stamps)


def test_items_have_quads_and_angles(ann):
    assert ann.backend == "rapidocr"
    for it in ann.items:
        assert len(it.quad) == 4
        assert 0.0 <= it.confidence <= 1.0
        assert min(abs(it.angle_deg), abs(it.angle_deg - 90)) < 15


def test_a3_page_runtime(fx):
    """An A3 sheet at 200 dpi (3307 x 2339 px) with the fixture pasted twice: under 30 s."""
    img = np.full((2339, 3307), 255, np.uint8)
    h, w = fx.image.shape
    img[100 : 100 + h, 50 : 50 + w] = fx.image
    img[700 : 700 + h, 1500 : 1500 + w] = np.minimum(img[700 : 700 + h, 1500 : 1500 + w],
                                                      fx.image)
    page = RasterPage(image=img, ink=img < 128, dpi=200, source="a3")
    ocr.get_engine("rapidocr")  # model loading is not part of the per-page budget
    t0 = time.perf_counter()
    a = ocr.run(page)
    dt = time.perf_counter() - t0
    # 30 s on an idle machine; on a shared node that is oversubscribed (load > cores) the wall-clock
    # budget scales with the oversubscription, so other users' jobs do not fail this test.
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    over = max(1.0, os.getloadavg()[0] / max(1, os.cpu_count() or 1))
    assert dt < 30 * over, (dt, over, ncpu)
    assert len(a.dimensions) >= 2 * len(fx.chains) - 2


def test_synthetic_plan_dimensions():
    """Seed 0 of the synthetic generator, undegraded, resampled to 200 dpi. Measured at the time
    of writing: 36/42 dimensions read, 30/31 linked spans within 4 px of the drawn ticks."""
    synth = pytest.importorskip("kelpod.evaluation.synth")
    cv2 = pytest.importorskip("cv2")
    plan = synth.generate(0, degrade=0.0)
    sc = 200 / plan.dpi
    img = cv2.resize(plan.image, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    page = RasterPage(image=img, ink=img < 128, dpi=200, source="synth-0")
    ann = ocr.run(page)
    gt = [a for a in plan.annotations if a.kind == "dimension"]
    remaining = [a.value for a in gt]
    hits = 0
    for d in ann.dimensions:
        for i, v in enumerate(remaining):
            if abs(v - d.value_m) < 1e-3:
                hits += 1
                remaining.pop(i)
                break
    assert hits >= 0.6 * len(gt)
    assert hits >= 0.75 * len(ann.dimensions)  # precision
    linked = [d for d in ann.dimensions if d.line is not None]
    good = 0
    for d in linked:
        g = min(gt, key=lambda a: math.dist(d.anchor, (a.center[0] * sc, a.center[1] * sc)))
        exp = sorted((p[0] * sc, p[1] * sc) for p in g.line)
        if all(math.dist(p, e) <= 4 for p, e in zip(sorted(d.line), exp, strict=True)):
            good += 1
    assert len(linked) >= 0.5 * len(gt)
    assert good >= 0.85 * len(linked)
