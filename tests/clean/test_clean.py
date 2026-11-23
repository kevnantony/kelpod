from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from fixtures import degrade, draw_plan, mm, rotate_ccw  # noqa: E402

from kelpod import clean  # noqa: E402
from kelpod.clean import clean_image, deskew_angle, load  # noqa: E402
from kelpod.clean.io import DEFAULT_IMAGE_DPI  # noqa: E402
from kelpod.contracts import RasterPage  # noqa: E402

DPI = 200.0


@pytest.fixture(scope="module")
def a4_plan() -> dict:
    """A smaller page (A4 at 150 dpi) for the many-angle tests, to keep them fast."""
    return draw_plan(150.0, size_mm=(297.0, 210.0)) | {"dpi": 150.0}


@pytest.fixture(scope="module")
def a3_plan() -> dict:
    return draw_plan(DPI)


def _dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


# ------------------------------------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------------------------------------


def test_png_uses_dpi_metadata_and_resamples(tmp_path, a3_plan):
    p = tmp_path / "plan.png"
    Image.fromarray(a3_plan["image"]).save(p, dpi=(200, 200))
    (page,) = load(p, dpi=100)
    h, w = a3_plan["image"].shape
    assert page.dtype == np.uint8 and page.ndim == 2
    assert abs(page.shape[0] - h / 2) <= 1 and abs(page.shape[1] - w / 2) <= 1


def test_image_without_dpi_defaults_to_300(tmp_path):
    img = np.full((600, 900), 255, np.uint8)
    p = tmp_path / "plan.jpg"
    cv2.imwrite(str(p), img)  # OpenCV writes no usable DPI
    (page,) = load(p, dpi=150)
    assert page.shape == (
        round(600 * 150 / DEFAULT_IMAGE_DPI),
        round(900 * 150 / DEFAULT_IMAGE_DPI),
    )


def test_rgba_and_colour_become_gray_on_white(tmp_path):
    rgba = np.zeros((50, 60, 4), np.uint8)  # fully transparent black
    rgba[10:20, 10:20] = (255, 0, 0, 255)  # opaque red square
    p = tmp_path / "t.png"
    Image.fromarray(rgba, "RGBA").save(p, dpi=(300, 300))
    (page,) = load(p, dpi=300)
    assert page.shape == (50, 60)
    assert page[0, 0] == 255  # transparency -> white paper
    assert page[15, 15] < 128


def test_multipage_tiff(tmp_path):
    frames = [Image.fromarray(np.full((300, 400), v, np.uint8)) for v in (255, 128, 0)]
    p = tmp_path / "multi.tif"
    frames[0].save(p, save_all=True, append_images=frames[1:], dpi=(300, 300))
    pages = load(p, dpi=150)
    assert len(pages) == 3
    assert all(pg.shape == (150, 200) for pg in pages)
    assert [int(pg.mean()) for pg in pages] == [255, 128, 0]


def test_multipage_pdf(tmp_path, a4_plan):
    img = Image.fromarray(a4_plan["image"])
    blank = Image.fromarray(np.full_like(a4_plan["image"], 255))
    p = tmp_path / "plan.pdf"
    img.save(p, save_all=True, append_images=[blank], resolution=a4_plan["dpi"])
    pages = load(p, dpi=100)
    assert len(pages) == 2
    h, w = a4_plan["image"].shape
    f = 100 / a4_plan["dpi"]
    assert abs(pages[0].shape[0] - h * f) <= 2 and abs(pages[0].shape[1] - w * f) <= 2
    assert pages[0].min() < 64 and pages[1].min() > 200


def test_run_returns_raster_pages(tmp_path, a4_plan):
    p = tmp_path / "plan.pdf"
    Image.fromarray(a4_plan["image"]).save(p, save_all=True, resolution=a4_plan["dpi"])
    pages = clean.run(p, dpi=150, plan_scale=50)
    assert len(pages) == 1
    pg = pages[0]
    assert isinstance(pg, RasterPage)
    assert pg.image.dtype == np.uint8 and pg.ink.dtype == bool
    assert pg.image.shape == pg.ink.shape
    assert pg.dpi == 150 and pg.plan_scale == 50 and pg.page_index == 0
    assert pg.source == str(p)
    assert pg.deskew_deg == 0.0


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.pdf")


# ------------------------------------------------------------------------------------------------
# Deskew
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.15, -0.6, 1.7, -3.0, 4.4, -7.3, 9.8, -10.0])
def test_deskew_accuracy(a4_plan, angle):
    g = degrade(a4_plan["image"], a4_plan["dpi"], seed=3, angle_deg=angle)
    res = clean_image(g, a4_plan["dpi"])
    assert abs(res.deskew_deg - angle) < 0.1


def test_straight_plan_is_unchanged(a4_plan):
    res = clean_image(a4_plan["image"], a4_plan["dpi"])
    assert res.deskew_deg == 0.0
    gt = a4_plan["image"] < 128
    # No resampling happened: the ink lies exactly on the drawn ink.
    assert (res.ink & ~_dilate(gt, 1)).sum() == 0


def test_deskew_restores_geometry(a4_plan):
    angle = 2.5
    g = rotate_ccw(a4_plan["image"], angle)
    res = clean_image(g, a4_plan["dpi"])
    gt = a4_plan["image"] < 128
    # Away from the corners (which rotation crops), the straightened ink matches the original.
    h, w = gt.shape
    roi = (slice(h // 6, -h // 6), slice(w // 6, -w // 6))
    ink, ref = res.ink[roi], gt[roi]
    assert (ref & _dilate(ink, 2)).sum() / ref.sum() > 0.97
    assert (ink & _dilate(ref, 2)).sum() / ink.sum() > 0.97


def test_deskew_angle_sign_convention():
    ink = np.zeros((400, 600), bool)
    ink[200, 50:550] = True
    ink[50:350, 300] = True
    rotated = rotate_ccw(np.where(ink, 0, 255).astype(np.uint8), 4.0) < 128
    assert deskew_angle(rotated) == pytest.approx(4.0, abs=0.05)


def test_blank_page():
    res = clean_image(np.full((500, 700), 255, np.uint8), DPI)
    assert res.deskew_deg == 0.0 and not res.ink.any()
    assert (res.image == 255).all()


# ------------------------------------------------------------------------------------------------
# Cleaning quality
# ------------------------------------------------------------------------------------------------


def test_degraded_page_cleans_to_drawn_ink(a3_plan):
    g = degrade(a3_plan["image"], DPI, seed=1, noise=20, stains=5, fold=True, uneven=0.3)
    res = clean_image(g, DPI)
    gt = a3_plan["image"] < 128
    precision = (res.ink & _dilate(gt)).sum() / res.ink.sum()
    recall = (gt & _dilate(res.ink)).sum() / gt.sum()
    assert precision > 0.99 and recall > 0.99
    # A naive global threshold of the same scan is much worse (stains, folds, shading).
    naive = g < 128
    assert (naive & ~_dilate(gt)).sum() > 20 * max(1, (res.ink & ~_dilate(gt)).sum())


def test_thin_lines_kept(a3_plan):
    g = degrade(a3_plan["image"], DPI, seed=2, noise=15, blur=0.7)
    res = clean_image(g, DPI)
    thin = a3_plan["thin"]
    assert (thin & _dilate(res.ink)).sum() / thin.sum() > 0.97


def test_fold_lines_and_stains_removed(a3_plan):
    g = degrade(a3_plan["image"], DPI, seed=4, noise=10, stains=6, fold=True, uneven=0.4)
    res = clean_image(g, DPI)
    gt = _dilate(a3_plan["image"] < 128, 2)
    spurious = res.ink & ~gt
    # No long spurious structure survives (a fold would be a page-long line).
    n, _, stats, _ = cv2.connectedComponentsWithStats(spurious.astype(np.uint8), connectivity=8)
    extent = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    assert extent.max(initial=0) < mm(2.0, DPI)
    # Paper is white in the cleaned image, also under stains.
    assert np.median(res.image[~gt]) == 255


def test_speckles_removed_text_dots_kept():
    img = np.full((400, 400), 255, np.uint8)
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 400, 300), rng.integers(0, 400, 300)
    img[ys, xs] = 0  # isolated salt-and-pepper speckles
    cv2.circle(img, (200, 200), 2, 0, -1)  # a decimal-comma-sized dot (~13 px)
    res = clean_image(img, DPI, deskew=False)
    assert res.ink[200, 200]
    assert res.ink.sum() < 40


def test_a3_timing(a3_plan):
    g = degrade(a3_plan["image"], DPI, seed=5, angle_deg=1.3)
    clean_image(g[:200, :200], DPI)  # warm-up
    t0 = time.perf_counter()
    clean_image(g, DPI)
    assert time.perf_counter() - t0 < 3.0
