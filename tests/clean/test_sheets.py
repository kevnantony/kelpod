"""Plan-region splitting on a synthetic sheet drawn with cv2."""

from __future__ import annotations

import cv2
import numpy as np

from kelpod.clean import crop_page, split_plans
from kelpod.contracts import RasterPage


def _plan(ink, x, y, w, h, t=6):
    cv2.rectangle(ink, (x, y), (x + w, y + h), 1, t)
    cv2.line(ink, (x + w // 2, y), (x + w // 2, y + h), 1, t)
    cv2.line(ink, (x, y + h // 2), (x + w // 2, y + h // 2), 1, t)


def _sheet():
    ink = np.zeros((800, 1100), np.uint8)
    _plan(ink, 60, 60, 300, 600)  # big plan
    _plan(ink, 500, 80, 220, 200)
    _plan(ink, 780, 120, 200, 220)
    cv2.line(ink, (480, 500), (900, 500), 1, 2)  # scale bar-like line
    for x in range(480, 900, 20):
        cv2.line(ink, (x, 494), (x, 500), 1, 2)
    cv2.putText(ink, "ERDGESCHOSS", (520, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1, 2)
    ink[:, :12] = 1  # black scan margin
    return ink.astype(bool)


def test_three_plans_found_largest_first():
    regs = split_plans(_sheet())
    assert len(regs) == 3
    assert regs[0].x0 < 100 and regs[0].y1 > 600
    assert regs[0].ink_px >= regs[1].ink_px >= regs[2].ink_px


def test_single_plan_is_not_a_sheet():
    ink = np.zeros((800, 600), np.uint8)
    _plan(ink, 80, 80, 400, 600)
    assert len(split_plans(ink.astype(bool))) == 1


def test_empty_page_gives_nothing():
    assert split_plans(np.zeros((300, 300), bool)) == []


def test_crop_page_keeps_origin_and_shape():
    ink = _sheet()
    page = RasterPage(image=np.where(ink, 0, 255).astype(np.uint8), ink=ink, dpi=200.0, source="x")
    reg = split_plans(ink)[1]
    sub = crop_page(page, reg)
    assert sub.image.shape == sub.ink.shape
    assert sub.origin[0] <= reg.x0 and sub.origin[1] <= reg.y0
    assert sub.image.shape[1] < page.image.shape[1]


def test_dimension_chain_strips_around_a_hatched_plan_are_not_plans():
    ink = np.zeros((900, 1200), np.uint8)
    cv2.rectangle(ink, (200, 200), (1000, 700), 1, 2)  # thin-walled plan: no heavy ink
    for y in (120, 800):  # chains: long thin line with ticks
        cv2.line(ink, (200, y), (1000, y), 1, 2)
        for x in range(200, 1001, 50):
            cv2.line(ink, (x, y - 8), (x, y + 8), 1, 2)
    for x in (120, 1080):
        cv2.line(ink, (x, 200), (x, 700), 1, 2)
    assert len(split_plans(ink.astype(bool))) < 2
