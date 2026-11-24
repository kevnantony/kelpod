"""Sheets with several plans on one page (e.g. a plate of farmhouse plans).

``split_plans`` finds the separate plan regions: large connected wall structures (strokes closed
into blobs), ignoring scan borders, scale bars, captions and lone text. Plans are never merged.
The result is ordered by size; ``crop_page`` makes one ``RasterPage`` per region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kelpod.contracts import RasterPage


@dataclass
class PlanRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    ink_px: int
    solid: float = 1.0
    """Share of the region's ink that is heavy (walls/poche): thin-line and hatched DIN plans, and
    their dimension chains, are low."""
    """Heavy (wall) ink inside the box: the ordering key (largest plan first)."""

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def split_plans(
    ink: np.ndarray, *, close_frac: float = 0.002, min_side_frac: float = 0.04
) -> list[PlanRegion]:
    """Plan regions of a sheet, largest first. A single plan (or nothing sheet-like) gives
    at most one region; callers treat ``len(regions) < 2`` as "not a sheet"."""
    h, w = ink.shape
    m = ink.astype(np.uint8)
    # scan borders / black margins: solid or long components touching the page edge
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, a = (int(v) for v in st[i])
        edge = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        if edge and (cw > 0.5 * w or ch > 0.5 * h or a > 0.5 * cw * ch):
            m[lab == i] = 0
    # heavy strokes only (walls, poche): thin lines and text do not make a plan
    k = max(2, int(round(0.004 * max(h, w))))
    heavy = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    blob = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (c := 2 * int(close_frac * max(h, w)) + 1, c))
    )
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    out: list[PlanRegion] = []
    side = min_side_frac * max(h, w)
    for i in range(1, n):
        x, y, cw, ch, _ = (int(v) for v in st[i])
        if cw < side or ch < side or min(cw, ch) < 0.25 * max(cw, ch):
            continue  # too small, or a thin strip (dimension chain, caption line, border)
        hv = int(heavy[y : y + ch, x : x + cw][lab[y : y + ch, x : x + cw] == i].sum())
        if hv < 0.004 * cw * ch:  # a bar, a caption block: no wall mass
            continue
        # a plan encloses several rooms; chains, captions and scale bars enclose nothing
        sub = (lab[y : y + ch, x : x + cw] == i).astype(np.uint8)
        cs, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = sum(
            1
            for k, c in enumerate(cs)
            if hier[0][k][3] >= 0 and cv2.contourArea(c) >= 0.01 * cw * ch
        )
        if holes < 2:
            continue
        total = int(m[y : y + ch, x : x + cw][lab[y : y + ch, x : x + cw] == i].sum())
        out.append(PlanRegion(x, y, x + cw, y + ch, hv, hv / max(total, 1)))
    out.sort(key=lambda r: -r.ink_px)
    if out:
        out = [r for r in out if r.ink_px >= 0.015 * out[0].ink_px]
    # a sheet: at least two clearly solid-walled plans (poche). Anything else (DIN plans with
    # their chains, a single historic plan with a caption) is left to the normal pipeline.
    if sum(r.solid >= 0.25 for r in out) < 2:
        return out[:1]
    return out


def crop_page(page: RasterPage, region: PlanRegion, pad_frac: float = 0.03) -> RasterPage:
    """The region (plus a small margin) as its own page; ``origin`` keeps the offset."""
    h, w = page.image.shape
    pad = int(pad_frac * max(region.x1 - region.x0, region.y1 - region.y0))
    x0, y0 = max(0, region.x0 - pad), max(0, region.y0 - pad)
    x1, y1 = min(w, region.x1 + pad), min(h, region.y1 + pad)
    return page.model_copy(
        update={
            "image": page.image[y0:y1, x0:x1].copy(),
            "ink": page.ink[y0:y1, x0:x1].copy(),
            "origin": (x0 + page.origin[0], y0 + page.origin[1]),
        }
    )
