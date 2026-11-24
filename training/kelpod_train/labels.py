"""Training samples: a cleaned page image (as ``kelpod.clean`` produces it) plus a per-pixel
class map in exactly the same pixel grid.

Classes (``CLASSES``): 0 background / outside, 1 wall body (hatching included, opening gaps
excluded), 2 window (the gap box in the wall), 3 door (the gap box in the wall), 4 room interior
(inner wall faces). 255 = ignore.

Pages are produced like the pipeline does it: draw at 300 dpi (1:100), degrade like a scan,
resample to 200 dpi, run ``kelpod.clean.clean_image`` (deskew, flatten, binarise). Labels follow
the same transforms with nearest-neighbour warps, so image and label stay aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from kelpod.contracts import OpeningKind, PlanGraph, m_to_px

CLASSES = ("background", "wall", "window", "door", "room")
BG, WALL, WINDOW, DOOR, ROOM = range(5)
IGNORE = 255
PAGE_DPI = 200.0


@dataclass
class Sample:
    image: np.ndarray  # (H, W) uint8 cleaned page
    label: np.ndarray  # (H, W) uint8 classes
    meta: dict = field(default_factory=dict)


def _affine3(m) -> np.ndarray:
    a = np.eye(3)
    a[:2] = np.asarray(m, float)
    return a


def opening_boxes_m(graph: PlanGraph) -> list[tuple[OpeningKind, np.ndarray]]:
    """Gap rectangle (4 corners, plan metres) of every opening in a ground-truth graph."""
    walls = {w.id: w for w in graph.walls}
    pos = {n.id: np.array([n.x, n.y]) for n in graph.nodes}
    out = []
    for o in graph.openings:
        w = walls.get(o.wall_id)
        if w is None:
            continue
        a, b = pos[w.start], pos[w.end]
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        t = w.thickness_m / 2 + 0.01
        p0 = a + u * o.offset_m
        p1 = a + u * (o.offset_m + o.width_m)
        out.append((o.kind, np.array([p0 + n * t, p1 + n * t, p1 - n * t, p0 - n * t])))
    return out


def rasterise_graph_labels(
    graph: PlanGraph, shape: tuple[int, int], m_per_px: float, wall_mask: np.ndarray | None
) -> np.ndarray:
    """Class map in clean-image px for a ground-truth graph (rooms, walls, then openings)."""
    h = shape[0]
    lab = np.zeros(shape, np.uint8)

    def px(poly) -> np.ndarray:
        return np.round([m_to_px(tuple(p), m_per_px, h) for p in poly]).astype(np.int32)

    for r in graph.rooms:
        cv2.fillPoly(lab, [px(r.polygon)], ROOM)
    if wall_mask is not None:
        lab[wall_mask] = WALL
    for kind, box in opening_boxes_m(graph):
        cv2.fillPoly(lab, [px(box)], DOOR if kind == OpeningKind.DOOR else WINDOW)
    return lab


def to_page(
    degraded: np.ndarray,
    label_clean: np.ndarray,
    affine_px,
    src_dpi: float,
    *,
    dpi: float = PAGE_DPI,
    use_clean: bool = True,
) -> Sample:
    """Resample a degraded 300-dpi drawing to ``dpi``, clean it, and warp the clean-image label
    through the degradation affine, the resampling and the deskew rotation."""
    from kelpod.clean.core import clean_image
    from kelpod.clean.io import resample

    f = dpi / src_dpi
    gray = resample(degraded, src_dpi, dpi)
    if use_clean:
        res = clean_image(gray, dpi)
        image, angle = res.image, res.deskew_deg
    else:
        image, angle = gray, 0.0
    h, w = image.shape
    rot = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -angle, 1.0)
    scale = np.array([[f, 0, 0.5 * f - 0.5], [0, f, 0.5 * f - 0.5], [0, 0, 1]])
    m = _affine3(rot) @ scale @ _affine3(affine_px)
    label = cv2.warpAffine(
        label_clean, m[:2], (w, h), flags=cv2.INTER_NEAREST, borderValue=BG
    )
    return Sample(image=image, label=label, meta={"deskew_deg": angle, "page_affine": m[:2].tolist()})


def synth_sample(seed: int, degrade: float, *, dpi: float = PAGE_DPI) -> Sample:
    """One ``kelpod.evaluation.synth`` plan as a training sample."""
    from kelpod.evaluation import synth

    sp = synth.generate(seed, degrade=degrade)
    lab = rasterise_graph_labels(sp.graph, sp.clean_image.shape, sp.m_per_px, sp.wall_mask)
    s = to_page(sp.image, lab, sp.degradation.affine_px, sp.dpi, dpi=dpi)
    s.meta.update(source="synth", seed=seed, degrade=degrade)
    return s


def random_degrade(img: np.ndarray, label: np.ndarray, strength: float, seed: int):
    """Apply the synth generator's scan degradation; returns (degraded image, affine)."""
    from kelpod.evaluation.synth.degrade import degrade

    out, info = degrade(img, float(strength), np.random.default_rng([int(seed), 0x64656772]))
    return out, info.affine_px


def wall_thickness_px(label: np.ndarray) -> float:
    """Median wall thickness of a class map (2 x median distance-transform ridge value)."""
    m = (label == WALL).astype(np.uint8)
    if m.sum() < 50:
        return math.nan
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = d[(d >= cv2.dilate(d, np.ones((3, 3), np.uint8))) & (d > 0.5)]
    return float(2 * np.median(ridge)) if ridge.size else math.nan
