"""Tiny DIN-style plan renderer for segmentation tests (cv2 only, image space in metres).

Coordinates here are metres from the top-left of the drawing area, y down; ``to_px`` adds the
margin. Ground truth is returned in image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from kelpod.contracts import RasterPage


@dataclass
class W:
    x0: float
    y0: float
    x1: float
    y1: float  # centreline, axis-aligned
    th: float
    hatched: bool = True

    def rect(self):
        h = self.th / 2
        return box(
            min(self.x0, self.x1) - h, min(self.y0, self.y1) - h,
            max(self.x0, self.x1) + h, max(self.y0, self.y1) + h,
        )


@dataclass
class O:
    wall: int
    t0: float  # offset of the near edge along the wall from (x0, y0)
    width: float
    kind: str  # "door" | "window2" | "window3" | "gap"
    hinge_end: int = 0  # door: 0 = hinge at t0, 1 = at t0 + width
    side: int = 1  # door swing side: +1 = +normal


@dataclass
class Plan:
    walls: list[W]
    openings: list[O] = field(default_factory=list)
    rooms: list[tuple[float, float, str]] = field(default_factory=list)  # label positions
    dims: bool = True


def render(plan: Plan, *, dpi: float = 200, size_m=(12.0, 10.0), margin_m: float = 1.5):
    mpp = 0.0254 / dpi * 100
    W_ = int((size_m[0] + 2 * margin_m) / mpp)
    H_ = int((size_m[1] + 2 * margin_m) / mpp)
    img = np.full((H_, W_), 255, np.uint8)

    def P(x, y):
        return (int(round((x + margin_m) / mpp)), int(round((y + margin_m) / mpp)))

    lw_thick = max(2, int(round(0.0005 / 0.0254 * dpi)))  # 0.5 mm pen
    lw_thin = max(1, int(round(0.00025 / 0.0254 * dpi)))  # 0.25 mm pen

    def unit(w: W):
        d = np.array([w.x1 - w.x0, w.y1 - w.y0], float)
        L = float(np.linalg.norm(d))
        return d / L, np.array([-d[1], d[0]]) / L, L

    cuts = []
    gt_openings = []
    for o in plan.openings:
        w = plan.walls[o.wall]
        u, n, _ = unit(w)
        a = np.array([w.x0, w.y0]) + u * o.t0
        b = a + u * o.width
        hn = n * (w.th / 2 + 0.002)
        cuts.append(Polygon([a - hn, b - hn, b + hn, a + hn]))
        gt_openings.append((o, a, b))
    cut = unary_union(cuts) if cuts else Polygon()

    hatched = unary_union([w.rect() for w in plan.walls if w.hatched]).difference(cut)
    light = unary_union([w.rect() for w in plan.walls if not w.hatched])
    light = light.difference(hatched.buffer(0)).difference(cut)

    def polys(g):
        return [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))

    def draw_outline(g, lw):
        for poly in polys(g):
            if poly.is_empty:
                continue
            for ring in [poly.exterior, *poly.interiors]:
                pts = np.array([P(x, y) for x, y in ring.coords], np.int32)
                cv2.polylines(img, [pts], True, 0, lw, cv2.LINE_8)

    # hatching: 45 deg lines every 1.2 mm on paper, clipped to the masonry region
    region = np.zeros_like(img)
    for poly in polys(hatched):
        pts = np.array([P(x, y) for x, y in poly.exterior.coords], np.int32)
        cv2.fillPoly(region, [pts], 255)
    hatch = np.full_like(img, 255)
    step = int(round(0.0012 / 0.0254 * dpi))
    for k in range(-H_, W_, step):
        cv2.line(hatch, (k, 0), (k + H_, H_), 0, lw_thin)
    img[(region > 0) & (hatch == 0)] = 0
    draw_outline(hatched, lw_thick)
    draw_outline(light, lw_thick)

    for o, a, b in gt_openings:
        w = plan.walls[o.wall]
        u, n, _ = unit(w)
        if o.kind.startswith("window"):
            offs = [-w.th / 2, w.th / 2] if o.kind == "window2" else [-w.th / 2, 0, w.th / 2]
            for t in offs:
                cv2.line(img, P(*(a + n * t)), P(*(b + n * t)), 0, lw_thin)
        elif o.kind == "door":
            hinge = (a if o.hinge_end == 0 else b) + n * o.side * w.th / 2
            other = (b if o.hinge_end == 0 else a) + n * o.side * w.th / 2
            leaf_end = hinge + n * o.side * o.width
            cv2.line(img, P(*hinge), P(*leaf_end), 0, lw_thin + 1)
            r = o.width
            v0 = other - hinge
            v1 = leaf_end - hinge
            angs = np.linspace(0, 1, 60)
            a0 = np.arctan2(v0[1], v0[0])
            a1 = np.arctan2(v1[1], v1[0])
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            pts = np.array(
                [P(*(hinge + r * np.array([np.cos(a0 + da * t), np.sin(a0 + da * t)])))
                 for t in angs],
                np.int32,
            )
            cv2.polylines(img, [pts], False, 0, lw_thin)

    # room labels and a dimension chain under the plan (distractors)
    for x, y, text in plan.rooms:
        cv2.putText(img, text, P(x - 0.6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        cv2.putText(img, "14,52 m2", P(x - 0.6, y + 0.4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    if plan.dims:
        xs = sorted({w.x0 for w in plan.walls} | {w.x1 for w in plan.walls})
        ymax = max(max(w.y0, w.y1) + w.th / 2 for w in plan.walls)
        yd = ymax + 0.8
        cv2.line(img, P(xs[0], yd), P(xs[-1], yd), 0, lw_thin)
        for x in xs:
            cv2.line(img, P(x - 0.08, yd + 0.08), P(x + 0.08, yd - 0.08), 0, lw_thin + 1)
            cv2.line(img, P(x, ymax + 0.3), P(x, yd + 0.15), 0, lw_thin)
        for xa, xb in zip(xs[:-1], xs[1:], strict=False):
            cv2.putText(img, f"{xb - xa:.2f}".replace(".", ","), P((xa + xb) / 2 - 0.3, yd - 0.1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    gt = {
        "walls": [
            (P(w.x0, w.y0), P(w.x1, w.y1), w.th / mpp, w.hatched) for w in plan.walls
        ],
        "openings": [
            (o.kind, P(*a), P(*b), o.wall) for o, a, b in gt_openings
        ],
        "m_per_px": mpp,
    }
    return img, gt


def page_from(img: np.ndarray, dpi: float = 200) -> RasterPage:
    return RasterPage(image=img, ink=img < 128, dpi=dpi, source="fixture", plan_scale=100.0)


def house() -> Plan:
    """8.0 x 6.0 m (centrelines) house: hatched 36.5 exterior, hatched 24 interior wall at x=4.5,
    an 11.5 partition at y=3.0 in the right part; 3 doors, 3 windows, 3 rooms."""
    ext = 0.365
    walls = [
        W(0, 0, 8, 0, ext),  # 0 top
        W(8, 0, 8, 6, ext),  # 1 right
        W(8, 6, 0, 6, ext),  # 2 bottom (runs right to left)
        W(0, 6, 0, 0, ext),  # 3 left
        W(4.5, 0, 4.5, 6, 0.24),  # 4 interior
        W(4.5, 3.0, 8, 3.0, 0.115, hatched=False),  # 5 partition
    ]
    openings = [
        O(0, 1.2, 1.51, "window3"),
        O(1, 0.9, 1.26, "window2"),
        O(2, 5.5, 1.01, "door", hinge_end=0, side=1),  # entrance, bottom wall, swings in
        O(4, 1.0, 0.885, "door", hinge_end=1, side=1),
        O(5, 1.2, 0.76, "door", hinge_end=0, side=-1),
        O(3, 2.0, 1.26, "window3"),
    ]
    rooms = [(2.2, 3.0, "Wohnen"), (6.2, 1.5, "Kind"), (6.2, 4.5, "Bad")]
    return Plan(walls, openings, rooms)
