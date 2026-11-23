"""Hand-built Segmentation / OcrAnnotations fixtures for the geometry tests.

Houses are described in plan metres (origin ``ORIGIN`` inside the page) and converted to image
pixels, optionally with noise, rotation and per-axis distortion, so the tests do not depend on the
segment and ocr modules.
"""

from __future__ import annotations

import math

import numpy as np

from kelpod.contracts import (
    DimensionLabel,
    OcrAnnotations,
    OpeningKind,
    OpeningPx,
    RasterPage,
    RoomRegion,
    RoomStamp,
    Segmentation,
    SillHeight,
    WallKind,
    WallSegment,
)

DPI = 200.0
M_PER_PX = 0.0254 / DPI * 100  # 1:100 at 200 dpi
H = W = 1200
ORIGIN = (2.0, 2.0)
EXT, INT = 0.365, 0.24


def page(plan_scale: float | None = 100.0, h: int = H, w: int = W) -> RasterPage:
    img = np.full((h, w), 255, np.uint8)
    return RasterPage(image=img, ink=img < 128, dpi=DPI, source="fixture", plan_scale=plan_scale)


class Drawer:
    """Maps plan metres (house frame) to image px with optional distortion and rotation."""

    def __init__(self, m_per_px=M_PER_PX, sx=1.0, sy=1.0, rot_deg=0.0, noise_px=0.0, seed=0):
        self.m = m_per_px
        self.sx, self.sy = sx, sy
        self.rot = math.radians(rot_deg)
        self.noise = noise_px
        self.rng = np.random.default_rng(seed)
        self._wid = 0
        self._oid = 0

    def xform(self, x: float, y: float) -> tuple[float, float]:
        """Distortion is applied in the house frame (drawn length = true length * s)."""
        return (x * self.sx, y * self.sy)

    def px(self, p, noise=True):
        x, y = self.xform(*p)
        x, y = x + ORIGIN[0], y + ORIGIN[1]
        u, v = x / self.m, H - y / self.m
        if self.rot:
            c, s = math.cos(self.rot), math.sin(self.rot)
            du, dv = u - W / 2, v - H / 2
            u, v = W / 2 + c * du - s * dv, H / 2 + s * du + c * dv
        if noise and self.noise:
            u += self.rng.normal(0, self.noise)
            v += self.rng.normal(0, self.noise)
        return (float(u), float(v))

    def wall(self, p0, p1, t, kind=WallKind.EXTERIOR, shrink=0.0):
        """``shrink`` > 0 shortens each end by that many metres (a skeleton stopping short),
        < 0 overshoots."""
        d = np.subtract(p1, p0)
        u = d / np.linalg.norm(d)
        a, b = np.add(p0, u * shrink), np.subtract(p1, u * shrink)
        tp = t * math.sqrt(self.sx * self.sy) / self.m
        if self.noise:
            tp += self.rng.normal(0, self.noise / 2)
        w = WallSegment(id=self._wid, p0=self.px(a), p1=self.px(b), thickness_px=tp, kind=kind)
        self._wid += 1
        return w

    def opening(self, kind, p0, p1):
        o = OpeningPx(id=self._oid, kind=kind, p0=self.px(p0, False), p1=self.px(p1, False))
        self._oid += 1
        return o

    def dim(self, value, p0, p1, text=None):
        a, b = self.px(p0, False), self.px(p1, False)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ang = 0.0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 90.0
        return DimensionLabel(
            value_m=value,
            text=text or f"{value:.3f}".replace(".", ","),
            anchor=mid,
            angle_deg=ang,
            line=(a, b),
        )

    def room(self, rid, xs, ys):
        pts = [
            self.px((x, y), False)
            for x, y in [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
        ]
        seed = self.px(((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2), False)
        area = abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[1][1]))
        return RoomRegion(id=rid, polygon=pts, seed=seed, area_px=area)


def two_room_house(dr: Drawer, shrink=0.0):
    """8 x 6 m (centrelines) exterior, interior wall at x = 4 (two T-junctions)."""
    s = shrink
    walls = [
        dr.wall((0, 0), (8, 0), EXT, shrink=s),
        dr.wall((8, 0), (8, 6), EXT, shrink=-s),
        dr.wall((8, 6), (0, 6), EXT, shrink=s),
        dr.wall((0, 6), (0, 0), EXT, shrink=-s),
        dr.wall((4, 0), (4, 6), INT, WallKind.INTERIOR, shrink=s),
    ]
    rooms = [dr.room(0, (0.2, 3.9), (0.2, 5.8)), dr.room(1, (4.1, 7.8), (0.2, 5.8))]
    return walls, rooms


def face_dims(dr: Drawer):
    """Dimension chains measured to the raw wall faces (true values; ticks where drawn)."""
    e, i = EXT / 2, INT / 2
    return [
        dr.dim(8 + EXT, (-e, -1.0), (8 + e, -1.0)),
        dr.dim(4 - e - i, (e, -0.7), (4 - i, -0.7)),
        dr.dim(4 - e - i, (4 + i, -0.7), (8 - e, -0.7)),
        dr.dim(6 + EXT, (-1.0, -e), (-1.0, 6 + e)),
        dr.dim(6 - EXT, (-0.7, e), (-0.7, 6 - e)),
    ]


def seg(walls, openings=(), rooms=()):
    return Segmentation(
        wall_mask=np.zeros((H, W), bool),
        walls=list(walls),
        openings=list(openings),
        rooms=list(rooms),
    )


def ocr(dims=(), stamps=(), sills=()):
    return OcrAnnotations(dimensions=list(dims), room_stamps=list(stamps), sill_heights=list(sills))


def stamp(dr: Drawer, name, area, at):
    return RoomStamp(name=name, area_m2=area, text=f"{name} {area}", anchor=dr.px(at, False))


def sill(dr: Drawer, value, at):
    return SillHeight(value_m=value, text=f"BRH {value}", anchor=dr.px(at, False))


__all__ = [
    "EXT",
    "INT",
    "Drawer",
    "OpeningKind",
    "face_dims",
    "ocr",
    "page",
    "seg",
    "sill",
    "stamp",
    "two_room_house",
]


def inputs_from_synth(plan, rng, noise_px=1.5, distort=None):
    """Simulate segment + ocr output from a synthetic plan's ground truth: walls cut at openings,
    ends stopping short or overshooting by up to ~18 cm, endpoint noise, thickness noise, and the
    plan's own dimension / BRH / stamp labels. ``distort(px) -> px`` warps the drawing (labels keep
    their true values)."""
    from shapely.geometry import Polygon

    from kelpod.contracts import RoomStamp, SillHeight, m_to_px

    f = distort or (lambda q: q)
    g, m, h = plan.graph, plan.m_per_px, plan.image.shape[0]
    pos = {n.id: np.array((n.x, n.y)) for n in g.nodes}

    def px(q):
        return f(m_to_px((float(q[0]), float(q[1])), m, h))

    walls, opens = [], []
    for w in g.walls:
        a, b = pos[w.start], pos[w.end]
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        s = 0.0
        cuts = sorted(
            (o.offset_m, o.offset_m + o.width_m, o) for o in g.openings if o.wall_id == w.id
        )
        for lo, hi, o in cuts:
            opens.append(
                OpeningPx(id=len(opens), kind=o.kind, p0=px(a + u * lo), p1=px(a + u * hi))
            )
            if lo - s > 0.05:
                walls.append((a + u * s, a + u * lo, w))
            s = hi
        if length - s > 0.05:
            walls.append((a + u * s, a + u * length, w))
    segs = []
    for i, (p0, p1, w) in enumerate(walls):
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        p0 = p0 + u * rng.uniform(-0.12, 0.18)
        p1 = p1 - u * rng.uniform(-0.12, 0.18)
        q0, q1 = px(p0), px(p1)
        segs.append(
            WallSegment(
                id=i,
                p0=(q0[0] + rng.normal(0, noise_px), q0[1] + rng.normal(0, noise_px)),
                p1=(q1[0] + rng.normal(0, noise_px), q1[1] + rng.normal(0, noise_px)),
                thickness_px=w.thickness_m / m + rng.normal(0, noise_px / 2),
                kind=w.kind,
            )
        )
    stamps = [
        RoomStamp(
            name=r.name or "Raum",
            area_m2=r.stamp_area_m2 or r.area_m2,
            text="",
            anchor=px(Polygon(r.polygon).representative_point().coords[0]),
        )
        for r in g.rooms
    ]
    dims = [
        DimensionLabel(value_m=a.value, text=a.text, anchor=f(a.center), line=tuple(map(f, a.line)))
        for a in plan.annotations
        if a.kind == "dimension" and a.line
    ]
    sills = [
        SillHeight(value_m=a.value, text=a.text, anchor=f(a.center))
        for a in plan.annotations
        if a.kind == "sill"
    ]
    img = np.full(plan.image.shape, 255, np.uint8)
    pg = RasterPage(image=img, ink=img < 128, dpi=plan.dpi, source="synth", plan_scale=100)
    sg = Segmentation(wall_mask=img < 0, walls=segs, openings=opens)
    oc = OcrAnnotations(dimensions=dims, sill_heights=sills, room_stamps=stamps)
    return pg, sg, oc
