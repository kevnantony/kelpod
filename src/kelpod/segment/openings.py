"""Door and window detection in the gaps between wall runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kelpod.contracts import OpeningKind
from kelpod.segment.walls import H, Run


@dataclass
class Gap:
    orient: str
    c: float
    a0: float
    a1: float
    th: float
    left: Run
    right: Run | None  # collinear partner; None if the gap ends at a perpendicular wall face
    kind: OpeningKind = OpeningKind.UNKNOWN
    hinge: tuple[float, float] | None = None
    swing_into: tuple[float, float] | None = None
    confidence: float = 0.0

    @property
    def width(self) -> float:
        return self.a1 - self.a0

    def to_xy(self, a: float, n: float) -> tuple[float, float]:
        """(along, across) -> image (x, y)."""
        return (a, n) if self.orient == H else (n, a)


def find_gaps(
    walls: list[Run], *, min_w_px: float, max_w_px: float, blockers: list[Run]
) -> list[Gap]:
    """Gaps between a free wall end and the next collinear wall, or the face of a perpendicular
    wall, along the wall axis. ``blockers`` (e.g. window runs) end a search without a gap."""
    gaps: list[Gap] = []
    for w in walls:
        for end in (1, 0):
            if w.meta.get(f"snap{end}"):
                continue
            sign = 1 if end == 1 else -1
            a = w.a1 if end == 1 else w.a0
            best: tuple[float, Run | None, bool] | None = None  # (distance, partner, blocked)
            for o in walls + blockers:
                if o is w:
                    continue
                if o.orient == w.orient:
                    if abs(o.c - w.c) > 0.5 * max(o.th, w.th):
                        continue
                    near = o.a0 if sign > 0 else o.a1
                    d = (near - a) * sign
                    if d < -1:
                        continue
                    cand = (d, o if not o.is_window else None, o.is_window)
                else:
                    if not (o.a0 - 1 <= w.c <= o.a1 + 1):
                        continue
                    face = o.c - sign * o.th / 2
                    d = (face - a) * sign
                    if d < -1:
                        continue
                    cand = (d, None, o.is_window)
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is None:
                continue
            d, partner, blocked = best
            if blocked or not (min_w_px <= d <= max_w_px):
                continue
            if partner is not None and end == 0:
                continue  # the collinear pair is found from the other wall's end 1
            a0, a1 = (a, a + d) if sign > 0 else (a - d, a)
            th = w.th if partner is None else (w.th + partner.th) / 2
            c = w.c if partner is None else (w.c + partner.c) / 2
            gaps.append(Gap(w.orient, c, a0, a1, th, w, partner))
    return gaps


def _sample(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xs = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
    inside = (pts[..., 0] >= 0) & (pts[..., 0] < w) & (pts[..., 1] >= 0) & (pts[..., 1] < h)
    return mask[ys, xs] & inside


def door_arc_score(gap: Gap, ink_d: np.ndarray) -> tuple[float, tuple, tuple] | None:
    """Best quarter-arc fit over the four hinge candidates. Returns (score, hinge, swing_into)."""
    w = gap.width
    thetas = np.deg2rad(np.linspace(8, 82, 24))
    radii = w * np.linspace(0.82, 1.12, 7)
    best = None
    for hinge_a, u in ((gap.a0, 1.0), (gap.a1, -1.0)):
        for side in (-1.0, 1.0):
            hn = gap.c + side * gap.th / 2
            da = u * np.cos(thetas)[:, None] * radii[None, :]
            dn = side * np.sin(thetas)[:, None] * radii[None, :]
            a = hinge_a + da
            n = hn + dn
            pts = np.stack([a, n], -1) if gap.orient == H else np.stack([n, a], -1)
            hit = _sample(ink_d, pts).any(axis=1)
            arc = float(hit.mean())
            # door leaf: line from hinge along the normal
            t = np.linspace(0.15, 0.85, 12) * w
            la = np.full_like(t, hinge_a)
            ln = hn + side * t
            lp = np.stack([la, ln], -1) if gap.orient == H else np.stack([ln, la], -1)
            leaf = float(_sample(ink_d, lp).mean())
            score = 0.75 * arc + 0.25 * leaf
            if best is None or score > best[0]:
                hinge = gap.to_xy(hinge_a, hn)
                swing = gap.to_xy(hinge_a + u * 0.4 * w, hn + side * 0.4 * w)
                best = (score, hinge, swing, arc)
    if best is None:
        return None
    return best[0], best[1], best[2]


def window_lines(gap: Gap, ink: np.ndarray) -> int:
    """Number of thin lines running along the wall across the gap (frame / glazing)."""
    a0, a1 = int(round(gap.a0 + 2)), int(round(gap.a1 - 2))
    n0, n1 = int(round(gap.c - gap.th / 2 - 1)), int(round(gap.c + gap.th / 2 + 1))
    if a1 - a0 < 3 or n1 - n0 < 2:
        return 0
    sub = ink[n0:n1, a0:a1] if gap.orient == H else ink[a0:a1, n0:n1].T
    if sub.size == 0:
        return 0
    frac = sub.mean(axis=1) >= 0.7
    # count groups of consecutive line rows
    return int(np.sum(frac[1:] & ~frac[:-1]) + (1 if frac.size and frac[0] else 0))


def classify_gaps(gaps: list[Gap], ink: np.ndarray, ink_d: np.ndarray, *, door_thr: float = 0.6):
    """Decide door / window / unknown for every gap; returns the kept gaps."""
    kept = []
    for g in gaps:
        nlines = window_lines(g, ink)
        door = door_arc_score(g, ink_d)
        if door is not None and door[0] >= door_thr:
            g.kind = OpeningKind.DOOR
            g.confidence = min(1.0, door[0])
            g.hinge, g.swing_into = door[1], door[2]
        elif nlines >= 2:
            g.kind = OpeningKind.WINDOW
            g.confidence = 0.8
        elif g.right is not None:
            g.kind = OpeningKind.UNKNOWN  # plain wall gap (passage) between collinear walls
            g.confidence = 0.3
        else:
            continue
        kept.append(g)
    return kept
