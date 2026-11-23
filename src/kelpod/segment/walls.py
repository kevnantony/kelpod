"""Wall mask and wall vectorisation for rectilinear (deskewed) plans.

1. :func:`wall_body_mask`: close the ink a little, fill *narrow enclosed* background (the cells
   between wall outlines and hatch strokes, max inscribed radius below ``max_fill_radius``), open by
   about half the thinnest wall to remove thin lines (dimension chains, door arcs, text strokes),
   drop small components (glyph blobs).
2. :func:`extract_runs`: per orientation, open with a line kernel longer than the thickest wall, so
   only that orientation survives; walk each component along its axis, reading the thickness and
   centre of every column, and split into runs where thickness / centre change or where the
   interior texture changes (hatched, empty, a continuous centre line = window glazing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

H, V = "h", "v"


@dataclass
class Run:
    """A straight axis-aligned piece of the wall mask.

    For ``orient == "h"`` the run spans x in ``[a0, a1]`` at y = ``c``; for ``"v"`` it spans y in
    ``[a0, a1]`` at x = ``c`` (image px, pixel edges)."""

    orient: str
    c: float
    a0: float
    a1: float
    th: float
    label: str  # "H" hatched/solid, "E" empty interior (outline only), "L" centre line (glazing)
    hatched_frac: float = 0.0
    comp: int = -1  # component index within its orientation (runs of a component are ordered)
    order: int = 0
    is_window: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.a1 - self.a0

    def p0(self) -> tuple[float, float]:
        return (self.a0, self.c) if self.orient == H else (self.c, self.a0)

    def p1(self) -> tuple[float, float]:
        return (self.a1, self.c) if self.orient == H else (self.c, self.a1)


# ----------------------------------------------------------------------------------------------
# Wall body mask
# ----------------------------------------------------------------------------------------------


def fill_narrow_holes(ink: np.ndarray, max_radius_px: float) -> np.ndarray:
    """Ink plus every enclosed background region whose largest inscribed circle has radius
    <= ``max_radius_px`` (cells between outlines and hatching, glyph counters)."""
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bg = (closed == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    if n <= 1:
        return closed.astype(bool)
    dt = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
    maxdt = ndimage.maximum(dt, lab, index=np.arange(n))
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = maxdt <= max_radius_px
    small[0] = False  # label 0 is the ink itself
    small[border] = False
    return closed.astype(bool) | small[lab]


def wall_body_mask(
    ink: np.ndarray, *, max_radius_px: float, open_px: int, min_extent_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(filled, body)``: the hole-filled ink and the opened, size-filtered wall bodies."""
    filled = fill_narrow_holes(ink, max_radius_px)
    k = max(3, int(open_px)) | 1  # odd: even kernels shift the result by a pixel
    body = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
    keep = np.zeros(n, bool)
    ext = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep[1:] = ext[1:] >= min_extent_px
    return filled, keep[lab]


def estimate_max_thickness(body: np.ndarray) -> float:
    """Thickness of the thickest common wall: 95th percentile of twice the distance transform on
    its ridge (medial axis) pixels, so junction squares and odd blobs do not dominate."""
    if not body.any():
        return 0.0
    dt = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)
    vals = dt[ridge] if ridge.any() else dt[dt > 0]
    return float(2.0 * np.percentile(vals, 95))


# ----------------------------------------------------------------------------------------------
# Runs along one orientation
# ----------------------------------------------------------------------------------------------


def _split_runs(thick: np.ndarray, centre: np.ndarray, tol_frac: float) -> list[tuple[int, int]]:
    """Greedy split of a column profile into runs of constant thickness and centre."""
    runs: list[tuple[int, int]] = []
    start = 0
    ref_t, ref_c, cnt = float(thick[0]), float(centre[0]), 1
    for i in range(1, len(thick)):
        tol = max(2.0, tol_frac * ref_t)
        if abs(thick[i] - ref_t) > tol or abs(centre[i] - ref_c) > tol:
            runs.append((start, i))
            start, ref_t, ref_c, cnt = i, float(thick[i]), float(centre[i]), 1
        else:
            cnt += 1
            ref_t += (thick[i] - ref_t) / cnt
            ref_c += (centre[i] - ref_c) / cnt
    runs.append((start, len(thick)))
    return runs


def _smooth_labels(labels: list[str], min_len: int) -> list[str]:
    """Absorb label runs shorter than ``min_len`` into the longer neighbour."""
    if not labels:
        return labels
    labels = list(labels)
    for _ in range(3):
        segs: list[list] = []
        for lab in labels:
            if segs and segs[-1][0] == lab:
                segs[-1][1] += 1
            else:
                segs.append([lab, 1])
        if len(segs) == 1:
            break
        changed = False
        for i, (_lab, n) in enumerate(segs):
            if n < min_len:
                left = segs[i - 1] if i > 0 else None
                right = segs[i + 1] if i + 1 < len(segs) else None
                nb = max((s for s in (left, right) if s is not None), key=lambda s: s[1])
                segs[i][0] = nb[0]
                changed = True
        labels = [lab for lab, n in segs for _ in range(n)]
        if not changed:
            break
    return labels


def _column_labels(band: np.ndarray, th: float) -> tuple[list[str], float]:
    """Label every column of a wall band (rows across the wall, columns along it).

    Returns per-column labels in {"H", "E", "L"} and the width (px) of the face outlines
    (0 if it could not be measured, e.g. solid poché)."""
    rows, cols = band.shape
    if rows < 3 or cols == 0:
        return ["E"] * cols, 0.0
    frac = band.mean(axis=1)
    if (frac >= 0.75).all():  # solid poché
        return ["H"] * cols, 0.0
    top, bot = 0, rows
    while top < rows and frac[top] >= 0.6:
        top += 1
    while bot > top and frac[bot - 1] >= 0.6:
        bot -= 1
    face_lw = (top + (rows - bot)) / 2.0
    top, bot = top + 1, bot - 1
    if bot - top < 2:
        return ["E"] * cols, face_lw
    inner = band[top:bot].astype(np.float32)
    w = max(5, int(round(th)))
    local = ndimage.uniform_filter1d(inner, size=w, axis=1, mode="nearest")  # rows x cols
    lmax = local.max(axis=0)
    lmed = np.median(local, axis=0)
    dens = ndimage.uniform_filter1d(inner.mean(axis=0), size=5, mode="nearest")
    labels = []
    for j in range(cols):
        if lmax[j] >= 0.8 and lmed[j] <= 0.5:
            labels.append("L")
        elif dens[j] >= 0.04:
            labels.append("H")
        else:
            labels.append("E")
    labels = _smooth_labels(labels, max(4, int(0.6 * th)))
    return _snap_to_jambs(labels, band, th), face_lw


def _snap_to_jambs(labels: list[str], band: np.ndarray, th: float) -> list[str]:
    """Move each label boundary onto a nearby jamb line (a column inked across the band)."""
    colfrac = band.mean(axis=0)
    jamb = colfrac >= 0.8
    out = list(labels)
    r = max(3, int(th / 2))
    for j in range(1, len(labels)):
        if labels[j] == labels[j - 1]:
            continue
        lo, hi = max(0, j - r), min(len(labels), j + r)
        idx = np.nonzero(jamb[lo:hi])[0] + lo
        if idx.size == 0:
            continue
        k = int(idx[np.argmin(np.abs(idx - j))])
        # centre of that jamb line
        k0 = k1 = k
        while k0 > 0 and jamb[k0 - 1]:
            k0 -= 1
        while k1 + 1 < len(jamb) and jamb[k1 + 1]:
            k1 += 1
        b = (k0 + k1 + 1) // 2
        left, right = labels[j - 1], labels[j]
        for t in range(min(b, j), max(b, j)):
            out[t] = left if t < b else right
    return out


def _absorb_protrusions(geo: list[list[float]], tol: float = 1.5) -> list[list[float]]:
    """``geo`` = ordered ``[s, e, th, c, top, bot]`` runs of one component. A short run whose
    band contains its neighbour's band (the neighbour plus something touching one face: a door
    leaf or arc foot, a T-junction bump) is absorbed into that neighbour."""
    changed = True
    while changed and len(geo) > 1:
        changed = False
        for i, r in enumerate(geo):
            if r[1] - r[0] >= max(4.0, 1.0 * r[2]):
                continue
            for j in (i - 1, i + 1):
                if not 0 <= j < len(geo):
                    continue
                nb = geo[j]
                if nb[1] - nb[0] <= r[1] - r[0]:
                    continue
                if r[4] <= nb[4] + tol and r[5] >= nb[5] - tol:
                    nb[0], nb[1] = min(nb[0], r[0]), max(nb[1], r[1])
                    del geo[i]
                    changed = True
                    break
            if changed:
                break
    return geo


def extract_runs(
    body: np.ndarray,
    ink: np.ndarray,
    orient: str,
    *,
    kernel_len: int,
    min_th_px: float,
    min_len_px: float,
) -> list[Run]:
    """Runs of one orientation. ``body``/``ink`` are image-space masks; for ``"v"`` they are
    transposed internally so the same code walks along the run."""
    m = body if orient == H else body.T
    k = ink if orient == H else ink.T
    lines = cv2.morphologyEx(
        m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(3, kernel_len) | 1), np.uint8)
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    out: list[Run] = []
    for ci in range(1, n):
        x, y, w, h = stats[ci, :4]
        sub = lab[y : y + h, x : x + w] == ci
        has = sub.any(axis=0)
        if not has.any():
            continue
        top = np.argmax(sub, axis=0)
        bot = h - 1 - np.argmax(sub[::-1], axis=0)
        cols = np.nonzero(has)[0]
        thick = (bot - top + 1)[cols].astype(np.float64)
        centre = ((top + bot + 1) / 2.0)[cols].astype(np.float64)
        if len(cols) >= 5:
            thick = ndimage.median_filter(thick, size=5, mode="nearest")
            centre = ndimage.median_filter(centre, size=5, mode="nearest")
        geo = []
        for s0, e0 in _split_runs(thick, centre, 0.15):
            th = float(np.median(thick[s0:e0]))
            c = float(np.median(centre[s0:e0]))
            geo.append([s0, e0, th, c, c - th / 2, c + th / 2])
        geo = _absorb_protrusions(geo)
        order = 0
        for s0, e0, th, c_loc, _, _ in geo:
            if th < min_th_px:
                continue
            a0, a1 = float(x + cols[s0]), float(x + cols[e0 - 1] + 1)
            if a1 - a0 < max(3.0, 0.3 * th):
                continue
            c = float(y + c_loc)
            r0, r1 = int(round(c - th / 2)), int(round(c + th / 2))
            band = k[max(0, r0) : max(0, r1), int(a0) : int(a1)]
            labels, face_lw = _column_labels(band, th)
            # split the geometric run further by texture label
            j = 0
            while j < len(labels):
                jj = j
                while jj < len(labels) and labels[jj] == labels[j]:
                    jj += 1
                run = Run(
                    orient,
                    c,
                    a0 + j,
                    a0 + jj,
                    th,
                    labels[j],
                    hatched_frac=1.0 if labels[j] == "H" else 0.0,
                    comp=ci,
                    order=order,
                    meta={"face_lw": face_lw},
                )
                order += 1
                out.append(run)
                j = jj
    # drop runs too short to be anything but noise (keep short ones inside a component's sequence
    # so neighbours stay adjacent; they get merged or dropped later)
    return [r for r in out if r.length >= min(min_len_px, 0.3 * r.th) or r.label == "L"]


def _close(a: Run, b: Run, tol: float) -> bool:
    return abs(a.c - b.c) <= tol and abs(a.th - b.th) <= tol


def classify_windows(
    runs: list[Run], *, min_w_px: float, max_w_px: float, min_ext_th_px: float
) -> None:
    """Mark window runs in place: a glazing centre line ("L"), or an empty outline ("E") set
    collinearly between two hatched runs of a thick wall (two-line window)."""
    by_comp: dict[tuple[str, int], list[Run]] = {}
    for r in runs:
        by_comp.setdefault((r.orient, r.comp), []).append(r)
    for seq in by_comp.values():
        seq.sort(key=lambda r: r.a0)
        for i, r in enumerate(seq):
            if not (min_w_px <= r.length <= max_w_px):
                continue
            left = seq[i - 1] if i > 0 else None
            right = seq[i + 1] if i + 1 < len(seq) else None
            if r.label == "L":
                r.is_window = True
            elif r.label == "E" and left is not None and right is not None:
                if (
                    left.label == "H"
                    and right.label == "H"
                    and min(left.th, right.th) >= min_ext_th_px
                    and r.th <= 1.15 * max(left.th, right.th)
                    and abs(left.c - right.c) <= 0.25 * left.th
                ):
                    r.is_window = True


def merge_walls(runs: list[Run], *, max_gap_px: float) -> list[Run]:
    """Merge consecutive collinear non-window runs (same orientation, centre and thickness)."""
    out: list[Run] = []
    for orient in (H, V):
        rs = sorted((r for r in runs if r.orient == orient and not r.is_window), key=lambda r: r.a0)
        merged: list[Run] = []
        for r in rs:
            tol = max(2.0, 0.2 * r.th)
            target = None
            for m in merged:
                if _close(m, r, tol) and r.a0 - m.a1 <= max_gap_px and r.a1 > m.a0:
                    target = m
                    break
            if target is None:
                merged.append(Run(r.orient, r.c, r.a0, r.a1, r.th, r.label, r.hatched_frac, r.comp))
                continue
            la, lb = target.length, r.length
            tot = la + lb
            target.c = (target.c * la + r.c * lb) / tot
            target.th = (target.th * la + r.th * lb) / tot
            target.hatched_frac = (target.hatched_frac * la + r.hatched_frac * lb) / tot
            target.a0 = min(target.a0, r.a0)
            target.a1 = max(target.a1, r.a1)
            target.label = "H" if target.hatched_frac >= 0.5 else "E"
        out.extend(merged)
    return out


def snap_junctions(walls: list[Run], tol_px: float) -> None:
    """Move wall ends that stop inside (or at the face of) a perpendicular wall onto that wall's
    centreline, so centrelines meet at nodes. Marks snapped ends in ``meta``."""
    for w in walls:
        w.meta.setdefault("snap0", False)
        w.meta.setdefault("snap1", False)
    for w in walls:
        for end in (0, 1):
            a = w.a0 if end == 0 else w.a1
            best = None
            for p in walls:
                if p.orient == w.orient:
                    continue
                half = p.th / 2 + tol_px
                if abs(a - p.c) > half:
                    continue
                if not (p.a0 - w.th / 2 - tol_px <= w.c <= p.a1 + w.th / 2 + tol_px):
                    continue
                d = abs(a - p.c)
                if best is None or d < best[0]:
                    best = (d, p)
            if best is not None:
                p = best[1]
                if end == 0:
                    w.a0 = p.c
                else:
                    w.a1 = p.c
                w.meta[f"snap{end}"] = True
                # extend the perpendicular wall if it stops short of this centreline
                if w.c < p.a0 and p.a0 - w.c <= w.th / 2 + tol_px:
                    p.a0 = w.c
                elif w.c > p.a1 and w.c - p.a1 <= w.th / 2 + tol_px:
                    p.a1 = w.c
