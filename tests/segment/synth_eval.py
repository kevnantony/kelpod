"""Segmentation-vs-ground-truth check on ``kelpod.evaluation.synth`` plans (image space).

Ground truth (metres, plan space) is mapped into the page that segmentation sees: clean px at the
synth dpi -> degradation affine -> resample to the page dpi -> deskew rotation found by clean.
Run as a script: ``uv run python tests/segment/synth_eval.py --seeds 0 2 4 --degrade 0 0.5``.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from kelpod.contracts import RasterPage, Segmentation, m_to_px


def make_page(sp, dpi: float = 200.0, *, use_clean: bool = True):
    """RasterPage for a SyntheticPlan plus a function mapping GT metres -> page px."""
    from kelpod.clean.core import clean_image
    from kelpod.clean.io import resample

    f = dpi / sp.dpi
    gray = resample(sp.image, sp.dpi, dpi)
    if use_clean:
        res = clean_image(gray, dpi)
        image, ink, angle = res.image, res.ink, res.deskew_deg
    else:
        image, ink, angle = gray, gray < 128, 0.0
    h, w = image.shape
    rot = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -angle, 1.0)
    aff = np.array(sp.degradation.affine_px, float)
    h0 = sp.clean_image.shape[0]

    def to_page(p_m):
        x, y = m_to_px(p_m, sp.m_per_px, h0)
        x, y = aff @ np.array([x, y, 1.0])
        x, y = (x + 0.5) * f - 0.5, (y + 0.5) * f - 0.5
        x, y = rot @ np.array([x, y, 1.0])
        return float(x), float(y)

    page = RasterPage(
        image=image,
        ink=ink,
        dpi=dpi,
        source=f"synth-{sp.seed}",
        deskew_deg=angle,
        plan_scale=sp.plan_scale,
    )
    return page, to_page


def _seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def _covered(src, dst, tol, step=3.0):
    """Share of the length of segments ``src`` lying within ``tol`` of any segment in ``dst``."""
    tot = cov = 0.0
    for a, b in src:
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(2, int(L / step))
        for t in np.linspace(0, 1, n):
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            tot += L / n
            if any(_seg_dist(p, c, d) <= tol for c, d in dst):
                cov += L / n
    return cov / tot if tot else 1.0


def evaluate(sp, seg: Segmentation, to_page, page: RasterPage) -> dict:
    g = sp.graph
    m_per_px = page.m_per_px_from_scale
    tol = 0.10 / m_per_px
    nodes = {n.id: to_page((n.x, n.y)) for n in g.nodes}
    gt_walls = []
    for w in g.walls:  # GT walls minus their opening spans
        a, b = np.array(nodes[w.start]), np.array(nodes[w.end])
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L
        cuts = sorted(
            (o.offset_m / m_per_px, (o.offset_m + o.width_m) / m_per_px)
            for o in g.openings
            if o.wall_id == w.id
        )
        t = 0.0
        for c0, c1 in cuts + [(L, L)]:
            if c0 > t:
                gt_walls.append((tuple(a + u * t), tuple(a + u * c0)))
            t = max(t, c1)
    pr_walls = [(w.p0, w.p1) for w in seg.walls]
    th_err = []
    for w in g.walls:
        a, b = nodes[w.start], nodes[w.end]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        best = min(seg.walls, key=lambda s: _seg_dist(mid, s.p0, s.p1), default=None)
        if best is not None and _seg_dist(mid, best.p0, best.p1) <= tol:
            th_err.append(abs(best.thickness_px * m_per_px - w.thickness_m))
    # openings: centre within 0.25 m, same kind
    walls = {w.id: w for w in g.walls}
    gt_open = []
    for o in g.openings:
        w = walls[o.wall_id]
        a = np.array([g.node(w.start).x, g.node(w.start).y])
        b = np.array([g.node(w.end).x, g.node(w.end).y])
        u = (b - a) / np.linalg.norm(b - a)
        c = a + u * (o.offset_m + o.width_m / 2)
        gt_open.append((o.kind, to_page(tuple(c)), o.width_m))
    used = set()
    hit = {"door": [0, 0], "window": [0, 0]}
    width_err = []
    for kind, c, wm in gt_open:
        k = str(kind)
        hit.setdefault(k, [0, 0])
        hit[k][1] += 1
        for i, o in enumerate(seg.openings):
            oc = ((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2)
            if i in used or np.hypot(oc[0] - c[0], oc[1] - c[1]) > 0.25 / m_per_px:
                continue
            if str(o.kind) == k:
                used.add(i)
                hit[k][0] += 1
                width_err.append(
                    abs(np.hypot(o.p1[0] - o.p0[0], o.p1[1] - o.p0[1]) * m_per_px - wm)
                )
                break
    # rooms: IoU on rasterised polygons, greedy matching
    shape = page.ink.shape

    def raster(poly):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
        return m.astype(bool)

    gt_r = [raster([to_page(p) for p in r.polygon]) for r in g.rooms]
    pr_r = [raster(r.polygon) for r in seg.rooms]
    ious = np.zeros((len(gt_r), len(pr_r)))
    for i, a in enumerate(gt_r):
        for j, b in enumerate(pr_r):
            inter = np.logical_and(a, b).sum()
            ious[i, j] = inter / max(1, np.logical_or(a, b).sum())
    matched = []
    while ious.size and ious.max() >= 0.5:
        i, j = np.unravel_index(int(np.argmax(ious)), ious.shape)
        matched.append(float(ious[i, j]))
        ious[i, :] = 0
        ious[:, j] = 0
    n_open_pred = len(seg.openings)
    return {
        "wall_recall": _covered(gt_walls, pr_walls, tol),
        "wall_precision": _covered(pr_walls, gt_walls, tol),
        "thickness_err_m": float(np.median(th_err)) if th_err else None,
        "door_recall": hit["door"][0] / hit["door"][1] if hit["door"][1] else None,
        "window_recall": hit["window"][0] / hit["window"][1] if hit["window"][1] else None,
        "opening_precision": len(used) / n_open_pred if n_open_pred else 1.0,
        "opening_width_err_m": float(np.median(width_err)) if width_err else None,
        "room_recall": len(matched) / len(gt_r) if gt_r else 1.0,
        "room_precision": len(matched) / len(pr_r) if pr_r else 1.0,
        "room_iou": float(np.mean(matched)) if matched else 0.0,
        "n_gt": (len(g.walls), len(g.openings), len(g.rooms)),
        "n_pred": (len(seg.walls), len(seg.openings), len(seg.rooms)),
    }


def draw(page, seg, to_page, sp, path):
    vis = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    vis[seg.wall_mask] = (0.4 * vis[seg.wall_mask] + [0, 0, 150]).astype(np.uint8)
    g = sp.graph
    for w in g.walls:
        a, b = (
            to_page((g.node(w.start).x, g.node(w.start).y)),
            to_page((g.node(w.end).x, g.node(w.end).y)),
        )
        cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 1)
    for w in seg.walls:
        cv2.line(vis, tuple(map(int, w.p0)), tuple(map(int, w.p1)), (255, 0, 0), 2)
    for o in seg.openings:
        col = {"door": (0, 180, 0), "window": (200, 0, 200)}.get(str(o.kind), (0, 120, 255))
        cv2.line(vis, tuple(map(int, o.p0)), tuple(map(int, o.p1)), col, 4)
    for r in seg.rooms:
        cv2.polylines(vis, [np.round(np.array(r.polygon)).astype(np.int32)], True, (0, 140, 255), 1)
    cv2.imwrite(str(path), vis)


def main():
    from kelpod import segment
    from kelpod.evaluation import synth

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 4, 5, 6, 7])
    ap.add_argument("--degrade", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--vis", default=None)
    a = ap.parse_args()
    for d in a.degrade:
        rows = []
        for s in a.seeds:
            try:
                sp = synth.generate(s, degrade=d)
            except Exception as e:  # generator still in progress for some seeds
                print(f"seed {s}: generator error {e}")
                continue
            page, to_page = make_page(sp)
            t = time.perf_counter()
            seg = segment.run(page)
            dt = time.perf_counter() - t
            r = evaluate(sp, seg, to_page, page)
            r["seconds"] = dt
            rows.append(r)
            print(
                f"degrade={d} seed={s}: "
                + ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                )
            )
            if a.vis:
                draw(page, seg, to_page, sp, f"{a.vis}/seed{s}_d{d}.png")
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)] if rows else []
        print(
            f"MEAN degrade={d}: "
            + ", ".join(
                f"{k}={np.mean([r[k] for r in rows if r[k] is not None]):.3f}" for k in keys
            )
        )


if __name__ == "__main__":
    main()
