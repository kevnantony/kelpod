"""Dev scorer for the hand-labelled real historic plans (3 plans, NOT a benchmark).

Only dev tooling uses this. It reads labels from a directory outside git, runs the pipeline and
reports per plan: room count, Hungarian room IoU (best translation, no rotation), wall F1 at a
tolerance in metres, room closure and scale error. Two IoU numbers: ``iou`` with the scale the
pipeline reports (what a user gets) and ``iou_oracle`` with the label scale (geometry only).

    uv run python scripts/dev/realdev_score.py [--data DIR] [--only NAME] [--tol 0.3] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

DATA = Path.home() / "kelpod-data/real/user-feedback-2026-11-24"
GRID = 0.1  # m per cell for the translation search


def labels_to_m(lab: dict, plan: int | None = None) -> dict:
    """Label JSON (px, y down) -> metres, y up. ``plan`` restricts to one plan of a sheet."""
    s, h = lab["m_per_px"], lab["size"][1]

    def f(p):
        return (p[0] * s, (h - p[1]) * s)

    walls = [(f(w["p"]), f(w["q"]), w["t"] * s) for w in lab["walls"] if plan in (None, w.get("plan"))]
    rooms = [
        (sp["name"], Polygon([f(p) for p in sp["polygon"]]))
        for sp in lab["spaces"]
        if sp["kind"] == "room" and len(sp.get("polygon", [])) >= 3 and plan in (None, sp.get("plan"))
    ]
    return {"walls": walls, "rooms": rooms}


def graph_to_m(graph, factor: float = 1.0) -> dict:
    """PlanGraph -> same structure; ``factor`` rescales (scale-oracle variant)."""
    nodes = {n.id: (n.x * factor, n.y * factor) for n in graph.nodes}
    walls = [(nodes[w.start], nodes[w.end], w.thickness_m * factor) for w in graph.walls]
    rooms = [(r.name or "", Polygon([(x * factor, y * factor) for x, y in r.polygon])) for r in graph.rooms]
    return {"walls": walls, "rooms": [(n, p) for n, p in rooms if p.is_valid and p.area > 0]}


def _raster(walls, ox, oy, shape):
    m = np.zeros(shape, np.float32)
    for p, q, _ in walls:
        a = (int((p[0] - ox) / GRID), int((p[1] - oy) / GRID))
        b = (int((q[0] - ox) / GRID), int((q[1] - oy) / GRID))
        cv2.line(m, a, b, 1.0, 3)
    return m


def best_shift(pred: dict, gt: dict) -> tuple[float, float]:
    """Translation (m) that maps pred onto gt: cross-correlation of wall rasters."""
    if not pred["walls"] or not gt["walls"]:
        return 0.0, 0.0
    allp = [p for w in pred["walls"] + gt["walls"] for p in w[:2]]
    ox, oy = min(p[0] for p in allp) - 1, min(p[1] for p in allp) - 1
    hx, hy = max(p[0] for p in allp) + 1, max(p[1] for p in allp) + 1
    shape = (min(int((hy - oy) / GRID) + 2, 4000), min(int((hx - ox) / GRID) + 2, 4000))
    g = _raster(gt["walls"], ox, oy, shape)
    pr = _raster(pred["walls"], ox, oy, shape)
    ys, xs = np.nonzero(pr)
    if len(xs) == 0:
        return 0.0, 0.0
    t = pr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    # pad gt so the template may hang over the border
    pad = ((t.shape[0],) * 2, (t.shape[1],) * 2)
    gp = np.pad(g, pad)
    res = cv2.matchTemplate(gp, t, cv2.TM_CCORR)
    # normalise by the template overlap with the (padded) image to avoid favouring dense spots
    iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
    dx = (ix - t.shape[1] + 0 - xs.min()) * GRID
    dy = (iy - t.shape[0] + 0 - ys.min()) * GRID
    return dx, dy


def shifted(d: dict, dx: float, dy: float) -> dict:
    from shapely.affinity import translate

    return {
        "walls": [((p[0] + dx, p[1] + dy), (q[0] + dx, q[1] + dy), t) for p, q, t in d["walls"]],
        "rooms": [(n, translate(p, dx, dy)) for n, p in d["rooms"]],
    }


def wall_f1(pred: dict, gt: dict, tol: float) -> tuple[float, float, float]:
    def cover(a, b):  # share of length of a within tol of b
        if not a:
            return 0.0
        if not b:
            return 0.0
        u = unary_union([LineString([p, q]).buffer(tol) for p, q, _ in b])
        tot = sum(LineString([p, q]).length for p, q, _ in a)
        ins = sum(LineString([p, q]).intersection(u).length for p, q, _ in a)
        return ins / tot if tot > 0 else 0.0

    p, r = cover(pred["walls"], gt["walls"]), cover(gt["walls"], pred["walls"])
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def room_iou(pred: dict, gt: dict) -> tuple[float, float, int]:
    """(mean IoU over GT rooms, recall at IoU>=0.5, matched count) by Hungarian assignment."""
    P, G = pred["rooms"], gt["rooms"]
    if not G:
        return 0.0, 0.0, 0
    if not P:
        return 0.0, 0.0, 0
    M = np.zeros((len(G), len(P)))
    for i, (_, g) in enumerate(G):
        for j, (_, p) in enumerate(P):
            u = g.union(p).area
            M[i, j] = g.intersection(p).area / u if u > 0 else 0.0
    r, c = linear_sum_assignment(-M)
    vals = M[r, c]
    return float(vals.sum() / len(G)), float((vals >= 0.5).sum() / len(G)), int((vals >= 0.5).sum())


def score(graph, lab: dict, plan: int | None, m_per_orig_px: float, tol: float = 0.3) -> dict:
    gt = labels_to_m(lab, plan)
    out = {"gt_rooms": len(gt["rooms"]), "pred_rooms": len(graph.rooms)}
    ratio = m_per_orig_px / lab["m_per_px"]  # pred scale error factor
    out["scale_err_pct"] = round(abs(ratio - 1) * 100, 1)
    out["scale_source"] = graph.scale_source
    rep = graph.report
    out["room_closure"] = round(rep.room_closure, 2) if rep else None
    out["dangling"] = len(rep.dangling_nodes) if rep else None
    for tag, fac in (("", 1.0), ("_oracle", 1.0 / ratio if ratio > 0 else 1.0)):
        pred = graph_to_m(graph, fac)
        dx, dy = best_shift(pred, gt)
        pred = shifted(pred, dx, dy)
        f1, p, r = wall_f1(pred, gt, tol)
        iou, rec, n = room_iou(pred, gt)
        out.update({f"wall_f1{tag}": round(f1, 2), f"wall_p{tag}": round(p, 2), f"wall_r{tag}": round(r, 2),
                    f"iou{tag}": round(iou, 2), f"room_recall{tag}": round(rec, 2)})
    return out


def run_pipeline(path: Path, all_plans: bool = False):
    """Returns [(PlanGraph, m_per_orig_px)] for the plans the pipeline produced."""
    from kelpod import pipeline

    if hasattr(pipeline, "convert_all"):
        results = pipeline.convert_all(path, None, all_plans=all_plans)
    else:
        results = [pipeline.convert(path, None)]
    from kelpod import clean

    page = clean.run(path)[0]
    w0 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape[1]
    out = []
    for r in results:
        f = page.width / w0
        out.append((r.graph, r.graph.m_per_px * f))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--only")
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--all-plans", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    rows = {}
    for lp in sorted((a.data / "labels").glob("*.json")):
        if a.only and a.only not in lp.stem:
            continue
        lab = json.loads(lp.read_text())
        graphs = run_pipeline(a.data / lab["image"], a.all_plans)
        plans = [p["n"] for p in lab["plans"] if p.get("labelled")] or [None]
        for pl in plans:
            best = None
            for g, mpp in graphs:
                s = score(g, lab, pl, mpp, a.tol)
                if best is None or s["iou_oracle"] + s["wall_f1_oracle"] > best["iou_oracle"] + best["wall_f1_oracle"]:
                    best = s
            best["n_graphs"] = len(graphs)
            rows[f"{lp.stem}{'' if pl is None else f'#plan{pl}'}"] = best
    for k, v in rows.items():
        print(k)
        print("  " + ", ".join(f"{a_}={b}" for a_, b in v.items()))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
