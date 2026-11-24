"""Dev measurement for the dimension-chain decoder (exp/ocr-chainsum). Not a test, not a locked run.

For each synthetic dev plan (``kelpod.evaluation.synth``, train style): save the PDF, run clean,
segment and the OCR *reading* exactly as the pipeline does, and cache the result (reading is the
slow part). Then, per decode mode (``greedy`` = v0, ``chain`` = joint chain decoding), interpret
the cached reading, score the dimension labels against the drawn annotations and run geometry +
the §3 panel against the ground truth graph.

    uv run python tests/ocr/chain_eval.py --seeds 200000-200099 --degrade 0.5 1.0 --workers 4

Dimension metrics (per plan, summed over plans):

* a predicted label is a true positive if an unmatched drawn label has the same value (±0.5 mm)
  and either its text centre lies within 0.6 m (plan) of the predicted anchor or both span ends
  agree within 0.05 m; matching is greedy by distance, one-to-one;
* precision = TP / predicted, recall = TP / drawn;
* linked = TP whose span (``line``) ends lie within 0.05 m of the drawn ticks; ``linked_recall``
  = linked / drawn, ``line_precision`` = linked / predicted labels with a span.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import statistics
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

CACHE = Path(os.environ.get("KELPOD_DEV_CACHE", Path.home() / "kelpod-data/dev-cache/ocr-chainsum"))
DPI = 200.0
VALUE_TOL = 0.0005
ANCHOR_TOL_M = 0.6
SPAN_TOL_M = 0.05


def _key(seed: int, degrade: float) -> Path:
    return CACHE / f"s{seed}_d{degrade:g}.pkl"


def prepare(args: tuple[int, float]) -> str:
    """Generate, clean, segment and read one plan; cache everything decode needs."""
    seed, degrade = args
    path = _key(seed, degrade)
    if path.exists():
        return f"{seed}/{degrade}: cached"
    from kelpod import clean, segment
    from kelpod.evaluation.synth import generate, save
    from kelpod.ocr.engines import get_engine, read_page, remove_long_lines

    sp = generate(seed, degrade)
    with tempfile.TemporaryDirectory() as td:
        save(sp, td)
        page = clean.run(Path(td) / "plan.pdf", dpi=DPI, plan_scale=100.0)[0]
    t0 = time.perf_counter()
    seg = segment.run(page)
    t_seg = time.perf_counter() - t0
    t0 = time.perf_counter()
    gray = remove_long_lines(page.image, page.ink, round(12.0 / 25.4 * page.dpi))
    items = read_page(gray, get_engine("rapidocr"))
    t_read = time.perf_counter() - t0

    # drawn annotations (undegraded px at sp.dpi) -> page px
    f = DPI / sp.dpi
    h, w = page.image.shape
    rot = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -page.deskew_deg, 1.0)
    aff = np.array(sp.degradation.affine_px, float)

    def to_page(p):
        x, y = aff @ np.array([p[0], p[1], 1.0])
        x, y = (x + 0.5) * f - 0.5, (y + 0.5) * f - 0.5
        x, y = rot @ np.array([x, y, 1.0])
        return float(x), float(y)

    gt_dims = []
    for a in sp.annotations:
        if a.kind != "dimension":
            continue
        cx, cy = to_page(a.center)
        line = None if a.line is None else (to_page(a.line[0]), to_page(a.line[1]))
        gt_dims.append({"value": a.value, "text": a.text, "center": (cx, cy), "line": line})
    data = {
        "seed": seed, "degrade": degrade, "page": page, "seg": seg, "items": items,
        "gt_dims": gt_dims, "gt_graph": sp.graph, "t_seg": t_seg, "t_read": t_read,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return f"{seed}/{degrade}: read {t_read:.1f} s, {len(items)} items"


def _d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _span_ok(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    return (_d(a[0], b[0]) <= tol and _d(a[1], b[1]) <= tol) or (
        _d(a[0], b[1]) <= tol and _d(a[1], b[0]) <= tol)


def score_dims(dims, gt_dims, m_per_px: float) -> dict:
    a_tol, s_tol = ANCHOR_TOL_M / m_per_px, SPAN_TOL_M / m_per_px
    pairs = []
    for i, d in enumerate(dims):
        for j, g in enumerate(gt_dims):
            if abs(d.value_m - g["value"]) > VALUE_TOL:
                continue
            dist = _d(d.anchor, g["center"])
            if dist <= a_tol or _span_ok(d.line, g["line"], s_tol):
                pairs.append((dist, i, j))
    used_p, used_g, tp, linked = set(), set(), 0, 0
    for _, i, j in sorted(pairs):
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
        linked += _span_ok(dims[i].line, gt_dims[j]["line"], s_tol)
    with_line = sum(d.line is not None for d in dims)
    by_origin: dict[str, list[int]] = {}
    for i, d in enumerate(dims):
        o = getattr(d, "origin", "read")
        by_origin.setdefault(o, [0, 0])
        by_origin[o][0] += 1
        by_origin[o][1] += i in used_p
    return {"n_pred": len(dims), "n_gt": len(gt_dims), "tp": tp, "linked": linked,
            "with_line": with_line, "by_origin": by_origin}


def evaluate(path: Path, mode: str) -> dict:
    from kelpod import geometry
    from kelpod.evaluation.panel import plan_panel
    from kelpod.ocr.interpret import annotate

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    page = data["page"]
    t0 = time.perf_counter()
    ann = annotate(data["items"], page.ink, "rapidocr", decode=mode)
    t_dec = time.perf_counter() - t0
    t0 = time.perf_counter()
    graph = geometry.run(page, data["seg"], ann)
    t_geo = time.perf_counter() - t0
    panel = plan_panel(graph, data["gt_graph"])
    row = {
        "seed": data["seed"], "degrade": data["degrade"], "mode": mode,
        "t_read": data["t_read"], "t_decode": t_dec, "t_geo": t_geo, "t_seg": data["t_seg"],
        "scale_source": graph.scale_source,
        "m_per_px_err": graph.m_per_px / (0.0254 / DPI * 100) - 1.0,
        "dim_residual": graph.report.dimension_residual_m if graph.report else None,
        **score_dims(ann.dimensions, data["gt_dims"], page.m_per_px_from_scale),
        **{k: panel.get(k) for k in ("correct_closed", "closed", "room_iou", "total_area_error",
                                       "room_area_error", "wall_f1", "edit_cost",
                                       "ext_wall_length_error")},
    }
    return row


def _eval_job(args):
    return evaluate(*args)


def summarise(rows: list[dict]) -> dict:
    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return statistics.fmean(v) if v else None

    n_pred = sum(r["n_pred"] for r in rows)
    n_gt = sum(r["n_gt"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    linked = sum(r["linked"] for r in rows)
    wl = sum(r["with_line"] for r in rows)
    origin: dict[str, list[int]] = {}
    for r in rows:
        for o, (n, t) in r["by_origin"].items():
            origin.setdefault(o, [0, 0])
            origin[o][0] += n
            origin[o][1] += t
    dec = sorted(r["t_decode"] for r in rows)
    return {
        "plans": len(rows),
        "dim_precision": tp / n_pred if n_pred else None,
        "dim_recall": tp / n_gt if n_gt else None,
        "linked_recall": linked / n_gt if n_gt else None,
        "line_precision": linked / wl if wl else None,
        "origin_n_tp": {o: v for o, v in origin.items()},
        "scale_from_dims": statistics.fmean(r["scale_source"] == "dimensions" for r in rows),
        "abs_scale_err_mean": statistics.fmean(abs(r["m_per_px_err"]) for r in rows),
        **{k: mean(k) for k in ("correct_closed", "closed", "room_iou", "total_area_error",
                                "room_area_error", "wall_f1", "edit_cost",
                                "ext_wall_length_error", "dim_residual")},
        "t_read_mean": mean("t_read"),
        "t_seg_mean": mean("t_seg"),
        "t_decode_mean": mean("t_decode"),
        "t_decode_p95": dec[int(0.95 * (len(dec) - 1))] if dec else None,
        "t_geo_mean": mean("t_geo"),
    }


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", default="200000-200019")
    ap.add_argument("--degrade", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--modes", nargs="+", default=["greedy", "chain"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write rows + summary as JSON")
    args = ap.parse_args(argv)
    jobs = [(s, d) for d in args.degrade for s in _seeds(args.seeds)]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(prepare, jobs):
            print(msg, flush=True)
        if args.prepare_only:
            return 0
        ev = [(_key(s, d), m) for m in args.modes for (s, d) in jobs]
        rows = pool.map(_eval_job, ev, chunksize=4)
    result = {}
    for d in args.degrade:
        for m in args.modes:
            sel = [r for r in rows if r["degrade"] == d and r["mode"] == m]
            result[f"d{d:g}/{m}"] = summarise(sel)
    for k, v in result.items():
        print(f"\n== {k}")
        for kk, vv in v.items():
            print(f"  {kk}: {round(vv, 4) if isinstance(vv, float) else vv}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": result, "rows": rows}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
