"""Dev tool (not pipeline code): time the OCR step on 1 core and score dimension / room-stamp reads
against the synthetic ground truth. Usage (pin the core yourself):

    taskset -c 3 uv run python scripts/dev_ocr_speed.py --n 30 --engine rapidocr_fast
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "ORT_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

from kelpod import clean, ocr  # noqa: E402

ROOT = Path.home() / "kelpod-data/eval/val-synth"


def score(ann, gt, tol_px=60.0):
    dims = [a for a in gt["annotations"] if a["kind"] == "dimension"]
    stamps = [a for a in gt["annotations"] if a["kind"] == "room_name"]
    areas = {a["room_id"]: a for a in gt["annotations"] if a["kind"] == "room_area"}
    used, tp = set(), 0
    for g in dims:
        cx = sum(p[0] for p in g["quad"]) / 4
        cy = sum(p[1] for p in g["quad"]) / 4
        for i, d in enumerate(ann.dimensions):
            if i in used:
                continue
            if abs(d.value_m - g["value"]) < 0.0015 and math.hypot(d.anchor[0] - cx, d.anchor[1] - cy) < tol_px:
                used.add(i)
                tp += 1
                break
    st = 0
    for g in stamps:
        a = areas.get(g["room_id"])
        for r in ann.room_stamps:
            if r.name.strip().lower() == g["text"].strip().lower() and (
                a is None or r.area_m2 is None or abs(r.area_m2 - a["value"]) < 0.006
            ):
                st += 1
                break
    return tp, len(dims), len(ann.dimensions), st, len(stamps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--engine", default="rapidocr")
    ap.add_argument("--tile", type=int, default=0, help="0 = engine default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.tile:
        from kelpod.ocr import engines

        engines.register_engine("dev", lambda: engines.RapidOcrEngine(tile=a.tile))
        a.engine = "dev"
    ids = sorted(p.name for p in ROOT.iterdir())[: a.n]
    tot = [0, 0, 0, 0, 0]
    secs = []
    per = {}
    for pid in ids:
        gt = json.loads((ROOT / pid / "ground_truth.json").read_text())
        page = clean.run(ROOT / pid / "plan.png", dpi=gt["dpi"], plan_scale=gt["plan_scale"])[0]
        ocr.run(page, backend=a.engine) if pid == ids[0] else None  # warm-up (model load)
        t0 = time.perf_counter()
        ann = ocr.run(page, backend=a.engine)
        secs.append(time.perf_counter() - t0)
        s = score(ann, gt)
        for i, v in enumerate(s):
            tot[i] += v
        per[pid] = [(d.value_m, round(d.anchor[0]), round(d.anchor[1])) for d in ann.dimensions]
    tp, ng, npred, st, ns = tot
    secs_sorted = sorted(secs)
    print(f"engine={a.engine} n={len(ids)} mean_s={sum(secs)/len(secs):.2f} "
          f"median_s={secs_sorted[len(secs)//2]:.2f} dim_recall={tp/ng:.4f} "
          f"dim_precision={tp/max(npred,1):.4f} stamps={st}/{ns}")
    if a.out:
        Path(a.out).write_text(json.dumps(per))


if __name__ == "__main__":
    main()
