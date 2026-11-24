"""Classical vs learned on the frozen ``val-synth`` split (200 plans, train style), scored with the
eval branch's panel (``plan_panel`` from ``~/kelpod-wt/protocol-v1``, loaded by file path).

``scripts/locked_eval.py`` runs each worktree's *default* pipeline in a clean environment, so it
cannot select ``segment_backend="learned"``; this script does the same per plan (copy only
``plan.pdf`` into an empty temp dir, ``kelpod.pipeline.convert``, re-open the IFC) with an
explicit backend, then applies the same panel. val is for model selection: these are dev numbers,
not a locked evaluation.

    KELPOD_SEG_CHECKPOINT=~/kelpod-models/kelpod-seg/r34/model.pt \
    uv run python -m kelpod_train.val_synth --backends classical learned --out runs/val_synth
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROTOCOL_WT = Path.home() / "kelpod-wt" / "protocol-v1"
EVAL_DATA = Path.home() / "kelpod-data" / "eval"


def _panel_module():
    spec = importlib.util.spec_from_file_location(
        "protocol_panel", PROTOCOL_WT / "src" / "kelpod" / "evaluation" / "panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["protocol_panel"] = mod
    spec.loader.exec_module(mod)
    return mod


def _one(args) -> dict:
    item_dir, item_id, backend = args
    from kelpod.contracts import PlanGraph
    from kelpod.pipeline import convert

    panel = _panel_module()
    gt = PlanGraph.model_validate(
        json.loads((item_dir / "ground_truth.json").read_text())["graph"]
    )
    work = Path(tempfile.mkdtemp(prefix="kelpod-val-"))
    row = {"id": item_id, "backend": backend}
    try:
        shutil.copyfile(item_dir / "plan.pdf", work / "plan.pdf")
        t = time.perf_counter()
        try:
            res = convert(work / "plan.pdf", work / "model.ifc", segment_backend=backend)
            runtime = time.perf_counter() - t
            pred, ifc_ok = res.graph, bool(res.export and res.export.reopened_ok)
            row["timings"] = {x.step: x.seconds for x in res.timings}
        except Exception as exc:  # noqa: BLE001
            pred, ifc_ok, runtime = None, False, None
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["panel"] = panel.plan_panel(pred, gt, runtime_s=runtime, ifc_ok=ifc_ok)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return row


def _init(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import cv2

    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["classical", "learned"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.loads((PROTOCOL_WT / "eval" / "manifests" / "val-synth.json").read_text())
    items = man["items"][: a.limit]
    jobs = [(EVAL_DATA / "val-synth" / it["id"], it["id"], b) for b in a.backends for it in items]
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.threads,)) as ex:
        for i, r in enumerate(ex.map(_one, jobs)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    keys = sorted({k for r in rows for k, v in r["panel"].items()
                   if isinstance(v, (int, float, bool)) or v is None})
    lines = ["| metric | " + " | ".join(a.backends) + " |", "|---|" + "---|" * len(a.backends)]
    for k in keys:
        cells = []
        for b in a.backends:
            vals = [float(r["panel"][k]) for r in rows
                    if r["backend"] == b and r["panel"].get(k) is not None]
            cells.append(f"{statistics.mean(vals):.3f}" if vals else "–")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
