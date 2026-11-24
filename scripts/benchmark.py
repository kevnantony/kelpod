"""Benchmark the whole pipeline on synthetic plans.

For each seed: generate a plan, save it (``plan.pdf`` + ground truth), run
``kelpod.pipeline.convert`` on the PDF, then compute the closure metric and the comparison with the
ground truth. Errors are caught per plan (the failing step is read from the traceback). Writes
``benchmark.json`` and ``benchmark.md`` into ``--out``.

    uv run python scripts/benchmark.py --n 10 --degrade 0.5 --out runs/bench
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

STEPS = ("clean", "segment", "ocr", "geometry", "model", "export")


def _failed_step(exc: BaseException) -> str:
    """Deepest ``kelpod.<step>`` module in the traceback (``pipeline`` if none)."""
    step = "pipeline"
    for frame in traceback.extract_tb(exc.__traceback__):
        parts = Path(frame.filename).parts
        if "kelpod" in parts:
            i = len(parts) - 1 - parts[::-1].index("kelpod")
            if i + 1 < len(parts) and parts[i + 1] in STEPS:
                step = parts[i + 1]
    return step


def run_one(
    seed: int, out: Path, *, degrade: float, ocr_backend: str, schema: str, export: bool
) -> dict:
    from kelpod.evaluation.metrics import closure_metrics, compare_to_ground_truth
    from kelpod.evaluation.synth import generate, save
    from kelpod.pipeline import convert

    d = out / f"seed_{seed:04d}"
    t0 = time.perf_counter()
    plan = generate(seed, degrade)
    save(plan, d)
    row: dict = {
        "seed": seed,
        "n_rooms_gt": len(plan.graph.rooms),
        "rotation_deg": round(plan.degradation.rotation_deg, 3),
        "synth_s": round(time.perf_counter() - t0, 3),
        "ok": False,
    }
    t1 = time.perf_counter()
    try:
        res = convert(
            d / "plan.pdf",
            d / "model.ifc" if export else None,
            schema=schema,
            ocr_backend=ocr_backend,
        )
    except Exception as exc:  # noqa: BLE001 - one failing plan must not stop the benchmark
        row.update(
            error=f"{type(exc).__name__}: {exc}"[:300],
            failed_step=_failed_step(exc),
            total_s=round(time.perf_counter() - t1, 3),
        )
        return row
    row["ok"] = True
    row["timings"] = {t.step: round(t.seconds, 3) for t in res.timings}
    row["total_s"] = round(res.total_seconds, 3)
    rep = closure_metrics(res.graph)
    row["closure"] = {
        "is_closed": rep.is_closed,
        "node_closure": round(rep.node_closure, 4),
        "room_closure": round(rep.room_closure, 4),
        "n_components": rep.n_components,
        "n_dangling": len(rep.dangling_nodes),
        "dimension_residual_m": rep.dimension_residual_m,
        "scale_source": res.graph.scale_source,
    }
    row["compare"] = compare_to_ground_truth(res.graph, plan.graph).model_dump()
    if res.export is not None:
        row["export"] = {
            "reopened_ok": res.export.reopened_ok,
            "n_warnings": len(res.export.warnings),
            "entities": res.export.entity_counts,
        }
    return row


def _fmt(v, digits: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        "# Kelpod benchmark on synthetic plans",
        "",
        f"{meta['date']}, {meta['n']} plans, degrade {meta['degrade']}, "
        f"OCR `{meta['ocr_backend']}`,"
        f" {meta['machine']}. Synthetic plans only: these numbers say nothing about real scans.",
        "",
        "| seed | rooms | ok | closed | node cl. | room cl. | room IoU | room P/R | wall F1 "
        "| doors | windows | area err | total s | " + " | ".join(STEPS) + " | error |",
        "|" + "---|" * (14 + len(STEPS)),
    ]
    for r in rows:
        c, m, t = r.get("closure", {}), r.get("compare", {}), r.get("timings", {})
        orec = m.get("opening_recall", {})
        pr = f"{_fmt(m['room_precision'])}/{_fmt(m['room_recall'])}" if m else "–"
        err = f"{r['failed_step']}: {r['error']}"[:90] if not r["ok"] else ""
        lines.append(
            f"| {r['seed']} | {r['n_rooms_gt']} | {_fmt(r['ok'])} | {_fmt(c.get('is_closed'))} "
            f"| {_fmt(c.get('node_closure'))} | {_fmt(c.get('room_closure'))} "
            f"| {_fmt(m.get('room_iou_mean'))} | {pr} | {_fmt(m.get('wall_f1'))} "
            f"| {_fmt(orec.get('door'))} | {_fmt(orec.get('window'))} "
            f"| {_fmt(m.get('room_area_error'), 3)} | {_fmt(r.get('total_s'), 1)} | "
            + " | ".join(_fmt(t.get(s), 1) for s in STEPS)
            + f" | {err.replace('|', '/')} |"
        )
    s = meta["summary"]
    lines += ["", "## Summary", ""]
    lines += [f"- {k}: {_fmt(v, 3)}" for k, v in s.items()]
    return "\n".join(lines) + "\n"


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.fmean(vals) if vals else None

    failures: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            failures[r["failed_step"]] = failures.get(r["failed_step"], 0) + 1
    return {
        "plans": len(rows),
        "pipeline_ok": len(ok),
        "failures_by_step": json.dumps(failures) if failures else "none",
        "closed_share": mean([float(r["closure"]["is_closed"]) for r in ok]),
        "node_closure_mean": mean([r["closure"]["node_closure"] for r in ok]),
        "room_closure_mean": mean([r["closure"]["room_closure"] for r in ok]),
        "room_iou_mean": mean([r["compare"]["room_iou_mean"] for r in ok]),
        "room_recall_mean": mean([r["compare"]["room_recall"] for r in ok]),
        "wall_f1_mean": mean([r["compare"]["wall_f1"] for r in ok]),
        "door_recall_mean": mean([r["compare"]["opening_recall"].get("door") for r in ok]),
        "window_recall_mean": mean([r["compare"]["opening_recall"].get("window") for r in ok]),
        "room_area_error_mean": mean([r["compare"]["room_area_error"] for r in ok]),
        "total_s_mean": mean([r["total_s"] for r in ok]),
        "total_s_max": max((r["total_s"] for r in ok), default=None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--degrade", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("runs/benchmark"))
    ap.add_argument("--ocr-backend", default="rapidocr")
    ap.add_argument("--schema", default="IFC4", choices=["IFC4", "IFC2X3"])
    ap.add_argument("--no-export", action="store_true", help="skip writing IFC")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seed, args.seed + args.n):
        row = run_one(
            seed,
            args.out,
            degrade=args.degrade,
            ocr_backend=args.ocr_backend,
            schema=args.schema,
            export=not args.no_export,
        )
        rows.append(row)
        status = "ok" if row["ok"] else f"FAILED in {row['failed_step']}: {row['error'][:80]}"
        print(f"seed {seed}: {status} ({row.get('total_s', 0):.1f} s)", flush=True)
    meta = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "n": args.n,
        "degrade": args.degrade,
        "ocr_backend": args.ocr_backend,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "summary": summarise(rows),
    }
    (args.out / "benchmark.json").write_text(json.dumps({"meta": meta, "plans": rows}, indent=1))
    md = to_markdown(rows, meta)
    (args.out / "benchmark.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
