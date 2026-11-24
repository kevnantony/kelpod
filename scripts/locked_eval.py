"""Locked evaluation: candidate vs. baseline on the frozen splits (docs/EVAL_PROTOCOL.md §2-§5).

    # promotion run (called by scripts/promotion_check.sh)
    uv run python scripts/locked_eval.py --candidate WT --baseline WB --branch exp/x \\
        --log eval/TEST_LOG.md
    # integrity only: frozen-split hashes + training-manifest overlap
    uv run python scripts/locked_eval.py --verify-manifests [--training-manifests a.json ...]
    # smoke / val runs (logged as such, never count against the budget)
    uv run python scripts/locked_eval.py --candidate . --baseline . --branch eval/x \\
        --splits val-synth --smoke --log eval/TEST_LOG.md

For every plan, only the input file is copied into an empty temp dir, and the worktree's own
pipeline runs there in a subprocess (``<worktree>/.venv`` python, ``-I``, the runner
``src/kelpod/evaluation/locked_runner.py`` of *this* script's checkout). Ground truth stays in
this process. Results with ``scale_source == "ground_truth"`` are rejected (scored as crashes, and
the run fails). The IFC is re-opened here, independently of the pipeline's own check.

Output: a Markdown comparison table (stdout and ``<out>/comparison.md``), ``<out>/results.json``
(per-plan panels, for audits only: locked splits are read as aggregates), and one appended line in
``--log``. Exit 0 = promotion rule met (or a smoke/val run without integrity problems).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNNER = HERE / "src" / "kelpod" / "evaluation" / "locked_runner.py"
TIMEOUT_S = 900
LOG_HEADER = (
    "| date (UTC) | kind | branch | candidate | baseline | splits (n plans) "
    "| Δ correct-closed on primary [95% CI] | decision | job |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


# --------------------------------------------------------------------------------------------
# worktrees
# --------------------------------------------------------------------------------------------


def worktree_python(wt: Path) -> str:
    """Interpreter of a worktree's own environment (created with ``uv sync --frozen``)."""
    subprocess.run(["uv", "sync", "--project", str(wt), "--frozen", "--quiet"], check=True, cwd=wt)
    py = wt / ".venv" / "bin" / "python"
    probe = subprocess.run(
        [str(py), "-I", "-c", "import kelpod, sys; print(kelpod.__file__)"],
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
        check=True,
    ).stdout.strip()
    if not Path(probe).resolve().is_relative_to(wt.resolve()):
        raise RuntimeError(f"{py} imports kelpod from {probe}, not from the worktree {wt}")
    return str(py)


def commit_of(wt: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------------------------
# one plan
# --------------------------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _ifc_reopens(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        import ifcopenshell

        f = ifcopenshell.open(str(path))
        return len(f.by_type("IfcWall")) > 0 and len(f.by_type("IfcSpace")) > 0
    except Exception:  # noqa: BLE001
        return False


def run_task(task: dict) -> dict:
    """Run one worktree on one input and score it (runs in a pool worker)."""
    from kelpod.contracts import PlanGraph
    from kelpod.evaluation.panel import plan_panel, proxy_panel

    src = Path(task["input"])
    work = Path(tempfile.mkdtemp(prefix="kelpod-locked-"))
    res_dir = Path(tempfile.mkdtemp(prefix="kelpod-locked-res-"))
    name = "plan" + src.suffix.lower()
    shutil.copyfile(src, work / name)  # ONLY the input file
    out = res_dir / "result.json"
    row = {k: task[k] for k in ("split", "id", "side")}
    try:
        try:
            subprocess.run(
                [task["python"], "-I", str(RUNNER), name, str(out)],
                cwd=work,
                env=_clean_env(),
                timeout=TIMEOUT_S,
                capture_output=True,
                check=False,
            )
            result = json.loads(out.read_text()) if out.is_file() else {"ok": False}
            if not out.is_file():
                result["error"] = "runner wrote no result"
        except subprocess.TimeoutExpired:
            result = {"ok": False, "error": f"timeout after {TIMEOUT_S} s"}
        pred = None
        if result.get("ok"):
            pred = PlanGraph.model_validate(result["graph"])
            if pred.scale_source == "ground_truth":
                row["rejected"] = "scale_source == ground_truth"
                pred = None
        ifc_ok = _ifc_reopens(work / "model.ifc") if pred is not None else False
        runtime = result.get("runtime_s") if pred is not None else None
        if task["gt"]:
            gt = PlanGraph.model_validate(json.loads(Path(task["gt"]).read_text())["graph"])
            row["panel"] = plan_panel(pred, gt, runtime_s=runtime, ifc_ok=ifc_ok)
        else:
            row["panel"] = proxy_panel(pred, runtime_s=runtime, ifc_ok=ifc_ok)
        if not result.get("ok"):
            row["error"] = result.get("error", "?")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(res_dir, ignore_errors=True)
    return row


# --------------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------------


def available_splits(manifests: dict, wanted: list[str] | None) -> tuple[list[dict], list[str]]:
    """Runnable splits (as descriptors) and notes about skipped ones."""
    from kelpod.evaluation.splits import LOCKED_FAMILIES, split_data_dir

    notes, out = [], []
    names = wanted or [n for n, m in manifests.items() if m.get("family") in LOCKED_FAMILIES]
    if wanted is None:
        names.append("fresh")
    for n in names:
        if n == "fresh":
            out.append({"split": "fresh", "family": "fresh", "kind": "synth"})
            continue
        m = manifests.get(n)
        if m is None:
            notes.append(f"{n}: no manifest, skipped")
        elif m["status"] != "frozen":
            notes.append(f"{n}: {m['status']} ({m.get('note', '')}), skipped")
        elif not split_data_dir(m).is_dir():
            notes.append(f"{n}: data directory {split_data_dir(m)} missing, skipped")
        else:
            out.append({"split": n, "family": m["family"], "kind": m["kind"], "manifest": m})
    return out, notes


def split_items(desc: dict, fresh_dir: Path | None, limit: int | None) -> list[dict]:
    """(id, input path, gt path or None) for every item of a split."""
    from kelpod.evaluation.splits import split_data_dir

    if desc["split"] == "fresh":
        items = sorted(json.loads((fresh_dir / "items.json").read_text()), key=lambda i: i["id"])
        rows = [
            {"id": i["id"], "input": fresh_dir / i["id"] / "plan.pdf",
             "gt": fresh_dir / i["id"] / "ground_truth.json"}
            for i in items
        ]  # fmt: skip
    else:
        m = desc["manifest"]
        d = split_data_dir(m)
        if m["kind"] == "synth":
            rows = [
                {"id": i["id"], "input": d / i["id"] / m["input_file"],
                 "gt": d / i["id"] / m["gt_file"]}
                for i in m["items"]
            ]  # fmt: skip
        else:  # proxy / sealed without annotations: the item is the input file
            exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            rows = [
                {"id": i["id"], "input": d / i["id"], "gt": None}
                for i in m["items"]
                if Path(i["id"]).suffix.lower() in exts
            ]
    return rows[:limit] if limit else rows


def build_fresh(cand_commit: str, base_commit: str, n: int, workers: int) -> Path:
    from kelpod.evaluation.splits import FRESH, fresh_params, generate_synth

    p = fresh_params(cand_commit, base_commit, n)
    d = Path(tempfile.mkdtemp(prefix="kelpod-fresh-"))
    items = generate_synth(p["seeds"], p["degrades"], FRESH["style"], d, workers)
    (d / "items.json").write_text(json.dumps(items))
    (d / "params.json").write_text(json.dumps({k: p[k] for k in ("digest",)}))
    return d


# --------------------------------------------------------------------------------------------
# aggregation and reporting
# --------------------------------------------------------------------------------------------


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _p95(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    import numpy as np

    return float(np.percentile(vals, 95))


def _seed_for(*parts: str) -> int:
    return int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:12], 16)


def compare(rows: list[dict], splits: list[dict], sides: list[str], commits: dict) -> dict:
    """Per split and per candidate side: Δ with CI for every panel metric."""
    from kelpod.evaluation.panel import METRICS, PROXY_METRICS
    from kelpod.evaluation.stats import compare_metric

    by = {}
    for r in rows:
        by.setdefault((r["split"], r["side"]), {})[r["id"]] = r["panel"]
    out: dict = {}
    for side in sides:
        res = {}
        for s in splits:
            name = s["split"]
            metrics = PROXY_METRICS if s["kind"] in ("proxy", "sealed") else METRICS
            b, c = by.get((name, "base"), {}), by.get((name, side), {})
            ids = sorted(set(b) & set(c))
            deltas, kinds = {}, {}
            for m, (direction, kind) in metrics.items():
                deltas[m] = compare_metric(
                    m, direction, kind, [c[i][m] for i in ids], [b[i][m] for i in ids],
                    seed=_seed_for(name, m, commits[side], commits["base"]),
                )  # fmt: skip
                kinds[m] = (direction, kind)
            res[name] = {
                "n": len(ids),
                "deltas": deltas,
                "kinds": kinds,
                "p95_cand": _p95([c[i]["runtime_s"] for i in ids]),
                "p95_base": _p95([b[i]["runtime_s"] for i in ids]),
                "family": s["family"],
                "ids": ids,
                "c": c,
                "b": b,
            }
        out[side] = res
    return out


def primary_of(res: dict, test_real_pages: int) -> tuple[list[str], str]:
    fam = {n: r["family"] for n, r in res.items()}
    if test_real_pages >= 40 and "test-real" in fam.values():
        return [n for n, f in fam.items() if f == "test-real"], "test-real"
    names = [n for n, f in fam.items() if f in ("test-ood", "fresh")]
    return names, "test-ood + fresh"


def pooled(res: dict, names: list[str], side: str, commits: dict) -> dict:
    from kelpod.evaluation.panel import METRICS
    from kelpod.evaluation.stats import compare_metric

    out = {}
    for m, (direction, kind) in METRICS.items():
        cs, bs = [], []
        for n in names:
            r = res[n]
            cs += [r["c"][i][m] for i in r["ids"]]
            bs += [r["b"][i][m] for i in r["ids"]]
        out[m] = compare_metric(
            m, direction, kind, cs, bs, seed=_seed_for("primary", m, commits[side], commits["base"])
        )
    return out


def _fmt(v, rel=False):
    if v is None:
        return "–"
    return f"{v:+.1%}" if rel else f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"


def markdown(res: dict, decisions: dict, meta: dict) -> str:
    from kelpod.evaluation.stats import regression

    lines = [
        f"## Locked evaluation {meta['date']} ({meta['kind']})",
        "",
        f"- branch `{meta['branch']}`; candidate `{meta['commits']['cand0'][:10]}`, baseline "
        f"`{meta['commits']['base'][:10]}`; {meta['machine']}; {meta['workers']} workers",
        f"- splits: {', '.join(f'{n} ({r["n"]})' for n, r in res['cand0'].items())}",
        *[f"- note: {n}" for n in meta["notes"]],
        "- synthetic splits only say something about synthetic plans, not real scans",
        "",
    ]
    for side, splits in res.items():
        if len(res) > 1:
            lines += [f"### candidate {side}", ""]
        for name, r in splits.items():
            lines += [
                f"#### {name} (n = {r['n']}, family {r['family']})",
                "",
                "| metric | baseline | candidate | Δ [95% CI] | rule 3 |",
                "|---|---|---|---|---|",
            ]
            for m, d in r["deltas"].items():
                direction, kind = r["kinds"][m]
                rel = d.relative
                ci = (
                    f"{_fmt(d.delta, rel)} [{_fmt(d.lo, rel)}, {_fmt(d.hi, rel)}]"
                    if d.delta is not None
                    else "–"
                )
                flag = "REGRESSION" if regression(d, direction, kind) else "ok"
                label = m + (" (median)" if d.stat == "median" else "")
                lines.append(f"| {label} | {_fmt(d.base)} | {_fmt(d.cand)} | {ci} | {flag} |")
            lines.append(
                f"| runtime p95 s | {_fmt(r['p95_base'])} | {_fmt(r['p95_cand'])} | | "
                f"{'ok' if (r['p95_cand'] or 0) < 180 else 'OVER 180 s'} |"
            )
            lines.append("")
    lines += ["### Promotion rule (§4)", ""]
    for side, dec in decisions.items():
        for rule in dec["rules"]:
            st = {True: "pass", False: "FAIL", None: "n/a"}[rule["passed"]]
            lines.append(f"- {side} rule {rule['rule']}: **{st}**. {rule['detail']}")
    lines += ["", f"**Decision: {meta['decision']}**", ""]
    return "\n".join(lines)


def append_log(path: Path, line: str) -> None:
    """Append-only: never rewrites existing lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "a") as f:
            f.write("# Locked-run log (append-only, docs/EVAL_PROTOCOL.md §5)\n\n" + LOG_HEADER)
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def locked_runs_of(path: Path, branch: str) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[1] == "locked" and cells[2] == branch:
            n += 1
    return n


# --------------------------------------------------------------------------------------------
# verify mode
# --------------------------------------------------------------------------------------------


def verify(training: list[Path]) -> int:
    from kelpod.evaluation.splits import check_training_overlap, load_manifests, verify_manifest

    manifests = load_manifests(HERE)
    problems = []
    for name, m in manifests.items():
        p = verify_manifest(m)
        problems += p
        n_files = sum(len(i.get("files") or [1]) for i in m.get("items", []))
        state = "OK" if not p else f"{len(p)} problem(s)"
        print(f"{name}: {m['status']}, {m.get('n_items', 0)} items, {n_files} files: {state}")
    found = sorted((HERE / "training" / "manifests").glob("*.json")) + list(training)
    overlap = check_training_overlap(found, manifests)
    print(f"training manifests checked: {len(found)}; overlap problems: {len(overlap)}")
    for p in problems + overlap:
        print("FAIL:", p)
    return 1 if problems or overlap else 0


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--branch")
    ap.add_argument("--log", type=Path, default=HERE / "eval" / "TEST_LOG.md")
    ap.add_argument("--splits", nargs="+", help="default: every locked split + fresh")
    ap.add_argument("--limit", type=int, help="first N plans per split (implies --smoke)")
    ap.add_argument("--smoke", action="store_true", help="log as smoke run; no promotion")
    ap.add_argument("--extra-candidate", type=Path, action="append", default=[],
                    help="same change from another training seed (rule 5)")  # fmt: skip
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--fresh-n", type=int, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "-"))
    ap.add_argument("--verify-manifests", action="store_true")
    ap.add_argument("--training-manifests", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)

    if args.verify_manifests:
        return verify(args.training_manifests)
    if not (args.candidate and args.baseline and args.branch):
        ap.error("--candidate, --baseline and --branch are needed (or --verify-manifests)")

    from kelpod.evaluation.splits import load_manifests, verify_manifest
    from kelpod.evaluation.stats import RuleResult, decide

    smoke = args.smoke or args.limit is not None
    manifests = load_manifests(HERE)
    splits, notes = available_splits(manifests, args.splits)
    families = {s["family"] for s in splits}
    locked = bool(families & {"test-synth", "test-ood", "test-real", "fresh"})
    kind = "smoke" if smoke else "locked" if locked else "val"
    if kind == "locked":
        used = locked_runs_of(args.log, args.branch)
        if used >= 2:
            print(f"FAIL rule 6: {used} locked runs already logged for {args.branch}")
            return 3
    for s in splits:
        if "manifest" in s:
            bad = verify_manifest(s["manifest"])
            if bad:
                print("\n".join(bad))
                print("FAIL: frozen split does not match its manifest")
                return 2
    test_real = manifests.get("test-real", {})
    test_real_pages = test_real.get("n_items", 0) if test_real.get("status") == "frozen" else 0

    cands = [args.candidate, *args.extra_candidate]
    sides = [f"cand{i}" for i in range(len(cands))]
    pythons = {"base": worktree_python(args.baseline)}
    commits = {"base": commit_of(args.baseline)}
    for side, wt in zip(sides, cands, strict=True):
        pythons[side] = worktree_python(wt)
        commits[side] = commit_of(wt)

    fresh_dir = None
    if any(s["split"] == "fresh" for s in splits):
        from kelpod.evaluation.splits import FRESH

        n = args.fresh_n or (min(args.limit, FRESH["n"]) if args.limit else FRESH["n"])
        t0 = time.time()
        fresh_dir = build_fresh(commits["cand0"], commits["base"], n, args.workers)
        print(f"fresh: {n} plans generated in {time.time() - t0:.0f} s", flush=True)

    tasks = []
    for s in splits:
        for it in split_items(s, fresh_dir, args.limit):
            for side in ["base", *sides]:  # interleaved: same load on both sides
                tasks.append({
                    "split": s["split"], "id": it["id"], "side": side, "python": pythons[side],
                    "input": str(it["input"]), "gt": str(it["gt"]) if it["gt"] else None,
                })  # fmt: skip
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for k, fu in enumerate(as_completed(futs), 1):
            rows.append(fu.result())
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} ({time.time() - t0:.0f} s)", flush=True)
    if fresh_dir is not None:
        shutil.rmtree(fresh_dir, ignore_errors=True)

    rejected = [r for r in rows if r.get("rejected")]
    res = compare(rows, splits, sides, commits)
    decisions = {}
    for side in sides:
        names, label = primary_of(res[side], test_real_pages)
        deltas = {n: r["deltas"] for n, r in res[side].items()}
        kinds = {n: r["kinds"] for n, r in res[side].items()}
        fams = {n: r["family"] for n, r in res[side].items()}
        primary = "primary"
        if names:
            deltas[primary] = pooled(res[side], names, side, commits)
            kinds[primary] = kinds[names[0]]
        dec = decide(
            deltas, kinds, primary=primary,
            p95_cand={n: r["p95_cand"] for n, r in res[side].items()},
            families=fams, test_real_pages=test_real_pages,
        )  # fmt: skip
        dec.rules[0].detail = f"[{label}: {', '.join(names) or 'none available'}] " + (
            dec.rules[0].detail
        )
        if not names:
            dec.rules[0].passed = False
        if rejected:
            dec.rules.insert(
                0, RuleResult("integrity", False, f"{len(rejected)} result(s) used GT scale")
            )
        decisions[side] = dec
    if len(sides) > 1:
        worst = all(d.passed for d in decisions.values())
        decisions["cand0"].rules.append(
            RuleResult(
                "5 seed stability",
                worst,
                f"rule held for {sum(d.passed for d in decisions.values())} of {len(sides)} seeds",
            )  # noqa: E501
        )
    else:
        decisions["cand0"].rules.append(
            RuleResult("5 seed stability", None, "report only: one candidate, no extra seeds")
        )
    passed = all(d.passed for d in decisions.values())
    decision = (
        f"smoke run, no promotion decision (rule {'met' if passed else 'not met'})"
        if smoke
        else ("PASS" if passed else "FAIL") if kind == "locked"
        else f"val run, no promotion decision (rule {'met' if passed else 'not met'})"
    )  # fmt: skip

    import platform

    date = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    meta = {
        "date": date + " UTC",
        "kind": kind,
        "branch": args.branch,
        "commits": commits,
        "machine": f"{platform.node()} {platform.machine()}, {os.cpu_count()} cpus",
        "workers": args.workers,
        "notes": notes + ([f"{len(rejected)} rejected results"] if rejected else []),
        "decision": decision,
        "limit": args.limit,
    }
    md = markdown(res, {s: d.to_dict() for s, d in decisions.items()}, meta)
    print(md)
    out = args.out or Path.home() / "kelpod-runs" / "locked" / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "_" + args.branch.replace("/", "-")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(md)
    (out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "decisions": {s: d.to_dict() for s, d in decisions.items()},
                "deltas": {
                    side: {
                        n: {m: d.to_dict() for m, d in r["deltas"].items()} for n, r in sp.items()
                    }
                    for side, sp in res.items()
                },
                "plans": rows,
            },
            indent=1,
        )
    )
    names, _ = primary_of(res["cand0"], test_real_pages)
    if names:
        d = pooled(res["cand0"], names, "cand0", commits)["correct_closed"]
        head = f"{d.delta:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] ({'+'.join(names)})"
    else:
        head = "n/a (no primary split in this run)"
    split_list = ", ".join(f"{n} ({r['n']})" for n, r in res["cand0"].items())
    branch_cell = args.branch if kind == "locked" else f"{args.branch} ({kind})"
    short = "PASS" if decision == "PASS" else "FAIL" if decision == "FAIL" else kind
    append_log(
        args.log,
        f"| {date} | {kind} | {branch_cell} | {commits['cand0'][:10]} | {commits['base'][:10]} "
        f"| {split_list} | {head} | {short} | {args.job_id} ({out}) |",
    )
    print(f"results: {out}; log: {args.log}")
    if rejected:
        return 2
    return 0 if (passed or kind != "locked") else 1


if __name__ == "__main__":
    sys.exit(main())
