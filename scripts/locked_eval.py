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

Each worktree may declare a run config, ``kelpod-run.toml`` (``kelpod.evaluation.run_config``):
backends, the uv project whose interpreter runs the pipeline, and one checkpoint per training
seed (path outside the worktree + sha256). Every checkpoint is hash-verified, its training
manifests are checked for test overlap, and it is staged read-only and passed only through
``KELPOD_SEG_CHECKPOINT``. Each candidate checkpoint is its own side (``cand0``, ``cand1``, ...);
the baseline runs once per declared seed too, candidate seed k is compared with baseline seed k
(``pair_sides``), and rule 5 must hold for the worse pair. The configs are printed in the report.

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


def worktree_python(wt: Path, project: str = ".") -> str:
    """Interpreter of a worktree's own environment (created with ``uv sync --frozen``), of the uv
    project ``project`` inside it (the run config's ``python_project``; "." = the root)."""
    proj = wt / project
    subprocess.run(
        ["uv", "sync", "--project", str(proj), "--frozen", "--quiet"], check=True, cwd=proj
    )
    py = proj / ".venv" / "bin" / "python"
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
# run configs (kelpod-run.toml) and sides (one per worktree and training seed)
# --------------------------------------------------------------------------------------------


def load_run_configs(
    baseline: Path, cands: list[Path], manifests: dict, kind: str, notes: list[str]
) -> tuple[dict, list[str]]:
    """Every worktree's declared run config, and the integrity problems that stop the run.
    A config that is not committed is refused in locked runs (it must be part of the commit that
    promotion_check sees) and only noted in smoke/val runs."""
    from kelpod.evaluation import run_config as rc

    configs, problems = {}, []
    wts = list(dict.fromkeys([baseline, *cands]))
    for wt in wts:
        try:
            cfg = rc.load(wt)
        except rc.RunConfigError as exc:
            problems.append(str(exc))
            continue
        cfg.state = rc.git_state(wt)
        if cfg.state in ("modified", "untracked"):
            msg = f"{wt}: {rc.FILENAME} is {cfg.state}; the run config must be committed"
            if kind == "locked":
                problems.append(msg)
            else:
                notes.append(f"{msg} (tolerated in a {kind} run only)")
        others = [w for w in (*wts, HERE) if w != wt]
        problems += [f"{wt}: {p}" for p in rc.check(wt, cfg, manifests, other_worktrees=others)]
        configs[wt] = cfg
    return configs, problems


def build_sides(baseline: Path, cands: list[Path], configs: dict, staging: Path) -> dict:
    """Baseline sides plus ``cand0``, ``cand1``, ...: one candidate side per declared checkpoint
    (training seed) of every candidate worktree (one side if it declares none). The baseline gets
    one side per declared checkpoint too (``base0``, ``base1``, ...; ``base`` if it declares none),
    so every seed is run on the baseline as well (see ``pair_sides``). Checkpoints are staged
    read-only under ``staging``."""
    from kelpod.evaluation import run_config as rc

    pythons: dict = {}
    sides: dict = {}

    def add(side: str, wt: Path, cfg, ck) -> None:
        key = (wt, cfg.python_project)
        if key not in pythons:
            pythons[key] = worktree_python(wt, cfg.python_project)
        env = {}
        if ck is not None:
            env[rc.CHECKPOINT_ENV] = str(rc.stage_checkpoint(ck, staging / side))
        sides[side] = {
            "worktree": str(wt), "commit": commit_of(wt), "python": pythons[key],
            "python_project": cfg.python_project, "args": cfg.runner_args(), "env": env,
            "seed": ck.seed if ck else None, "sha256": ck.sha256 if ck else None,
            "checkpoint": str(ck.path) if ck else None,
        }  # fmt: skip

    bcfg = configs[baseline]
    if bcfg.checkpoints:
        for i, ck in enumerate(bcfg.checkpoints):
            add(f"base{i}", baseline, bcfg, ck)
    else:
        add("base", baseline, bcfg, None)
    k = 0
    for wt in cands:
        cfg = configs[wt]
        for ck in cfg.checkpoints or [None]:
            add(f"cand{k}", wt, cfg, ck)
            k += 1
    return sides


def base_sides(sides: dict) -> list[str]:
    return [s for s in sides if s.startswith("base")]


def pair_sides(sides: dict) -> list[tuple[str, str, str]]:
    """(key, candidate side, baseline side) for every comparison. Rule 3 to 5 must hold for all.

    - baseline with at most one side: every candidate side against it (key = candidate side);
    - several baseline seeds: per candidate worktree, candidate seed k against baseline seed k when
      the seed labels match as sets; else by position when the counts are equal; else every
      candidate seed against every baseline seed (key ``candX~baseY``), so the rule must hold for
      the worst pairing."""
    bases = base_sides(sides)
    cands = [s for s in sides if s not in bases]
    if len(bases) <= 1:
        return [(c, c, bases[0]) for c in cands]
    out = []
    for wt in dict.fromkeys(sides[c]["worktree"] for c in cands):
        group = [c for c in cands if sides[c]["worktree"] == wt]
        cs = [sides[c]["seed"] for c in group]
        bs = [sides[b]["seed"] for b in bases]
        if None not in cs and len(set(cs)) == len(cs) and set(cs) == set(bs):
            by_seed = {sides[b]["seed"]: b for b in bases}
            out += [(c, c, by_seed[sides[c]["seed"]]) for c in group]
        elif len(group) == len(bases):
            out += [(c, c, b) for c, b in zip(group, bases, strict=True)]
        else:
            out += [(f"{c}~{b}", c, b) for c in group for b in bases]
    return out


def staged_checkpoints_changed(sides: dict) -> list[str]:
    """Problems if a staged checkpoint no longer has its declared hash after the runs."""
    from kelpod.evaluation import run_config as rc

    out = []
    for side, i in sides.items():
        staged = i["env"].get(rc.CHECKPOINT_ENV)
        if staged and rc.sha256_file(Path(staged)) != i["sha256"]:
            out.append(f"{side}: checkpoint changed during the run ({staged})")
    return out


def side_label(info: dict) -> str:
    args = " ".join(info["args"]) or "defaults"
    ck = f", seed {info['seed']} sha256 {info['sha256'][:12]}" if info["sha256"] else ""
    return f"{Path(info['worktree']).name}@{info['commit'][:10]}, {args}{ck}"


def pair_label(sides: dict, pair: tuple[str, str, str]) -> str:
    key, c, b = pair
    return f"{key}: {side_label(sides[c])} vs baseline {b} ({side_label(sides[b])})"


def log_cell(sides: dict, names: list[str]) -> str:
    """Commit(s) and checkpoint hashes of the given sides for the TEST_LOG line."""
    commits = list(dict.fromkeys(sides[n]["commit"][:10] for n in names))
    hashes = [f"s{sides[n]['seed']}:{sides[n]['sha256'][:12]}" for n in names if sides[n]["sha256"]]
    return " ".join(commits) + (f" ckpt {', '.join(hashes)}" if hashes else "")


# --------------------------------------------------------------------------------------------
# one plan
# --------------------------------------------------------------------------------------------


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The caller's environment without Python path overrides and without any ``KELPOD_*``
    variable (so an operator's ``KELPOD_SEG_CHECKPOINT`` cannot leak into a side that did not
    declare it), CPU only, then the side's declared variables."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "VIRTUAL_ENV") and not k.startswith("KELPOD_")
    }
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["CUDA_VISIBLE_DEVICES"] = ""  # the reference node is a CPU node
    env.update(extra or {})
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


def physical_cores() -> list[list[int]]:
    """Usable logical CPUs grouped by physical core (hyperthread siblings together)."""
    allowed = sorted(os.sched_getaffinity(0))
    groups: dict[str, list[int]] = {}
    for c in allowed:
        f = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        key = f.read_text().strip() if f.is_file() else str(c)
        groups.setdefault(key, []).append(c)
    return sorted(groups.values())


def _pin_worker(slots, cores_per_plan: int) -> None:
    """Pool initializer: give this worker its own ``cores_per_plan`` physical cores (all their
    hyperthreads; inherited by the pipeline subprocess), so parallel plans do not share cores
    and candidate and baseline run under the same conditions."""
    cores = physical_cores()
    k = slots.get()
    mine = [c for g in cores[k * cores_per_plan : (k + 1) * cores_per_plan] for c in g]
    if mine:
        os.sched_setaffinity(0, mine)


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
                [task["python"], "-I", str(RUNNER), name, str(out), *task.get("args", [])],
                cwd=work,
                env=_clean_env(task.get("env")),
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


def compare(rows: list[dict], splits: list[dict], pairs: list, commits: dict) -> dict:
    """Per split and per (key, candidate side, baseline side): Δ with CI for every panel metric.
    Paired bootstrap over the plans the two sides share. ``commits[key]`` seeds the bootstrap."""
    from kelpod.evaluation.panel import METRICS, PROXY_METRICS
    from kelpod.evaluation.stats import compare_metric

    by = {}
    for r in rows:
        by.setdefault((r["split"], r["side"]), {})[r["id"]] = r["panel"]
    out: dict = {}
    for side, cside, bside in pairs:
        res = {}
        for s in splits:
            name = s["split"]
            metrics = PROXY_METRICS if s["kind"] in ("proxy", "sealed") else METRICS
            b, c = by.get((name, bside), {}), by.get((name, cside), {})
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


def seed_rule(sides_info: dict, passed: dict[str, bool], heads: dict):
    """Rule 5: a trained candidate (one with checkpoints) needs at least 2 distinct training
    seeds, and the promotion rule must hold for every seed, i.e. for the worse one."""
    from kelpod.evaluation.stats import RuleResult

    sides = list(passed)
    worse = min(sides, key=lambda s: heads[s][0])
    trained = [s for s in sides if sides_info[s]["sha256"]]
    n_seeds = len({sides_info[s]["sha256"] for s in trained})
    held = sum(bool(passed[s]) for s in sides)
    worse_txt = f"worse seed: {worse} ({side_label(sides_info[worse])}), {heads[worse][1]}"
    if trained and n_seeds < 2:
        return RuleResult(
            "5 seed stability", False,
            f"trained model with {n_seeds} declared checkpoint(s); at least 2 training seeds "
            f"are needed. {worse_txt}",
        )  # fmt: skip
    if len(sides) > 1:
        return RuleResult(
            "5 seed stability", held == len(sides),
            f"rule held for {held} of {len(sides)} seeds (must hold for the worse seed); "
            f"{worse_txt}",
        )  # fmt: skip
    return RuleResult("5 seed stability", None, "report only: one candidate, no checkpoints")


def primary_headline(res_side: dict, test_real_pages: int, side: str, commits: dict):
    """(sort key, text) of Δ correct-closed on the primary split; without a primary split in the
    run (smoke/val), pooled over all splits of the run, so seeds can still be ranked."""
    names, _ = primary_of(res_side, test_real_pages)
    label = "+".join(names)
    if not names:
        names = list(res_side)
        label = "all splits, no primary in this run: " + "+".join(names)
    if not names:
        return (0.0, 0.0), "n/a (no split)"
    d = pooled(res_side, names, side, commits)["correct_closed"]
    if d.delta is None:
        return (0.0, 0.0), f"n/a ({label})"
    return (d.delta, d.lo), f"{d.delta:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}] ({label})"


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
        f"`{meta['commits']['base'][:10]}`; {meta['machine']}; {meta['workers']} workers x "
        f"{meta['cores_per_plan']} dedicated physical core(s) per plan",
        f"- splits: {', '.join(f'{n} ({r["n"]})' for n, r in next(iter(res.values())).items())}",
        *[f"- note: {n}" for n in meta["notes"]],
        *[f"- side {s}: {side_label(i)}" for s, i in meta["sides"].items()],
        *[f"- pair {p}" for p in meta["pairs"]],
        "- synthetic splits only say something about synthetic plans, not real scans",
        "",
    ]
    for side, splits in res.items():
        if len(res) > 1:
            lines += [f"### candidate {side} (baseline {meta['pair_base'][side]})", ""]
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
    lines += ["", f"**Decision: {meta['decision']}**", "", "### Run configs (kelpod-run.toml)", ""]
    for wt, c in meta["run_configs"].items():
        lines.append(f"- `{wt}` ({c['state']}): {c['summary']}")
        if c["text"]:
            lines += ["", "```toml", c["text"].rstrip("\n"), "```", ""]
    lines.append("")
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
    ap.add_argument("--cores-per-plan", type=int, default=1,
                    help="dedicated physical cores per run (reference: 1)")  # fmt: skip
    ap.add_argument("--workers", type=int, help="default: usable CPUs // cores-per-plan")
    ap.add_argument("--fresh-n", type=int, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "-"))
    ap.add_argument("--verify-manifests", action="store_true")
    ap.add_argument("--training-manifests", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)

    if args.verify_manifests:
        return verify(args.training_manifests)
    n_cores = len(physical_cores())
    if args.workers is None:
        args.workers = max(1, n_cores // args.cores_per_plan)
    if args.workers * args.cores_per_plan > n_cores:
        ap.error(f"{args.workers} workers x {args.cores_per_plan} > {n_cores} physical cores")
    if not (args.candidate and args.baseline and args.branch):
        ap.error("--candidate, --baseline and --branch are needed (or --verify-manifests)")

    from kelpod.evaluation import run_config as rc
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

    args.baseline = args.baseline.resolve()
    cands = [c.resolve() for c in (args.candidate, *args.extra_candidate)]
    configs, problems = load_run_configs(args.baseline, cands, manifests, kind, notes)
    if problems:
        print("\n".join(f"FAIL: {p}" for p in problems))
        print("FAIL: run config integrity (kelpod-run.toml); nothing was run")
        return 2
    staging = rc.staging_dir()
    try:
        sides_info = build_sides(args.baseline, cands, configs, staging)
    except rc.RunConfigError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"FAIL: {exc}")
        return 2
    pairs = pair_sides(sides_info)
    keys = [k for k, _, _ in pairs]
    pythons = {s: i["python"] for s, i in sides_info.items()}
    commits = {s: i["commit"] for s, i in sides_info.items()}
    commits["base"] = commits[base_sides(sides_info)[0]]
    # bootstrap seeds: per pair, so two seeds with the same commit still draw different resamples
    seed_commits = {
        k: commits[c] + ("" if b == "base" else f"/{b}") for k, c, b in pairs
    } | {"base": commits["base"]}
    order = list(dict.fromkeys(x for _, c, b in pairs for x in (b, c)))  # base k, cand k, ...

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
            for side in order:  # interleaved: paired baseline and candidate run back to back
                tasks.append({
                    "split": s["split"], "id": it["id"], "side": side, "python": pythons[side],
                    "input": str(it["input"]), "gt": str(it["gt"]) if it["gt"] else None,
                    "args": sides_info[side]["args"], "env": sides_info[side]["env"],
                })  # fmt: skip
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows = []
    t0 = time.time()
    import multiprocessing as mp

    slots = mp.Manager().Queue()
    for k in range(args.workers):
        slots.put(k)
    try:
        with ProcessPoolExecutor(
            args.workers, initializer=_pin_worker, initargs=(slots, args.cores_per_plan)
        ) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for k, fu in enumerate(as_completed(futs), 1):
                rows.append(fu.result())
                if k % 50 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} ({time.time() - t0:.0f} s)", flush=True)
        changed = staged_checkpoints_changed(sides_info)
    finally:
        if fresh_dir is not None:
            shutil.rmtree(fresh_dir, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    if changed:
        print("\n".join(f"FAIL: {c}" for c in changed))
        return 2

    rejected = [r for r in rows if r.get("rejected")]
    res = compare(rows, splits, pairs, seed_commits)
    decisions = {}
    for side in keys:
        names, label = primary_of(res[side], test_real_pages)
        deltas = {n: r["deltas"] for n, r in res[side].items()}
        kinds = {n: r["kinds"] for n, r in res[side].items()}
        fams = {n: r["family"] for n, r in res[side].items()}
        primary = "primary"
        if names:
            deltas[primary] = pooled(res[side], names, side, seed_commits)
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
    heads = {s: primary_headline(res[s], test_real_pages, s, seed_commits) for s in keys}
    worse = min(keys, key=lambda s: heads[s][0])
    pair_info = {k: sides_info[c] for k, c, _ in pairs}  # rule 5 counts the candidate's seeds
    rule5 = seed_rule(pair_info, {s: decisions[s].passed for s in keys}, heads)
    decisions[keys[0]].rules.append(rule5)
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
        "cores_per_plan": args.cores_per_plan,
        "notes": notes + ([f"{len(rejected)} rejected results"] if rejected else []),
        "pairs": [pair_label(sides_info, p) for p in pairs],
        "pair_base": {k: b for k, _, b in pairs},
        "sides": {
            s: {k: v for k, v in i.items() if k not in ("python", "env")}
            for s, i in sides_info.items()
        },
        "run_configs": {
            str(wt): {"state": c.state, "summary": c.summary(), "text": c.text}
            for wt, c in configs.items()
        },
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
    head = heads[worse][1]
    if len(keys) > 1:
        head = f"worse seed {worse}: {head}"
    split_list = ", ".join(f"{n} ({r['n']})" for n, r in res[keys[0]].items())
    branch_cell = args.branch if kind == "locked" else f"{args.branch} ({kind})"
    short = "PASS" if decision == "PASS" else "FAIL" if decision == "FAIL" else kind
    append_log(
        args.log,
        f"| {date} | {kind} | {branch_cell} | {log_cell(sides_info, [c for _, c, _ in pairs])} "
        f"| {log_cell(sides_info, base_sides(sides_info))} "
        f"| {split_list} | {head} | {short} | {args.job_id} "
        f"({str(out).replace(str(Path.home()), '~')}) |",
    )
    print(f"results: {out}; log: {args.log}")
    if rejected:
        return 2
    return 0 if (passed or kind != "locked") else 1


if __name__ == "__main__":
    sys.exit(main())
