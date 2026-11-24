"""scripts/locked_eval.py: append-only log, run budget, skipping pending splits, input-only run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "locked_eval", ROOT / "scripts" / "locked_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


le = _load()


def test_log_is_append_only_and_budget_counts_locked_only(tmp_path):
    log = tmp_path / "TEST_LOG.md"
    le.append_log(log, "| 2026-01-01 | smoke | exp/a (smoke) | c | b | x | n/a | smoke | - |")
    le.append_log(log, "| 2026-01-01 | locked | exp/a | c | b | x | +0.1 | FAIL | - |")
    first = log.read_text()
    le.append_log(log, "| 2026-01-02 | locked | exp/b | c | b | x | +0.1 | PASS | - |")
    assert log.read_text().startswith(first)
    assert le.locked_runs_of(log, "exp/a") == 1
    assert le.locked_runs_of(log, "exp/b") == 1
    assert le.locked_runs_of(log, "exp/c") == 0
    # promotion_check.sh counts "| <branch> |": smoke lines must not match
    assert sum("| exp/a |" in ln for ln in log.read_text().splitlines()) == 1


def test_pending_and_missing_splits_are_skipped_with_a_note(tmp_path):
    manifests = {
        "test-ood-swiss": {"split": "test-ood-swiss", "status": "pending", "family": "test-ood",
                           "kind": "dataset", "note": "no renderer"},
        "test-synth": {"split": "test-synth", "status": "frozen", "family": "test-synth",
                       "kind": "synth", "data_dir": str(tmp_path / "nope"), "items": []},
    }  # fmt: skip
    splits, notes = le.available_splits(manifests, None)
    assert [s["split"] for s in splits] == ["fresh"]
    assert any("test-ood-swiss: pending" in n for n in notes)
    assert any("test-synth: data directory" in n for n in notes)
    _, notes = le.available_splits(manifests, ["test-real"])
    assert notes == ["test-real: no manifest, skipped"]


def test_runner_sees_only_the_input(tmp_path, monkeypatch):
    """run_task copies only plan.pdf into an empty dir; a fake 'pipeline' checks its cwd."""
    from kelpod.evaluation.synth import generate, save

    d = save(generate(1, 0.0), tmp_path / "item")
    fake = tmp_path / "fakepy"
    fake.write_text(
        "#!/bin/sh\n"
        # $3 = input name, $4 = result path; record the files the run can see
        'ls -A > "$4.ls"; '
        'printf \'{"ok": false, "error": "fake"}\' > "$4"\n'
    )
    fake.chmod(0o755)
    seen = {}
    orig = le.subprocess.run

    def spy(cmd, **kw):
        out = orig(cmd, **kw)
        seen["ls"] = Path(cmd[-1] + ".ls").read_text().split()
        seen["cwd"] = kw["cwd"]
        return out

    monkeypatch.setattr(le.subprocess, "run", spy)
    row = le.run_task({
        "split": "s", "id": "i", "side": "base", "python": str(fake),
        "input": str(d / "plan.pdf"), "gt": str(d / "ground_truth.json"),
    })  # fmt: skip
    assert seen["ls"] == ["plan.pdf"]
    assert not Path(seen["cwd"]).exists()  # cleaned up
    assert row["panel"]["crash"] == 1.0 and row["error"] == "fake"


def test_ground_truth_scale_is_rejected(tmp_path, monkeypatch):
    from kelpod.evaluation.synth import generate, save

    plan = generate(1, 0.0)
    d = save(plan, tmp_path / "item")
    graph = json.loads((d / "ground_truth.json").read_text())["graph"]
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"ok": True, "graph": graph, "runtime_s": 1.0}))
    fake = tmp_path / "fakepy"
    fake.write_text(f'#!/bin/sh\ncp {result} "$4"\n')
    fake.chmod(0o755)
    row = le.run_task({
        "split": "s", "id": "i", "side": "base", "python": str(fake),
        "input": str(d / "plan.pdf"), "gt": str(d / "ground_truth.json"),
    })  # fmt: skip
    assert row["rejected"] == "scale_source == ground_truth"
    assert row["panel"]["crash"] == 1.0
