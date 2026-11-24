"""Declared run config (kelpod-run.toml) for locked runs: parsing, integrity checks, checkpoint
staging, seeds as sides, rule 5 and an end-to-end smoke run with a fake pipeline (no torch)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from kelpod.evaluation import locked_runner
from kelpod.evaluation import run_config as rc
from kelpod.evaluation.stats import RuleResult

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "locked_eval_rc", ROOT / "scripts" / "locked_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


le = _load()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(wt: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(wt), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True,
    )  # fmt: skip


def _repo(path: Path, config: str | None, *, commit_config: bool = True) -> Path:
    """A tiny git worktree: root and training/ uv projects, a clean training manifest."""
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (path / "training" / "manifests").mkdir(parents=True)
    (path / "training" / "pyproject.toml").write_text("[project]\nname='t'\n")
    (path / "training" / "manifests" / "synth.json").write_text(
        json.dumps({"source": "synth-train", "files": [{"path": "a.png", "sha256": "0" * 64,
                                                        "style": "train"}]})
    )  # fmt: skip
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "init")
    if config is not None:
        (path / rc.FILENAME).write_text(config)
        if commit_config:
            _git(path, "add", rc.FILENAME)
            _git(path, "commit", "-qm", "run config")
    return path


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Two fake checkpoints (two 'seeds') outside every worktree; eval data root redirected."""
    monkeypatch.setenv("KELPOD_EVAL_DATA", str(tmp_path / "evaldata"))
    d = tmp_path / "models"
    d.mkdir()
    out = []
    for k in range(2):
        p = d / f"s{k}.pt"
        p.write_bytes(f"fake weights {k}".encode())
        out.append((p, _sha(p.read_bytes())))
    return out


def _config(ckpts, extra: str = "") -> str:
    body = (
        'version = 1\nsegment_backend = "learned"\nocr_backend = "rapidocr"\n'
        'python_project = "training"\n' + extra
    )
    for k, (p, sha) in enumerate(ckpts):
        body += f'\n[[checkpoints]]\npath = "{p}"\nsha256 = "{sha}"\nseed = {k}\n'
    return body


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------


def test_absent_config_means_todays_defaults(tmp_path):
    cfg = rc.load(tmp_path)
    assert cfg.runner_args() == [] and cfg.python_project == "." and cfg.checkpoints == []
    assert locked_runner._parse(["plan.pdf", "r.json"])[2] == {}


def test_parse_valid(models):
    cfg = rc.parse(_config(models))
    assert cfg.runner_args() == ["--segment-backend", "learned", "--ocr-backend", "rapidocr"]
    assert [c.seed for c in cfg.checkpoints] == ["0", "1"]
    assert locked_runner._parse(["p", "r", *cfg.runner_args()])[2] == {
        "segment_backend": "learned", "ocr_backend": "rapidocr"
    }  # fmt: skip


@pytest.mark.parametrize(
    "text, msg",
    [
        ('version = 1\nfoo = 1\n', "unknown keys"),
        ('version = 2\n', "version"),
        ('version = 1\nsegment_backend = "x; rm -rf /"\n', "segment_backend"),
        ('version = 1\npython_project = "../other"\n', "python_project"),
        ('version = 1\npython_project = "/abs"\n', "python_project"),
        ('version = 1\nmanifests = ["/abs/m.json"]\n', "relative path"),
        ('version = 1\nmanifests = ["~/m.json"]\n', "relative path"),
        ('version = 1\n[[checkpoints]]\npath = "/m.pt"\nsha256 = "abc"\n', "sha256"),
        ('version = 1\n[[checkpoints]]\npath = "/m.pt"\nsha256 = "' + "a" * 64 + '"\nenv = 1\n',
         "unknown keys"),
        ('version = 1\n[[checkpoints]]\npath = "/a.pt"\nsha256 = "' + "a" * 64 + '"\n'
         '[[checkpoints]]\npath = "/b.pt"\nsha256 = "' + "a" * 64 + '"\n', "same sha256"),
    ],
)  # fmt: skip
def test_parse_rejects(text, msg):
    with pytest.raises(rc.RunConfigError, match=msg):
        rc.parse(text)


def test_runner_rejects_unknown_options():
    with pytest.raises(SystemExit):
        locked_runner._parse(["p", "r", "--weights", "x"])


# --------------------------------------------------------------------------------------------
# integrity checks
# --------------------------------------------------------------------------------------------


def test_check_ok_and_git_state(tmp_path, models):
    wt = _repo(tmp_path / "cand", _config(models))
    cfg = rc.load(wt)
    assert rc.git_state(wt) == "committed"
    assert rc.check(wt, cfg, {}) == []
    (wt / rc.FILENAME).write_text(_config(models[:1]))
    assert rc.git_state(wt) == "modified"
    wt2 = _repo(tmp_path / "cand2", _config(models), commit_config=False)
    assert rc.git_state(wt2) == "untracked"
    assert rc.git_state(_repo(tmp_path / "base", None)) == "absent"


def test_check_sha_mismatch_fails_loudly(tmp_path, models):
    wt = _repo(tmp_path / "cand", _config(models))
    models[1][0].write_bytes(b"silently retrained")
    problems = rc.check(wt, rc.load(wt), {})
    assert len(problems) == 1 and "seed 1" in problems[0] and "sha256 mismatch" in problems[0]


def test_check_refuses_checkpoint_inside_worktree_or_eval_data(tmp_path, models):
    wt = _repo(tmp_path / "cand", None)
    inside = wt / "model.pt"
    inside.write_bytes(b"w")
    sealed = tmp_path / "evaldata" / "test-synth" / "m.pt"
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"s")
    cfg = rc.parse(_config([(inside, _sha(b"w")), (sealed, _sha(b"s"))]))
    problems = rc.check(wt, cfg, {})
    assert any("seed 0" in p and "outside the worktrees" in p for p in problems)
    assert any("seed 1" in p and "evaluation data or sealed" in p for p in problems)
    # a symlink from outside into the eval data is resolved first
    link = tmp_path / "models" / "link.pt"
    link.symlink_to(sealed)
    problems = rc.check(wt, rc.parse(_config([(link, _sha(b"s"))])), {})
    assert any("evaluation data or sealed" in p for p in problems)


def test_check_refuses_training_manifest_overlapping_test(tmp_path, models):
    test_sha = "f" * 64
    manifests = {
        "test-synth": {
            "split": "test-synth",
            "family": "test-synth",
            "status": "frozen",
            "kind": "synth",
            "data_dir": str(tmp_path / "evaldata" / "ts"),
            "items": [{"id": "s1", "files": {"plan.pdf": test_sha}}],
        }
    }
    wt = _repo(tmp_path / "cand", _config(models, 'manifests = ["training/manifests/leak.json"]\n'))
    (wt / "training" / "manifests" / "leak.json").write_text(
        json.dumps({"files": [{"path": "x.png", "sha256": test_sha}]})
    )
    problems = rc.check(wt, rc.load(wt), manifests)
    assert len(problems) == 2 and all("hash of test file test-synth/s1" in p for p in problems)
    # default (training/manifests/*.json) also finds it
    wt2 = _repo(tmp_path / "cand2", _config(models))
    (wt2 / "training" / "manifests" / "leak.json").write_text(
        json.dumps({"files": [{"path": "x.png", "sha256": test_sha}]})
    )
    assert any("test-synth/s1" in p for p in rc.check(wt2, rc.load(wt2), manifests))


def test_check_needs_a_training_manifest_and_a_project(tmp_path, models):
    wt = _repo(tmp_path / "cand", _config(models[:1], 'manifests = ["training/missing.json"]\n'))
    problems = rc.check(wt, rc.load(wt), {})
    assert any("not found" in p for p in problems)
    cfg = rc.parse(_config(models[:1]).replace('"training"', '"nowhere"'))
    assert any("no pyproject.toml" in p for p in rc.check(wt, cfg, {}))


def test_stage_checkpoint_is_read_only_copy(tmp_path, models):
    ck = rc.parse(_config(models)).checkpoints[0]
    dest = rc.stage_checkpoint(ck, tmp_path / "stage" / "cand0")
    assert dest.read_bytes() == models[0][0].read_bytes()
    assert not dest.stat().st_mode & 0o222


def test_clean_env_never_leaks_operator_checkpoint(monkeypatch):
    monkeypatch.setenv("KELPOD_SEG_CHECKPOINT", "/operator/model.pt")
    monkeypatch.setenv("PYTHONPATH", "/evil")
    env = le._clean_env()
    assert "KELPOD_SEG_CHECKPOINT" not in env and "PYTHONPATH" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert le._clean_env({"KELPOD_SEG_CHECKPOINT": "/s.pt"})["KELPOD_SEG_CHECKPOINT"] == "/s.pt"


# --------------------------------------------------------------------------------------------
# rule 5
# --------------------------------------------------------------------------------------------


def _info(sha):
    return {"worktree": "/w", "commit": "c" * 40, "args": [], "seed": "0", "sha256": sha}


def test_seed_rule_needs_two_seeds_and_the_worse_one():
    heads = {"cand0": ((0.10, 0.05), "+0.100"), "cand1": ((0.02, -0.01), "+0.020")}
    two = {"cand0": _info("a" * 64), "cand1": _info("b" * 64)}
    r = le.seed_rule(two, {"cand0": True, "cand1": False}, heads)
    assert r.passed is False and "worse seed: cand1" in r.detail
    assert le.seed_rule(two, {"cand0": True, "cand1": True}, heads).passed is True
    one = {"cand0": _info("a" * 64)}
    r = le.seed_rule(one, {"cand0": True}, {"cand0": heads["cand0"]})
    assert r.passed is False and "at least 2 training seeds" in r.detail
    untrained = {"cand0": _info(None)}
    assert le.seed_rule(untrained, {"cand0": True}, {"cand0": heads["cand0"]}).passed is None
    assert isinstance(r, RuleResult)


# --------------------------------------------------------------------------------------------
# end to end: smoke run with a fake pipeline interpreter
# --------------------------------------------------------------------------------------------


class _SyncPool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, arg):
        from concurrent.futures import Future

        f = Future()
        f.set_result(fn(arg))
        return f


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Fake 'worktree python': records argv, visible files and the checkpoint it was given."""
    from kelpod.evaluation.synth import generate, save

    item = save(generate(1, 0.0), tmp_path / "item")
    rec = tmp_path / "rec"
    rec.mkdir()
    fake = tmp_path / "fakepy"
    fake.write_text(
        "#!/bin/sh\n"
        'f="$FAKE_REC/$(date +%s%N)-$$.txt"\n'
        '{ echo "args: $*"; echo "ls: $(ls -A | tr "\\n" " ")"; '
        'echo "ckpt: ${KELPOD_SEG_CHECKPOINT:-none}"; echo "cuda: [$CUDA_VISIBLE_DEVICES]"; '
        'if [ -n "$KELPOD_SEG_CHECKPOINT" ]; then echo "content: $(cat "$KELPOD_SEG_CHECKPOINT")";'
        " fi; } > \"$f\"\n"
        'printf \'{"ok": false, "error": "fake"}\' > "$4"\n'
    )  # fmt: skip
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_REC", str(rec))
    monkeypatch.setenv("KELPOD_SEG_CHECKPOINT", "/operator/should/not/leak.pt")
    synced = []
    monkeypatch.setattr(
        le, "worktree_python", lambda wt, p=".": synced.append((wt, p)) or str(fake)
    )
    monkeypatch.setattr(le, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(le, "_pin_worker", lambda *a: None)
    monkeypatch.setattr("kelpod.evaluation.splits.load_manifests", lambda root=None: {})
    split = {"split": "val-x", "family": "val", "kind": "synth"}
    monkeypatch.setattr(le, "available_splits", lambda m, w: ([split], []))
    monkeypatch.setattr(
        le, "split_items",
        lambda s, f, lim: [{"id": "i1", "input": item / "plan.pdf",
                            "gt": item / "ground_truth.json"}],
    )  # fmt: skip
    return {"rec": rec, "synced": synced}


def _records(rec: Path) -> list[dict]:
    out = []
    for f in sorted(rec.iterdir()):
        d = dict(ln.split(": ", 1) for ln in f.read_text().splitlines() if ": " in ln)
        out.append(d)
    return out


def test_end_to_end_two_seeds(tmp_path, models, fake_run):
    cand = _repo(tmp_path / "cand", _config(models))
    base = _repo(tmp_path / "base", None)
    log, out = tmp_path / "log.md", tmp_path / "out"
    code = le.main([
        "--candidate", str(cand), "--baseline", str(base), "--branch", "exp/x", "--smoke",
        "--limit", "1", "--workers", "1", "--log", str(log), "--out", str(out),
    ])  # fmt: skip
    assert code == 0
    recs = _records(fake_run["rec"])
    assert len(recs) == 3  # base + 2 seeds
    assert all(r["ls"].split() == ["plan.pdf"] for r in recs)
    assert all(r["cuda"] == "[]" for r in recs)
    base_r = [r for r in recs if r["ckpt"] == "none"]
    assert len(base_r) == 1 and "--segment-backend" not in base_r[0]["args"]
    seeds = sorted(r["content"] for r in recs if r["ckpt"] != "none")
    assert seeds == ["fake weights 0", "fake weights 1"]
    for r in recs:
        if r["ckpt"] != "none":
            assert "--segment-backend learned --ocr-backend rapidocr" in r["args"]
            assert not Path(r["ckpt"]).is_relative_to(cand) and "kelpod-ckpt-" in r["ckpt"]
            assert not Path(r["ckpt"]).exists()  # staging removed after the run
    assert (str(cand), "training") in [(str(w), p) for w, p in fake_run["synced"]]
    line = log.read_text().splitlines()[-1]
    assert f"ckpt s0:{models[0][1][:12]}, s1:{models[1][1][:12]}" in line
    assert "worse seed" in line
    md = (out / "comparison.md").read_text()
    assert "### Run configs (kelpod-run.toml)" in md and 'segment_backend = "learned"' in md
    assert "side cand1" in md and "5 seed stability" in md


def test_end_to_end_refuses_changed_checkpoint(tmp_path, models, fake_run):
    cand = _repo(tmp_path / "cand", _config(models))
    base = _repo(tmp_path / "base", None)
    models[0][0].write_bytes(b"swapped")
    code = le.main([
        "--candidate", str(cand), "--baseline", str(base), "--branch", "exp/x", "--smoke",
        "--limit", "1", "--workers", "1", "--log", str(tmp_path / "log.md"),
        "--out", str(tmp_path / "out"),
    ])  # fmt: skip
    assert code == 2 and _records(fake_run["rec"]) == []
    assert not (tmp_path / "log.md").exists()


def test_locked_run_refuses_uncommitted_config(tmp_path, models, fake_run, monkeypatch):
    split = {"split": "test-x", "family": "test-synth", "kind": "synth"}
    monkeypatch.setattr(le, "available_splits", lambda m, w: ([split], []))
    cand = _repo(tmp_path / "cand", _config(models), commit_config=False)
    base = _repo(tmp_path / "base", None)
    code = le.main([
        "--candidate", str(cand), "--baseline", str(base), "--branch", "exp/x",
        "--workers", "1", "--log", str(tmp_path / "log.md"), "--out", str(tmp_path / "out"),
    ])  # fmt: skip
    assert code == 2 and _records(fake_run["rec"]) == []
