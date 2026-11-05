"""Real MLX tests with the 0.5B 4-bit toy model (a few minutes in total)."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.manifest import stage_output
from zllf.core.offline import block_egress
from zllf.core.schemas import TrialParams, TrialResult
from zllf.stages import get_stage
from zllf.stages.train.hardware import find_model_config
from zllf.stages.train.trainer import MLXLoRATrainer, TrainJob

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(platform.machine() != "arm64" or platform.system() != "Darwin",
                       reason="needs Apple Silicon"),
    pytest.mark.skipif(not find_model_config(MODEL), reason="toy model not in HF cache"),
]


def _data(d: Path, n: int = 40) -> tuple[Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"messages": [
        {"role": "user", "content": f"What is the code for item {i}?"},
        {"role": "assistant", "content": f"The code for item {i} is ZX-{i * 7 % 97:02d}."}]}
        for i in range(n)]
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:8]) + "\n")
    return d / "train.jsonl", d / "valid.jsonl"


def _job(tmp_path: Path, **kw: object) -> TrainJob:
    train, valid = _data(tmp_path / "data")
    p = TrialParams(rank=8, alpha=16, learning_rate=2e-4, target_modules="all", dropout=0.0,
                    weight_decay=0.01, replay_ratio=0.1, batch_size=4, grad_accumulation=1)
    args: dict = dict(params=p, run_dir=tmp_path / "trial", train_file=train,
                      valid_file=valid, anchor_file=valid, model=MODEL, iters=20,
                      max_seq_length=256, num_layers=8, eval_every=10, save_every=5, seed=7)
    args.update(kw)
    return TrainJob(**args)


def test_lora_run_lowers_loss_and_adapter_loads(tmp_path: Path) -> None:
    out = MLXLoRATrainer().train(_job(tmp_path))
    assert out.status == "complete", out.error
    hist = out.metrics["history"]
    assert hist["train"][-1][1] < hist["train"][0][1]
    assert hist["val"][-1][1] < hist["val"][0][1]
    assert out.metrics["peak_memory_gb"] < 4
    ad = out.adapter_dir
    assert ad is not None and (ad / "adapters.safetensors").exists()
    from mlx_lm import load

    load(MODEL, adapter_path=str(ad))  # loadable by the inference backend


def test_resume_after_killed_subprocess(tmp_path: Path) -> None:
    job = _job(tmp_path, extra={"fail_at_step": 12, "fail_exit_code": 137})
    events: list[dict] = []
    out = MLXLoRATrainer().train(job, lambda ev: events.append(ev) or False)
    assert out.status == "complete", out.error
    assert [a["kind"] for a in out.attempts] == ["crash", "ok"]
    resumed = [e for e in events if e["event"] == "resumed"]
    assert resumed and resumed[0]["step"] == 10  # last checkpoint before the kill
    assert out.metrics["resumed_from"] == 10
    assert out.metrics["history"]["train"][-1][0] == 20


def test_hpo_smoke_two_trials_toy(tmp_path: Path) -> None:
    from train_helpers import make_ctx

    cfg = load_config("mode_a", "toy", {
        "models": {"max_seq_len": 256},
        "search_space": {"rank": [4, 8]},
        "hpo": {"n_trials_min": 2, "n_trials_max": 2, "trial_data_fraction": 0.5},
        "lora": {"batch_size": 4, "epochs": 1},
        "budget": {"total_seconds": 3600},
    })
    ctx = make_ctx(tmp_path, cfg, n_train=40, n_val=8, n_unans=4)
    with block_egress(), stage_output(ctx.paths, "train") as out:
        get_stage("train").run(ctx, out)
    out = ctx.paths.stage("train")
    trials = [TrialResult.model_validate(t)
              for t in json.loads((out / "trials.json").read_text())]
    assert len(trials) == 2
    assert trials[0].state == "complete" and trials[0].objective is not None
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["final"]["val_loss"] < rep["dry_run"]["base_losses"]["valid"]
    assert rep["final"]["halluc_loss"] is not None
    assert (out / "adapter" / "adapters.safetensors").exists()
    print("REPORT", json.dumps({k: rep[k] for k in ("final", "dry_run", "timings",
                                                     "memory_model")}, default=str))
