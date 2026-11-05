from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from zllf.core.errors import ZLLFError
from zllf.core.schemas import TrialParams
from zllf.stages.train.mlx_worker import EXIT_MODEL_MISSING, EXIT_OOM, PREFIX, is_oom_message
from zllf.stages.train.trainer import (
    FakeTrainer,
    MLXLoRATrainer,
    TrainJob,
    classify,
    oom_fallback,
)

P = TrialParams(rank=8, alpha=16, learning_rate=1e-4, target_modules="all", dropout=0.0,
                weight_decay=0.0, replay_ratio=0.1, batch_size=4, grad_accumulation=2)


def _job(tmp_path: Path) -> TrainJob:
    return TrainJob(params=P, run_dir=tmp_path / "trial", train_file=tmp_path / "t.jsonl",
                    valid_file=tmp_path / "v.jsonl", model="m", iters=20, max_seq_length=128,
                    eval_every=5, save_every=5)


class FakeRunner:
    """Scripted subprocess: each call pops (returncode, stdout events, stderr)."""

    def __init__(self, script: list[tuple[int, list[dict], str]]) -> None:
        self.script = list(script)
        self.jobs: list[dict] = []

    def __call__(self, cmd: list[str], on_line: Callable[[str], bool],
                 env: dict | None = None) -> tuple[int, str, bool]:
        job = json.loads(Path(cmd[-1]).read_text())
        self.jobs.append(job)
        rc, events, stderr = self.script.pop(0)
        killed = False
        for ev in events:
            if on_line(PREFIX + json.dumps(ev)):
                killed = True
                return -15, "", True
        if rc == 0:
            Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
            Path(job["adapter_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(job["out_dir"]) / "result.json").write_text(json.dumps(
                {"val_loss": 1.0, "anchor_loss": 1.5, "halluc_loss": 0.5}))
        return rc, stderr, killed


def test_oom_signatures() -> None:
    assert is_oom_message("[METAL] Command buffer execution failed: Insufficient Memory "
                          "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)")
    assert is_oom_message("RuntimeError: [metal::malloc] Attempting to allocate 9 bytes")
    assert not is_oom_message("ValueError: bad shape")


def test_classify() -> None:
    assert classify(0, "", False, []).kind == "ok"
    assert classify(EXIT_OOM, "", False, []).kind == "oom"
    assert classify(1, "libc++abi: std::bad_alloc", False, []).kind == "oom"
    assert classify(-9, "", False, []).kind == "crash"
    assert classify(137, "", False, []).kind == "crash"
    assert classify(EXIT_MODEL_MISSING, "", False, []).kind == "model_missing"
    assert classify(1, "boom", False, []).kind == "error"
    assert classify(-15, "", True, []).kind == "pruned"


def test_oom_fallback_ladder(tmp_path: Path) -> None:
    j = _job(tmp_path)
    j1 = oom_fallback(j)
    assert j1 and j1.params.batch_size == 2 and j1.params.grad_accumulation == 4
    assert j1.iters == 40  # same number of samples
    j2 = oom_fallback(j1)
    assert j2 and j2.params.batch_size == 1 and j2.params.grad_accumulation == 8
    j3 = oom_fallback(j2)
    assert j3 and j3.grad_checkpoint and j3.params.batch_size == 1
    assert oom_fallback(j3) is None


def test_oom_retry_then_success(tmp_path: Path) -> None:
    runner = FakeRunner([(EXIT_OOM, [{"event": "oom", "error": "out of memory"}], ""),
                         (1, [], "kIOGPUCommandBufferCallbackErrorOutOfMemory"),
                         (0, [], "")])
    out = MLXLoRATrainer(runner=runner).train(_job(tmp_path))
    assert out.status == "complete"
    assert [j["batch_size"] for j in runner.jobs] == [4, 2, 1]
    assert [j["grad_accumulation"] for j in runner.jobs] == [2, 4, 8]
    assert len(out.attempts) == 3


def test_oom_exhausted_marks_failed_oom(tmp_path: Path) -> None:
    runner = FakeRunner([(EXIT_OOM, [], "")] * 4)
    out = MLXLoRATrainer(runner=runner, max_oom_retries=3).train(_job(tmp_path))
    assert out.status == "failed_oom"
    assert len(runner.jobs) == 4
    assert runner.jobs[-1]["grad_checkpoint"] is True


def test_crash_resumes_same_config(tmp_path: Path) -> None:
    runner = FakeRunner([(-9, [{"event": "checkpoint", "step": 10}], ""), (0, [], "")])
    out = MLXLoRATrainer(runner=runner).train(_job(tmp_path))
    assert out.status == "complete"
    assert runner.jobs[0]["batch_size"] == runner.jobs[1]["batch_size"] == 4


def test_pruning_kills_run(tmp_path: Path) -> None:
    runner = FakeRunner([(0, [{"event": "eval", "step": 5, "progress": 0.25,
                               "val_loss": 3.0}], "")])
    out = MLXLoRATrainer(runner=runner).train(_job(tmp_path), lambda ev: ev["event"] == "eval")
    assert out.status == "pruned"
    assert out.metrics["val_loss"] == 3.0


def test_model_missing_raises(tmp_path: Path) -> None:
    runner = FakeRunner([(EXIT_MODEL_MISSING, [], "")])
    with pytest.raises(ZLLFError):
        MLXLoRATrainer(runner=runner).train(_job(tmp_path))


def test_job_dict_maps_params(tmp_path: Path) -> None:
    d = MLXLoRATrainer.job_dict(_job(tmp_path))
    assert d["scale"] == 2.0 and d["rank"] == 8 and d["target_modules"] == "all"
    assert d["save_every"] % d["grad_accumulation"] == 0
    assert d["warmup_steps"] >= 1


def test_fake_trainer_writes_adapter(tmp_path: Path) -> None:
    out = FakeTrainer().train(_job(tmp_path))
    assert out.status == "complete"
    assert (out.adapter_dir / "adapters.safetensors").stat().st_size > 8  # type: ignore[operator]
    cfg = json.loads((out.adapter_dir / "adapter_config.json").read_text())  # type: ignore[operator]
    assert cfg["lora_parameters"]["rank"] == 8
    h = out.metrics["history"]["train"]
    assert h[-1][1] < h[0][1]
