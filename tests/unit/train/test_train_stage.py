from __future__ import annotations

import json
import struct
from pathlib import Path

from train_helpers import make_ctx

from zllf.core.config import load_config
from zllf.core.manifest import stage_output
from zllf.core.offline import block_egress
from zllf.core.schemas import HardwareProfile, TrialParams, TrialResult
from zllf.stages import get_stage
from zllf.stages.train.hardware import MemoryModel, ModelShape, profile_hardware


def test_profile_hardware_fake() -> None:
    hw = profile_hardware(load_config("mode_a", "fake"))
    assert hw.memory_total_gb > 0 and hw.cpu_cores >= 1 and hw.backend == "fake"


def test_memory_model_calibration_and_fit() -> None:
    shape = ModelShape(hidden_size=896, intermediate_size=4864, num_layers=24, kv_dim=128)
    m = MemoryModel.calibrate(base_bytes=1e9, peak_bytes=5e9, rank=128, batch_size=4,
                              seq_len=1024, target_modules="all", shape=shape)
    assert abs(m.predict(128, 4, 1024, "all") - 5e9) < 1e3  # reproduces the measurement
    assert m.predict(8, 4, 1024, "attention") < m.predict(128, 4, 1024, "all")
    fit = m.fit_batch(rank=128, batch_size=4, grad_accumulation=1, seq_len=1024,
                      target_modules="all", limit_bytes=3.5e9)
    assert fit is not None and fit[0] < 4 and fit[0] * fit[1] == 4
    assert m.fit_batch(rank=128, batch_size=4, grad_accumulation=1, seq_len=1024,
                       target_modules="all", limit_bytes=0.5e9) is None


def test_train_stage_fake_end_to_end(tmp_path: Path) -> None:
    cfg = load_config("mode_a", "fake")
    ctx = make_ctx(tmp_path, cfg)
    events: list[dict] = []
    ctx.event_sink = events.append
    with block_egress(), stage_output(ctx.paths, "train") as out:
        result = get_stage("train").run(ctx, out)
    out = ctx.paths.stage("train")
    HardwareProfile.model_validate_json((out / "hw_profile.json").read_text())
    trials = [TrialResult.model_validate(t)
              for t in json.loads((out / "trials.json").read_text())]
    assert cfg.hpo.n_trials_min <= len(trials) <= cfg.hpo.n_trials_max
    assert any(t.state == "complete" for t in trials)
    st = (out / "adapter" / "adapters.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", st[:8])
    assert json.loads(st[8:8 + n])  # valid safetensors header
    ac = json.loads((out / "adapter" / "adapter_config.json").read_text())
    assert ac["fine_tune_type"] == "lora" and "lora_parameters" in ac
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["final"]["val_loss"] is not None
    assert rep["replay"]["sample_ids"]
    assert rep["replay"]["beta"] >= min(rep["replay"]["beta_min"], 0.30) - 1e-9
    assert rep["budget_plan"]["n_trials"] >= 1
    assert set(rep["loss_curves"]) == {"train", "val"}
    assert all(len(pt) == 2 for pt in rep["loss_curves"]["train"])
    TrialParams.model_validate(rep["final_params"])
    roles = {d["role"]: d for d in rep["datasets"]}
    assert set(roles) == {"train", "val", "replay"}
    assert all(len(d["sha256"]) == 64 for d in rep["datasets"])
    assert result.metrics["n_trials"] == len(trials)
    # scratch cleaned, study kept for audit; no document text in report/events
    work = ctx.paths.work("train")
    assert not (work / "trials").exists() and not (work / "data").exists()
    blob = json.dumps(rep) + json.dumps(events, default=str)
    assert "ZX-" not in blob and "item 3" not in blob
    assert any(e["type"] == "train_metrics" for e in events)
    assert any(e["type"] == "trial" for e in events)


def test_train_stage_hpo_disabled(tmp_path: Path) -> None:
    cfg = load_config("mode_a", "fake", {"hpo": {"enabled": False}})
    ctx = make_ctx(tmp_path, cfg)
    with stage_output(ctx.paths, "train") as out:
        get_stage("train").run(ctx, out)
    out = ctx.paths.stage("train")
    assert json.loads((out / "trials.json").read_text()) == []
    rep = json.loads((out / "train_report.json").read_text())
    assert rep["final_params"]["rank"] == cfg.lora.rank
    assert (out / "adapter" / "adapter_config.json").exists()


def test_train_stage_is_deterministic(tmp_path: Path) -> None:
    def run(d: Path) -> list:
        ctx = make_ctx(d, load_config("mode_a", "fake"))
        with stage_output(ctx.paths, "train") as out:
            get_stage("train").run(ctx, out)
        return json.loads((ctx.paths.stage("train") / "trials.json").read_text())

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    strip = lambda ts: [(t["params"], t["state"], t["objective"]) for t in ts]  # noqa: E731
    assert strip(a) == strip(b)
