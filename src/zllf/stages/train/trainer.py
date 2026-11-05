"""Trainer seam: ``MLXLoRATrainer`` (Apple Silicon, subprocess per run) and ``FakeTrainer``
(no model; deterministic dummy adapter + synthetic loss curve for CI).

A PEFT/TRL trainer for CUDA can be added behind the same interface (ADR-0001).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import TrialParams
from zllf.stages.train.mlx_worker import (
    EXIT_MODEL_MISSING,
    EXIT_OK,
    EXIT_OOM,
    PREFIX,
    is_oom_message,
)

# Called for every progress event; return True to stop the run (pruning).
MetricCallback = Callable[[dict[str, Any]], bool]
Status = Literal["complete", "pruned", "failed", "failed_oom"]


@dataclass
class TrainJob:
    """One training run (an HPO trial or the final fit)."""

    params: TrialParams
    run_dir: Path  # persistent per-run dir (checkpoints, metrics, adapter)
    train_file: Path
    valid_file: Path
    model: str
    iters: int  # micro-batches
    max_seq_length: int
    anchor_file: Path | None = None
    halluc_file: Path | None = None
    num_layers: int = -1
    seed: int = 0
    eval_every: int = 50
    save_every: int = 50
    save_seconds: float = 900.0  # NFR-REL-02: checkpoint at least every 15 min
    warmup_fraction: float = 0.05
    grad_checkpoint: bool = False
    mask_prompt: bool = True
    memory_limit_bytes: int | None = None
    max_eval_examples: int | None = 64
    extra: dict[str, Any] = field(default_factory=dict)  # passed verbatim to the worker

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"


@dataclass
class TrainOutcome:
    status: Status
    metrics: dict[str, Any] = field(default_factory=dict)
    adapter_dir: Path | None = None
    seconds: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_job: TrainJob | None = None
    error: str | None = None


@dataclass
class DryRunSpec:
    model: str
    run_dir: Path
    rank: int
    target_modules: str
    batch_size: int
    max_seq_length: int
    num_layers: int = -1
    seed: int = 0
    steps: int = 3
    anchor_file: Path | None = None
    halluc_file: Path | None = None
    valid_file: Path | None = None
    memory_limit_bytes: int | None = None
    max_eval_examples: int | None = 64


@dataclass
class DryRunResult:
    base_losses: dict[str, float | None]
    base_bytes: float
    peak_bytes: float
    tokens_per_s: float
    model_config: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    ok: bool = True
    error: str | None = None
    batch_size: int = 1


class Trainer(ABC):
    """Runs LoRA training jobs. Must never raise for a failed/OOM trial; return a status."""

    name: str

    @abstractmethod
    def dry_run(self, spec: DryRunSpec) -> DryRunResult:
        """Base-model losses on the eval sets + a short max-length run to measure memory."""

    @abstractmethod
    def train(self, job: TrainJob, on_metric: MetricCallback | None = None) -> TrainOutcome:
        """Train (resuming from a checkpoint in ``job.run_dir`` if present)."""


# ---------------------------------------------------------------- OOM-safe retry policy
def oom_fallback(job: TrainJob) -> TrainJob | None:
    """Next, less memory-hungry configuration (NFR-REL-01): halve micro-batch and double
    accumulation; at micro-batch 1 enable gradient checkpointing; else give up."""
    p = job.params
    if p.batch_size > 1:
        new_p = p.model_copy(update={"batch_size": max(1, p.batch_size // 2),
                                     "grad_accumulation": p.grad_accumulation * 2})
        # keep the number of samples seen constant
        return replace(job, params=new_p, iters=job.iters * 2,
                       eval_every=job.eval_every * 2, save_every=job.save_every * 2)
    if not job.grad_checkpoint:
        return replace(job, grad_checkpoint=True)
    return None


@dataclass
class AttemptResult:
    """Classified outcome of one subprocess attempt."""

    kind: Literal["ok", "oom", "pruned", "crash", "model_missing", "error"]
    returncode: int
    error: str | None = None


def run_with_retries(
    job: TrainJob,
    attempt: Callable[[TrainJob], AttemptResult],
    *,
    max_oom_retries: int = 3,
    max_crash_retries: int = 2,
) -> tuple[AttemptResult, TrainJob, list[dict[str, Any]]]:
    """Run ``attempt`` with the OOM/crash policy; never raises for OOM or crashes.

    - OOM → :func:`oom_fallback` and restart from scratch (≤ ``max_oom_retries``), then
      ``oom`` is returned (caller marks the trial ``failed_oom``).
    - crash (e.g. killed, power loss) → rerun; the worker resumes from its last checkpoint.
    - model missing → raised as :class:`ZLLFError` (not recoverable by retrying).
    """
    log: list[dict[str, Any]] = []
    ooms = crashes = 0
    current = job
    while True:
        res = attempt(current)
        log.append({"kind": res.kind, "returncode": res.returncode,
                    "batch_size": current.params.batch_size,
                    "grad_accumulation": current.params.grad_accumulation,
                    "grad_checkpoint": current.grad_checkpoint, "error": res.error})
        if res.kind in ("ok", "pruned", "error"):
            return res, current, log
        if res.kind == "model_missing":
            raise ZLLFError(ErrorCode.MODEL_MISSING, res.error or current.model)
        if res.kind == "oom":
            ooms += 1
            nxt = oom_fallback(current)
            if nxt is None or ooms > max_oom_retries:
                return res, current, log
            shutil.rmtree(current.run_dir / "ckpt", ignore_errors=True)  # shapes changed
            current = nxt
            continue
        crashes += 1  # "crash"
        if crashes > max_crash_retries:
            return AttemptResult("error", res.returncode, res.error or "worker crashed"), \
                current, log


# ---------------------------------------------------------------- subprocess runner
class ProcessRunner(Protocol):
    def __call__(self, cmd: list[str], on_line: Callable[[str], bool],
                 env: dict[str, str] | None = None) -> tuple[int, str, bool]:
        """Run ``cmd``; feed each stdout line to ``on_line`` (True → kill the process).
        Returns (returncode, stderr tail, killed_by_us)."""
        ...


def subprocess_runner(cmd: list[str], on_line: Callable[[str], bool],
                      env: dict[str, str] | None = None) -> tuple[int, str, bool]:
    """Default runner: streams stdout, keeps the last 50 stderr lines (for OOM detection)."""
    import threading

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, bufsize=1, start_new_session=True)
    tail: deque[str] = deque(maxlen=50)

    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip())

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    killed = False
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if on_line(line.rstrip("\n")) and not killed:
                killed = True
                proc.send_signal(signal.SIGTERM)
    finally:
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        t.join(timeout=5)
    return rc, "\n".join(tail), killed


def classify(rc: int, stderr: str, killed: bool, events: list[dict[str, Any]]) -> AttemptResult:
    """Map a worker exit to an :class:`AttemptResult`."""
    last_err = next((e.get("error") for e in reversed(events)
                     if e.get("event") in ("oom", "error", "model_missing")), None)
    if killed:
        return AttemptResult("pruned", rc)
    if rc == EXIT_OK:
        return AttemptResult("ok", rc)
    if rc == EXIT_OOM or is_oom_message(stderr) or (last_err and is_oom_message(last_err)):
        return AttemptResult("oom", rc, last_err or "out of memory")
    if rc == EXIT_MODEL_MISSING:
        return AttemptResult("model_missing", rc, last_err)
    if rc < 0 or rc >= 128:  # killed by a signal (SIGKILL, power loss, jetsam ...)
        return AttemptResult("crash", rc, f"signal {rc}")
    return AttemptResult("error", rc, last_err or stderr[-300:] or f"exit {rc}")


class MLXLoRATrainer(Trainer):
    """mlx-lm LoRA on (4-bit) MLX models; one ``mlx_worker`` subprocess per attempt."""

    name = "mlx"

    def __init__(self, runner: ProcessRunner = subprocess_runner,
                 python: str = sys.executable, max_oom_retries: int = 3) -> None:
        self.runner = runner
        self.python = python
        self.max_oom_retries = max_oom_retries

    def _cmd(self, job_file: Path) -> list[str]:
        return [self.python, "-m", "zllf.stages.train.mlx_worker", str(job_file)]

    @staticmethod
    def job_dict(job: TrainJob) -> dict[str, Any]:
        p = job.params
        updates = max(1, job.iters // max(p.grad_accumulation, 1))
        return {
            "mode": "train",
            "model": job.model,
            "out_dir": str(job.run_dir),
            "adapter_dir": str(job.adapter_dir),
            "train_file": str(job.train_file),
            "valid_file": str(job.valid_file),
            "anchor_file": str(job.anchor_file) if job.anchor_file else None,
            "halluc_file": str(job.halluc_file) if job.halluc_file else None,
            "rank": p.rank,
            "scale": p.alpha / p.rank,
            "dropout": p.dropout,
            "target_modules": p.target_modules,
            "num_layers": job.num_layers,
            "learning_rate": p.learning_rate,
            "weight_decay": p.weight_decay,
            "warmup_steps": math.ceil(job.warmup_fraction * updates),
            "iters": job.iters,
            "batch_size": p.batch_size,
            "grad_accumulation": p.grad_accumulation,
            "max_seq_length": job.max_seq_length,
            "grad_checkpoint": job.grad_checkpoint,
            "eval_every": job.eval_every,
            "save_every": _round_up(job.save_every, p.grad_accumulation),
            "save_seconds": job.save_seconds,
            "seed": job.seed,
            "mask_prompt": job.mask_prompt,
            "memory_limit_bytes": job.memory_limit_bytes,
            "max_eval_examples": job.max_eval_examples,
            **job.extra,
        }

    def _attempt(self, job: TrainJob, on_metric: MetricCallback | None,
                 events: list[dict[str, Any]]) -> AttemptResult:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        job_file = job.run_dir / "job.json"
        job_file.write_text(json.dumps(self.job_dict(job), indent=2))
        (job.run_dir / "result.json").unlink(missing_ok=True)

        def on_line(line: str) -> bool:
            if not line.startswith(PREFIX):
                return False
            try:
                ev = json.loads(line[len(PREFIX):])
            except ValueError:
                return False
            events.append(ev)
            return bool(on_metric(ev)) if on_metric else False

        rc, stderr, killed = self.runner(self._cmd(job_file), on_line, dict(os.environ))
        return classify(rc, stderr, killed, events)

    def train(self, job: TrainJob, on_metric: MetricCallback | None = None) -> TrainOutcome:
        t0 = time.time()
        events: list[dict[str, Any]] = []
        res, final_job, log = run_with_retries(
            job, lambda j: self._attempt(j, on_metric, events),
            max_oom_retries=self.max_oom_retries)
        seconds = time.time() - t0
        if res.kind == "ok":
            result = json.loads((final_job.run_dir / "result.json").read_text())
            return TrainOutcome("complete", metrics=result, adapter_dir=final_job.adapter_dir,
                                seconds=seconds, attempts=log, final_job=final_job)
        status: Status = {"pruned": "pruned", "oom": "failed_oom"}.get(res.kind, "failed")  # type: ignore[assignment]
        metrics = _last_metrics(events)
        return TrainOutcome(status, metrics=metrics, seconds=seconds, attempts=log,
                            final_job=final_job, error=res.error)

    def dry_run(self, spec: DryRunSpec) -> DryRunResult:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "mode": "dry_run", "model": spec.model, "out_dir": str(spec.run_dir),
            "rank": spec.rank, "scale": 1.0, "dropout": 0.0,
            "target_modules": spec.target_modules, "num_layers": spec.num_layers,
            "batch_size": spec.batch_size, "max_seq_length": spec.max_seq_length,
            "seed": spec.seed, "dry_run_steps": spec.steps,
            "anchor_file": _s(spec.anchor_file), "halluc_file": _s(spec.halluc_file),
            "valid_file": _s(spec.valid_file), "memory_limit_bytes": spec.memory_limit_bytes,
            "max_eval_examples": spec.max_eval_examples,
        }
        job_file = spec.run_dir / "job.json"
        job_file.write_text(json.dumps(job, indent=2))
        events: list[dict[str, Any]] = []

        def on_line(line: str) -> bool:
            if line.startswith(PREFIX):
                events.append(json.loads(line[len(PREFIX):]))
            return False

        rc, stderr, killed = self.runner(self._cmd(job_file), on_line, dict(os.environ))
        res = classify(rc, stderr, killed, events)
        if res.kind == "model_missing":
            raise ZLLFError(ErrorCode.MODEL_MISSING, res.error or spec.model)
        if res.kind != "ok":
            base = next((e for e in events if e.get("event") == "base_losses"), {})
            return DryRunResult(base_losses={k: base.get(k) for k in ("anchor", "halluc",
                                                                       "valid")},
                                base_bytes=0.0, peak_bytes=math.inf, tokens_per_s=0.0,
                                ok=False, error=f"{res.kind}: {res.error}",
                                batch_size=spec.batch_size)
        r = json.loads((spec.run_dir / "result.json").read_text())
        return DryRunResult(base_losses=r["base_losses"], base_bytes=r["base_bytes"],
                            peak_bytes=r["peak_bytes"], tokens_per_s=r["tokens_per_s"],
                            model_config=r.get("model_config", {}), seconds=r["seconds"],
                            batch_size=spec.batch_size)


def _s(p: Path | None) -> str | None:
    return str(p) if p else None


def _round_up(x: int, m: int) -> int:
    m = max(m, 1)
    return max(m, ((x + m - 1) // m) * m)


def _last_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ev in events:
        if ev.get("event") == "eval" and ev.get("val_loss") is not None:
            out["val_loss"] = ev["val_loss"]
            out["last_step"] = ev["step"]
        if ev.get("event") == "train":
            out["train_loss"] = ev["train_loss"]
            out["peak_memory_gb"] = ev.get("peak_memory_gb")
    return out


# ---------------------------------------------------------------- fake trainer
def _write_safetensors(path: Path, tensors: dict[str, list[float]]) -> None:
    """Minimal float32 safetensors writer (no extra dependency)."""
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, values in sorted(tensors.items()):
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [len(values)],
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def fake_loss_curve(params: TrialParams, steps: int, seed: int) -> list[float]:
    """Deterministic decreasing curve; best near lr=2e-4, rank 16, all-linear."""
    lr_pen = abs(math.log10(params.learning_rate / 2e-4))
    rank_pen = abs(math.log2(params.rank / 16)) * 0.03
    tm_pen = {"all": 0.0, "attention": 0.08, "mlp": 0.05}[params.target_modules]
    floor = 0.6 + 0.4 * lr_pen + rank_pen + tm_pen + 0.2 * params.dropout
    h = int(hashlib.sha256(f"{seed}:{params.model_dump_json()}".encode()).hexdigest()[:8], 16)
    jitter = (h % 1000) / 1000 * 0.02
    k = 3.0 / max(steps, 1)
    return [floor + jitter + 1.8 * math.exp(-k * (i + 1)) for i in range(steps)]


class FakeTrainer(Trainer):
    """No model: deterministic dummy adapter + synthetic decreasing loss, in milliseconds."""

    name = "fake"

    def dry_run(self, spec: DryRunSpec) -> DryRunResult:
        return DryRunResult(
            base_losses={"anchor": 1.5, "halluc": 2.0 if spec.halluc_file else None,
                         "valid": 2.4},
            base_bytes=0.5e9, peak_bytes=0.5e9 + spec.batch_size * spec.max_seq_length * 2e4,
            tokens_per_s=5000.0,
            model_config={"hidden_size": 64, "intermediate_size": 256,
                          "num_hidden_layers": 2, "num_attention_heads": 4},
            batch_size=spec.batch_size,
        )

    def train(self, job: TrainJob, on_metric: MetricCallback | None = None) -> TrainOutcome:
        t0 = time.time()
        p = job.params
        steps = max(job.iters, 1)
        curve = fake_loss_curve(p, steps, job.seed)
        job.run_dir.mkdir(parents=True, exist_ok=True)
        history: dict[str, list[list[float]]] = {"train": [], "val": []}
        pruned = False
        for i, loss in enumerate(curve, start=1):
            ev = {"event": "train", "step": i, "train_loss": loss, "tokens_per_s": 5000.0,
                  "peak_memory_gb": 0.6}
            history["train"].append([i, loss])
            stop = on_metric(ev) if on_metric else False
            if i % max(job.eval_every, 1) == 0 or i == steps:
                v = loss + 0.15 + 0.1 * p.weight_decay
                history["val"].append([i, v])
                stop = (on_metric({"event": "eval", "step": i, "progress": i / steps,
                                   "val_loss": v})
                        if on_metric else False) or stop
            if stop:
                pruned = True
                break
        with (job.run_dir / "metrics.jsonl").open("w") as f:
            for step_i, loss in history["train"]:
                f.write(json.dumps({"event": "train", "step": step_i, "train_loss": loss}) + "\n")
        val = history["val"][-1][1] if history["val"] else None
        metrics = {"val_loss": val, "train_loss": history["train"][-1][1],
                   "anchor_loss": 1.5 + 0.4 * p.learning_rate / 5e-4 * (1 - p.replay_ratio),
                   "halluc_loss": 1.0 + p.dropout, "history": history,
                   "tokens_per_s": 5000.0, "peak_memory_gb": 0.6, "iters": steps}
        if pruned:
            return TrainOutcome("pruned", metrics=metrics, seconds=time.time() - t0,
                                final_job=job)
        ad = job.adapter_dir
        ad.mkdir(parents=True, exist_ok=True)
        seed_vals = [((i * 2654435761 + p.rank) % 1000) / 1e5 for i in range(p.rank * 2)]
        _write_safetensors(ad / "adapters.safetensors",
                           {"layers.0.self_attn.q_proj.lora_a": seed_vals,
                            "layers.0.self_attn.q_proj.lora_b": [0.0] * (p.rank * 2)})
        (ad / "adapter_config.json").write_text(json.dumps({
            "model": job.model, "fine_tune_type": "lora", "num_layers": job.num_layers,
            "lora_parameters": {"rank": p.rank, "scale": p.alpha / p.rank,
                                "dropout": p.dropout, "keys": ["self_attn.q_proj"]},
            "max_seq_length": job.max_seq_length, "fake": True,
        }, indent=2))
        return TrainOutcome("complete", metrics=metrics, adapter_dir=ad,
                            seconds=time.time() - t0, final_job=job)


def get_trainer(backend: str) -> Trainer:
    if backend == "fake":
        return FakeTrainer()
    if backend == "mlx":
        return MLXLoRATrainer()
    raise ZLLFError(ErrorCode.TRAIN_FAILED, f"no trainer for backend {backend!r} yet")


def job_summary(job: TrainJob) -> dict[str, Any]:
    d = asdict(job)
    d["params"] = job.params.model_dump()
    return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
