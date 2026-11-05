"""Stage ④ TRAIN: hardware profile → dry run → HPO (time-budgeted) → final fit → adapter.

Inputs : ``03_synth/{train,val}.jsonl`` (SynthExample; only ``messages`` are trained on).
Outputs: ``hw_profile.json``, ``trials.json``, ``adapter/``, ``train_report.json``.
Scratch: ``ctx.paths.work("train")`` (trial dirs + checkpoints + Optuna SQLite) survives a
crash so a re-run resumes; non-winning trials and temp datasets are deleted at the end
(NFR-SEC-03 basic). No document text is logged or put into reports.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from zllf.core.config import RunConfig
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import SynthExample, TrialParams, TrialResult, read_jsonl
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.train.hardware import (
    GB,
    MemoryModel,
    ModelShape,
    find_model_config,
    metal_guidance,
    profile_hardware,
)
from zllf.stages.train.hpo import (
    HPOContext,
    HPOController,
    create_study,
    objective_value,
    phi_halluc,
)
from zllf.stages.train.planner import Plan, make_plan, train_budget_seconds
from zllf.stages.train.replay import (
    Sample,
    approx_tokens,
    beta_min,
    load_replay,
    mix_replay,
)
from zllf.stages.train.trainer import (
    DryRunResult,
    DryRunSpec,
    Trainer,
    TrainJob,
    get_trainer,
)


def _write_jsonl(rows: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")


def _to_sample(ex: SynthExample) -> Sample:
    return {"id": ex.example_id,
            "messages": [m.model_dump() for m in ex.messages]}


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("mlx", "mlx-lm", "optuna", "numpy", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def _prior_elapsed(ctx: RunContext) -> float | None:
    try:
        from zllf.core.manifest import read_manifest

        man = read_manifest(ctx.paths)
    except Exception:
        return None
    total = 0.0
    for name in ("ingest", "sanitize", "synth"):
        e = man.stages.get(name)
        if e is None or e.started_at is None or e.finished_at is None:
            return None
        total += e.finished_at - e.started_at
    return total


class TrainStage(Stage):
    name = "train"

    def __init__(self, trainer: Trainer | None = None) -> None:
        self._trainer = trainer

    # ------------------------------------------------------------ data
    @staticmethod
    def load_data(ctx: RunContext) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """(train, valid answerable, halluc unanswerable) as {"id", "messages"} samples."""
        synth = ctx.paths.stage("synth")
        train_ex = read_jsonl(synth / "train.jsonl", SynthExample)
        val_path = synth / "val.jsonl"
        val_ex = read_jsonl(val_path, SynthExample) if val_path.exists() else []
        train = [_to_sample(e) for e in train_ex if e.messages and
                 e.messages[-1].role == "assistant"]
        if not train:
            raise ZLLFError(ErrorCode.TRAIN_FAILED, "no training examples")
        valid = [_to_sample(e) for e in val_ex if e.answerable]
        halluc = [_to_sample(e) for e in val_ex if not e.answerable]
        if not valid:
            valid = [_to_sample(e) for e in val_ex]
        if not valid:  # carve a deterministic 10 % hold-out from train
            rng = ctx.rng("train-valid-carve")
            idx = list(range(len(train)))
            rng.shuffle(idx)
            k = max(1, len(train) // 10)
            hold = set(idx[:k])
            valid = [train[i] for i in sorted(hold)]
            train = [t for i, t in enumerate(train) if i not in hold]
        return train, valid, halluc

    # ------------------------------------------------------------ main
    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        t_stage = time.time()
        cfg = ctx.config
        work = ctx.paths.work("train")
        data_dir = work / "data"
        timings: dict[str, float] = {}

        # 1. hardware profile
        hw = profile_hardware(cfg, work)
        (out_dir / "hw_profile.json").write_text(hw.model_dump_json(indent=2))
        guidance = metal_guidance(hw) if cfg.models.backend == "mlx" else {}
        avail = hw.memory_available_gb * GB
        if guidance.get("memory_limit_gb"):
            avail = min(avail, guidance["memory_limit_gb"] * GB)
        limit_bytes = cfg.hpo.memory_fraction * avail
        ctx.emit("hw_profile", accelerator=hw.accelerator,
                 memory_available_gb=hw.memory_available_gb)

        # 2. data + replay
        train, valid, halluc = self.load_data(ctx)
        replay = load_replay(seed=cfg.seed)
        b_lo = beta_min(len(train))
        _write_jsonl(valid, data_dir / "valid.jsonl")
        _write_jsonl(replay.anchor, data_dir / "anchor.jsonl")
        halluc_file = data_dir / "halluc.jsonl" if halluc else None
        if halluc_file:
            _write_jsonl(halluc, halluc_file)
        trainer = self._trainer or get_trainer(cfg.models.backend)
        ctx.check_cancelled()

        # 3. dry run (cached in work dir so a resumed stage skips it)
        t = time.time()
        dry = self._dry_run(ctx, trainer, work, data_dir, halluc_file, limit_bytes)
        timings["dry_run"] = time.time() - t
        shape = ModelShape.from_model_config(dry.model_config or
                                             find_model_config(cfg.models.student))
        seq = cfg.models.max_seq_len
        max_rank = max(cfg.search_space.rank)
        if dry.ok:
            mem = MemoryModel.calibrate(
                base_bytes=dry.base_bytes, peak_bytes=dry.peak_bytes, rank=max_rank,
                batch_size=dry.batch_size, seq_len=seq, target_modules="all",
                shape=shape, num_layers=cfg.lora.num_layers)
        else:
            mem = MemoryModel.heuristic(dry.base_bytes or 1 * GB, shape, cfg.lora.num_layers)
        anchor_base = dry.base_losses.get("anchor")

        def fit_memory(p: TrialParams) -> tuple[TrialParams, bool, float] | None:
            fit = mem.fit_batch(rank=p.rank, batch_size=p.batch_size,
                                grad_accumulation=p.grad_accumulation, seq_len=seq,
                                target_modules=p.target_modules, limit_bytes=limit_bytes)
            if fit is None:
                return None
            bsz, acc, ckpt = fit
            pred = mem.predict(p.rank, bsz, seq, p.target_modules, ckpt)
            return p.model_copy(update={"batch_size": bsz, "grad_accumulation": acc}), ckpt, pred

        # 4. plan the budget
        domain_tokens = sum(approx_tokens(s["messages"]) for s in train)
        mean_beta = sum(cfg.search_space.replay_ratio) / 2
        tokens_with_replay = domain_tokens / (1 - max(mean_beta, b_lo))
        plan = make_plan(
            t_train=train_budget_seconds(cfg.budget.total_seconds, _prior_elapsed(ctx)),
            t_dry_run=timings["dry_run"], domain_tokens=tokens_with_replay,
            tokens_per_s=max(dry.tokens_per_s, 1.0), final_epochs=cfg.lora.epochs,
            trial_data_fraction=cfg.hpo.trial_data_fraction,
            trial_epochs=min(1.0, cfg.lora.epochs), n_trials_min=cfg.hpo.n_trials_min,
            n_trials_max=cfg.hpo.n_trials_max,
            min_resource_fraction=cfg.hpo.min_resource_fraction,
            reduction_factor=cfg.hpo.reduction_factor)
        ctx.emit("train_plan", n_trials=plan.n_trials, data_fraction=plan.data_fraction,
                 t_hpo=round(plan.t_hpo, 1))

        common = {"train": train, "valid": data_dir / "valid.jsonl",
                  "anchor": data_dir / "anchor.jsonl", "halluc": halluc_file,
                  "pool": replay.pool, "limit": int(avail)}
        replay_ids: dict[str, list[str]] = {}

        def make_job(p: TrialParams, number: int, run_dir: Path | None = None) -> TrainJob:
            rd = run_dir or work / "trials" / f"{number:04d}"
            job, ids = self._make_job(cfg, p, rd, common, eval_points=4)
            replay_ids[rd.name] = ids
            return job

        # 5. HPO
        t = time.time()
        trials: list[TrialResult] = []
        best_params: TrialParams
        hpo_info: dict[str, Any] = {"enabled": cfg.hpo.enabled}
        if cfg.hpo.enabled and plan.n_trials > 0:
            study = create_study(work / "hpo" / "study.db", seed=cfg.seed, hpo=cfg.hpo)
            hctx = HPOContext(
                trainer=trainer, make_job=make_job, space=cfg.search_space, hpo=cfg.hpo,
                base=cfg.lora, plan=plan, beta_lo=b_lo, anchor_base=anchor_base,
                fit_memory=fit_memory, emit=ctx.emit, check_cancelled=ctx.check_cancelled)
            outcome = HPOController(hctx, study).run()
            trials = outcome.trials
            hpo_info.update(stopped_reason=outcome.stopped_reason, n_planned=outcome.n_planned,
                            best_trial=outcome.best.number if outcome.best else None)
            best_params = outcome.best.params if outcome.best else self._default_params(cfg, b_lo)
        else:
            best_params = self._default_params(cfg, b_lo)
        timings["hpo"] = time.time() - t
        ctx.check_cancelled()

        # 6. final full-fidelity fit
        t = time.time()
        final_p = best_params.model_copy(update={
            "epochs": cfg.lora.epochs, "data_fraction": 1.0,
            "batch_size": cfg.lora.batch_size, "grad_accumulation": cfg.lora.grad_accumulation})
        fitted = fit_memory(final_p)
        if fitted is None:
            raise ZLLFError(ErrorCode.OOM_UNRECOVERABLE, "final config does not fit memory")
        final_p, ckpt, predicted = fitted
        final_job = make_job(final_p, -1, work / "final")
        final_job.grad_checkpoint = final_job.grad_checkpoint or ckpt
        final_job.save_every = max(1, final_job.iters // 10)

        def on_final(ev: dict[str, Any]) -> bool:
            ctx.emit("train_metrics", trial="final", **ev)
            return False

        final = trainer.train(final_job, on_final)
        timings["final_fit"] = time.time() - t
        if final.status != "complete" or final.adapter_dir is None:
            code = (ErrorCode.OOM_UNRECOVERABLE if final.status == "failed_oom"
                    else ErrorCode.TRAIN_FAILED)
            raise ZLLFError(code, f"final fit {final.status}: {final.error}")
        shutil.copytree(final.adapter_dir, out_dir / "adapter")

        fm = final.metrics
        final_obj = None
        parts: dict[str, float] = {}
        if fm.get("val_loss") is not None:
            final_obj, parts = objective_value(
                val_loss=fm["val_loss"], anchor_loss=fm.get("anchor_loss"),
                anchor_base=anchor_base, halluc_loss=fm.get("halluc_loss"),
                weights=cfg.hpo.objective_weights, epsilon=cfg.hpo.epsilon_forget)

        # 7. cleanup (NFR-SEC-03 basic): trial checkpoints/adapters and temp datasets
        deleted = self._cleanup(work)
        trials_out = [r.model_copy(update={"adapter_path": None}) for r in trials]
        (out_dir / "trials.json").write_text(
            json.dumps([json.loads(r.model_dump_json()) for r in trials_out], indent=2))
        fj = final.final_job or final_job
        report = {
            "backend": cfg.models.backend,
            "trainer": getattr(trainer, "name", type(trainer).__name__),
            "model": cfg.models.student,
            "seed": cfg.seed,
            "versions": _versions(),
            "final_params": fj.params.model_dump(),
            "final": {
                "objective": final_obj, **parts,
                "val_loss": fm.get("val_loss"), "anchor_loss": fm.get("anchor_loss"),
                "anchor_base_loss": anchor_base, "halluc_loss": fm.get("halluc_loss"),
                "phi_halluc": phi_halluc(fm.get("halluc_loss")),
                "train_loss": fm.get("train_loss"), "tokens_per_s": fm.get("tokens_per_s"),
                "peak_memory_gb": fm.get("peak_memory_gb"),
                "predicted_memory_gb": predicted / GB, "iters": fj.iters,
                "grad_checkpoint": fj.grad_checkpoint, "attempts": final.attempts,
                "seconds": final.seconds,
            },
            "loss_curves": {"train": fm.get("history", {}).get("train", []),
                            "val": fm.get("history", {}).get("val", [])},
            "datasets": self._datasets(ctx, replay),
            "data": {"n_train": len(train), "n_valid": len(valid), "n_halluc": len(halluc),
                     "domain_tokens_approx": domain_tokens},
            "replay": {"source": replay.source, "sha256": replay.sha256,
                       "beta": fj.params.replay_ratio, "beta_min": b_lo,
                       "n_anchor": len(replay.anchor), "n_pool": len(replay.pool),
                       "sample_ids": replay_ids.get("final", [])},
            "objective_proxies": {
                "domain_val": "mean completion loss on answerable val.jsonl",
                "anchor": "completion loss on held-out replay anchor set vs base",
                "halluc": "1-exp(-loss) of reference abstentions on unanswerable val",
            },
            "dry_run": {"ok": dry.ok, "error": dry.error, "tokens_per_s": dry.tokens_per_s,
                        "base_gb": dry.base_bytes / GB,
                        "peak_gb": (dry.peak_bytes / GB) if math.isfinite(dry.peak_bytes)
                        else None,
                        "base_losses": dry.base_losses, "rank": max_rank, "seq_len": seq},
            "memory_model": mem.to_dict(),
            "memory_limit_gb": limit_bytes / GB,
            "metal": guidance,
            "budget_plan": plan.to_dict(),
            "hpo": hpo_info,
            "timings": {**timings, "total": time.time() - t_stage},
            "cleanup": {"deleted": deleted},
        }
        (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, default=str))
        n_complete = sum(1 for r in trials if r.state == "complete")
        return StageResult(metrics={
            "n_trials": len(trials), "n_complete": n_complete,
            "final_val_loss": fm.get("val_loss"), "final_objective": final_obj,
            "seconds": time.time() - t_stage})

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _default_params(cfg: RunConfig, b_lo: float) -> TrialParams:
        lo, hi = cfg.search_space.replay_ratio
        beta = min(max(cfg.lora.replay_ratio, b_lo, lo), max(hi, lo))
        d = cfg.lora
        return TrialParams(rank=d.rank, alpha=d.alpha, learning_rate=d.learning_rate,
                           target_modules=d.target_modules, dropout=d.dropout,
                           weight_decay=d.weight_decay, replay_ratio=beta, epochs=d.epochs,
                           data_fraction=1.0, batch_size=d.batch_size,
                           grad_accumulation=d.grad_accumulation)

    @staticmethod
    def _make_job(cfg: RunConfig, p: TrialParams, run_dir: Path, common: dict[str, Any],
                  eval_points: int) -> tuple[TrainJob, list[str]]:
        """Materialise the (subset + replay-mixed) training file and size the run."""
        import random

        train: list[Sample] = common["train"]
        if p.data_fraction < 1.0:
            idx = list(range(len(train)))
            random.Random(f"{cfg.seed}:trial-subset").shuffle(idx)  # same subset per trial
            k = max(2, math.ceil(p.data_fraction * len(train)))
            train = [train[i] for i in sorted(idx[:k])]
        mixed, ids = mix_replay(train, common["pool"], p.replay_ratio, seed=cfg.seed,
                                salt="mix")
        data_file = run_dir / "data" / "train.jsonl"
        _write_jsonl(mixed, data_file)
        b = max(1, min(p.batch_size, len(mixed)))
        p = p.model_copy(update={"batch_size": b})
        acc = max(1, p.grad_accumulation)
        iters = max(acc, math.ceil(len(mixed) * p.epochs / b))
        iters = ((iters + acc - 1) // acc) * acc
        eval_every = max(1, math.ceil(iters / eval_points))
        job = TrainJob(
            params=p, run_dir=run_dir, train_file=data_file, valid_file=common["valid"],
            model=cfg.models.student, iters=iters, max_seq_length=cfg.models.max_seq_len,
            anchor_file=common["anchor"], halluc_file=common["halluc"],
            num_layers=cfg.lora.num_layers, seed=cfg.seed, eval_every=eval_every,
            save_every=eval_every, memory_limit_bytes=common["limit"])
        return job, ids

    def _dry_run(self, ctx: RunContext, trainer: Trainer, work: Path, data_dir: Path,
                 halluc_file: Path | None, limit_bytes: float) -> DryRunResult:
        cfg = ctx.config
        cache = work / "dry_run" / "dry_run.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            d["peak_bytes"] = d["peak_bytes"] if d["peak_bytes"] is not None else math.inf
            return DryRunResult(**d)
        spec = DryRunSpec(
            model=cfg.models.student, run_dir=work / "dry_run",
            rank=max(cfg.search_space.rank), target_modules="all",
            batch_size=cfg.lora.batch_size, max_seq_length=cfg.models.max_seq_len,
            num_layers=cfg.lora.num_layers, seed=cfg.seed, anchor_file=data_dir / "anchor.jsonl",
            halluc_file=halluc_file, valid_file=data_dir / "valid.jsonl",
            memory_limit_bytes=int(limit_bytes / max(cfg.hpo.memory_fraction, 1e-6)))
        dry = trainer.dry_run(spec)
        if not dry.ok and spec.batch_size > 1:  # retry the dry run at micro-batch 1
            spec.batch_size = 1
            dry = trainer.dry_run(spec)
        ctx.emit("dry_run", ok=dry.ok, tokens_per_s=round(dry.tokens_per_s, 1),
                 peak_gb=round(dry.peak_bytes / GB, 3) if math.isfinite(dry.peak_bytes) else None)
        if dry.ok:
            cache.parent.mkdir(parents=True, exist_ok=True)
            d = dict(dry.__dict__)
            cache.write_text(json.dumps(d))
        return dry

    @staticmethod
    def _datasets(ctx: RunContext, replay: Any) -> list[dict[str, str]]:
        """Data lineage for the package stage / SBOM: synth splits + the replay set used."""
        from zllf.core.hashing import sha256_file

        out = []
        synth = ctx.paths.stage("synth")
        for name, role in (("train.jsonl", "train"), ("val.jsonl", "val")):
            p = synth / name
            if p.exists():
                out.append({"path": str(p), "sha256": sha256_file(p), "role": role})
        out.append({"path": replay.path, "sha256": replay.sha256, "role": "replay"})
        return out

    @staticmethod
    def _cleanup(work: Path) -> list[str]:
        deleted = []
        for name in ("trials", "final", "data", "dry_run"):
            p = work / name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                deleted.append(name)
        return deleted


__all__ = ["Plan", "TrainStage"]
