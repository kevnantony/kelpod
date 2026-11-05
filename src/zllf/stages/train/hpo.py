"""HPO controller (FR-07, P2.4): Optuna TPE + successive halving under a time budget.

Objective (PRD eq. 3), lower is better::

    L = w1 · L_domain_val + w2 · max(0, L_anchor − L_anchor_base − ε) + w3 · Φ_halluc

Proxies (documented, cheap, no generation needed):

* ``L_domain_val`` – mean completion-token loss on the *answerable* val.jsonl examples.
* ``L_anchor``     – the same loss on a held-out slice of the general replay set (never
  trained on); ``L_anchor_base`` is the base model's loss on it, measured once in the dry run.
  Note ε is in nats here (the gate's ε is in accuracy points; same role, different unit).
* ``Φ_halluc``     – ``1 − exp(−L_abst)`` where ``L_abst`` is the loss of the reference
  *abstention* answers of unanswerable val examples: the complement of the geometric-mean
  token probability of abstaining, i.e. a soft "fabrication rate" in [0, 1). 0 if val.jsonl
  has no unanswerables.

Intermediate values reported for pruning are ``w1 · L_domain_val`` at each eval step; the
pruner step is training progress in percent so OOM fallbacks (which change the number of
micro-steps) do not shift the rungs.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zllf.core.config import HPOConfig, LoraDefaults, SearchSpace
from zllf.core.schemas import TrialParams, TrialResult
from zllf.stages.train.planner import Plan, replan, should_start_trial
from zllf.stages.train.trainer import Trainer, TrainJob, TrainOutcome

RESULT_ATTR = "zllf_result"


# ---------------------------------------------------------------- objective
def phi_halluc(halluc_loss: float | None) -> float:
    """Soft fabrication rate from the abstention loss (0 when there is no proxy set)."""
    if halluc_loss is None or not math.isfinite(halluc_loss):
        return 0.0
    return 1.0 - math.exp(-max(halluc_loss, 0.0))


def objective_value(
    *,
    val_loss: float,
    anchor_loss: float | None,
    anchor_base: float | None,
    halluc_loss: float | None,
    weights: dict[str, float],
    epsilon: float,
) -> tuple[float, dict[str, float]]:
    """PRD eq. (3). Returns (value, parts)."""
    forget = 0.0
    if anchor_loss is not None and anchor_base is not None:
        forget = max(0.0, anchor_loss - anchor_base - epsilon)
    phi = phi_halluc(halluc_loss)
    w1 = weights.get("domain_val", 1.0)
    w2 = weights.get("forgetting", 0.0)
    w3 = weights.get("hallucination", 0.0)
    value = w1 * val_loss + w2 * forget + w3 * phi
    return value, {"domain_term": w1 * val_loss, "forgetting_term": w2 * forget,
                   "halluc_term": w3 * phi, "forgetting": forget, "phi_halluc": phi}


# ---------------------------------------------------------------- search space
def replay_bounds(space: SearchSpace, beta_lo: float) -> tuple[float, float]:
    """β range with the corpus-size lower bound applied (never above the upper bound)."""
    lo, hi = space.replay_ratio
    lo_eff = min(hi, max(lo, beta_lo))
    return lo_eff, hi


def sample_params(
    trial: Any, space: SearchSpace, *, beta_lo: float, base: LoraDefaults,
    epochs: float, data_fraction: float,
) -> TrialParams:
    """Sample θ exactly from ``config.search_space``; α is conditional via α/r."""
    rank = int(trial.suggest_categorical("rank", list(space.rank)))
    ratio = float(trial.suggest_categorical("alpha_over_rank", list(space.alpha_over_rank)))
    lr = trial.suggest_float("learning_rate", *space.learning_rate, log=True)
    tm = trial.suggest_categorical("target_modules", list(space.target_modules))
    dropout = trial.suggest_float("dropout", *space.dropout)
    wd = trial.suggest_float("weight_decay", *space.weight_decay)
    b_lo, b_hi = replay_bounds(space, beta_lo)
    beta = trial.suggest_float("replay_ratio", b_lo, b_hi)
    return TrialParams(rank=rank, alpha=rank * ratio, learning_rate=lr, target_modules=tm,
                       dropout=dropout, weight_decay=wd, replay_ratio=beta, epochs=epochs,
                       data_fraction=data_fraction, batch_size=base.batch_size,
                       grad_accumulation=base.grad_accumulation)


def _snap(value: Any, choices: list[Any]) -> Any:
    if value in choices:
        return value
    return min(choices, key=lambda c: abs(math.log(c) - math.log(value)) if c > 0 else 1e9)


def _clip(v: float, lo_hi: tuple[float, float]) -> float:
    return float(min(max(v, lo_hi[0]), lo_hi[1]))


def warm_start_configs(space: SearchSpace, base: LoraDefaults, beta_lo: float
                       ) -> list[dict[str, Any]]:
    """Two known-good configs, snapped into the configured space (P2.4.5): the PRD/QLoRA
    default (r16, α=2r, 2e-4, all-linear) and a conservative small-rank variant."""
    b = replay_bounds(space, beta_lo)
    raw: list[dict[str, Any]] = [
        {"rank": base.rank, "alpha_over_rank": base.alpha / base.rank,
         "learning_rate": base.learning_rate, "target_modules": base.target_modules,
         "dropout": base.dropout, "weight_decay": base.weight_decay,
         "replay_ratio": base.replay_ratio},
        {"rank": 8, "alpha_over_rank": 2.0, "learning_rate": 1e-4,
         "target_modules": "all", "dropout": 0.1, "weight_decay": 0.01,
         "replay_ratio": (b[0] + b[1]) / 2},
    ]
    out = []
    for r in raw:
        tm = r["target_modules"] if r["target_modules"] in space.target_modules \
            else space.target_modules[0]
        out.append({
            "rank": _snap(r["rank"], list(space.rank)),
            "alpha_over_rank": _snap(r["alpha_over_rank"], list(space.alpha_over_rank)),
            "learning_rate": _clip(r["learning_rate"], space.learning_rate),
            "target_modules": tm,
            "dropout": _clip(r["dropout"], space.dropout),
            "weight_decay": _clip(r["weight_decay"], space.weight_decay),
            "replay_ratio": _clip(r["replay_ratio"], b),
        })
    return out


def create_study(storage_path: Path, *, seed: int, hpo: HPOConfig, name: str = "zllf") -> Any:
    """SQLite-backed study (resumable), TPE multivariate + successive halving."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                         n_startup_trials=max(2, hpo.n_trials_min // 2))
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=max(1, round(hpo.min_resource_fraction * 100)),
        reduction_factor=hpo.reduction_factor,
    )
    return optuna.create_study(study_name=name, storage=f"sqlite:///{storage_path}",
                               sampler=sampler, pruner=pruner, direction="minimize",
                               load_if_exists=True)


# ---------------------------------------------------------------- controller
@dataclass
class HPOContext:
    """Everything the controller needs from the stage (keeps hpo.py testable)."""

    trainer: Trainer
    make_job: Callable[[TrialParams, int], TrainJob]  # (params, trial number) -> job
    space: SearchSpace
    hpo: HPOConfig
    base: LoraDefaults
    plan: Plan
    beta_lo: float
    anchor_base: float | None
    fit_memory: Callable[[TrialParams], tuple[TrialParams, bool, float] | None]
    emit: Callable[..., None] = lambda *a, **k: None
    check_cancelled: Callable[[], None] = lambda: None
    clock: Callable[[], float] = time.time


@dataclass
class HPOOutcome:
    trials: list[TrialResult]
    best: TrialResult | None
    seconds: float
    n_planned: list[int] = field(default_factory=list)
    stopped_reason: str = ""


def _float_metrics(m: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in m.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


class HPOController:
    def __init__(self, hctx: HPOContext, study: Any) -> None:
        self.h = hctx
        self.study = study

    # ------------------------------------------------------------ helpers
    def _record(self, number: int, params: TrialParams, outcome: TrainOutcome | None,
                state: str, objective: float | None, metrics: dict[str, Any],
                trial_id_obj: Any) -> TrialResult:
        res = TrialResult(
            number=number, params=params, state=state,  # type: ignore[arg-type]
            objective=objective, metrics=_float_metrics(metrics),
            adapter_path=str(outcome.adapter_dir) if outcome and outcome.adapter_dir else None,
            seconds=outcome.seconds if outcome else 0.0,
        )
        trial_id_obj.set_user_attr(RESULT_ATTR, json.loads(res.model_dump_json()))
        return res

    def existing_results(self) -> list[TrialResult]:
        out = []
        for t in self.study.get_trials(deepcopy=False):
            if RESULT_ATTR in t.user_attrs and t.state.is_finished():
                out.append(TrialResult.model_validate(t.user_attrs[RESULT_ATTR]))
        return out

    def _evaluate(self, outcome: TrainOutcome) -> tuple[float, dict[str, float]]:
        m = outcome.metrics
        return objective_value(
            val_loss=float(m["val_loss"]), anchor_loss=m.get("anchor_loss"),
            anchor_base=self.h.anchor_base, halluc_loss=m.get("halluc_loss"),
            weights=self.h.hpo.objective_weights, epsilon=self.h.hpo.epsilon_forget)

    # ------------------------------------------------------------ one trial
    def run_trial(self, trial: Any, params: TrialParams) -> TrialResult:
        from optuna.trial import TrialState

        number = trial.number
        fitted = self.h.fit_memory(params)
        if fitted is None:
            res = self._record(number, params, None, "pruned", None,
                               {"memory_infeasible": 1.0}, trial)
            self.study.tell(trial, state=TrialState.PRUNED)
            return res
        params, ckpt, predicted = fitted
        job = self.h.make_job(params, number)
        job.grad_checkpoint = job.grad_checkpoint or ckpt
        w1 = self.h.hpo.objective_weights.get("domain_val", 1.0)

        def on_metric(ev: dict[str, Any]) -> bool:
            self.h.emit("train_metrics", trial=number, **ev)
            if ev.get("event") != "eval" or ev.get("val_loss") is None:
                return False
            step = round(100 * float(ev.get("progress", 0.0)))
            if step <= 0:
                return False
            trial.report(w1 * float(ev["val_loss"]), step)
            return bool(trial.should_prune())

        outcome = self.h.trainer.train(job, on_metric)
        metrics: dict[str, Any] = dict(outcome.metrics)
        metrics["predicted_memory_gb"] = predicted / 1024**3
        fj = outcome.final_job or job
        metrics["batch_size"] = fj.params.batch_size
        metrics["grad_accumulation"] = fj.params.grad_accumulation
        metrics["attempts"] = len(outcome.attempts) or 1
        if outcome.status == "complete" and outcome.metrics.get("val_loss") is not None:
            value, parts = self._evaluate(outcome)
            metrics.update(parts)
            res = self._record(number, fj.params, outcome, "complete", value, metrics, trial)
            self.study.tell(trial, value)
        elif outcome.status == "pruned":
            res = self._record(number, fj.params, outcome, "pruned", None, metrics, trial)
            self.study.tell(trial, state=TrialState.PRUNED)
        else:
            state = "failed_oom" if outcome.status == "failed_oom" else "failed"
            res = self._record(number, fj.params, outcome, state, None, metrics, trial)
            self.study.tell(trial, state=TrialState.FAIL)
        return res

    # ------------------------------------------------------------ loop
    def _resume_running(self, results: list[TrialResult]) -> None:
        """Trials left RUNNING by a crashed controller are re-run with the same params; the
        worker resumes from their checkpoints."""
        from optuna.trial import Trial, TrialState

        for ft in self.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            trial = Trial(self.study, ft._trial_id)
            p = ft.params
            params = TrialParams(
                rank=p["rank"], alpha=p["rank"] * p["alpha_over_rank"],
                learning_rate=p["learning_rate"], target_modules=p["target_modules"],
                dropout=p["dropout"], weight_decay=p["weight_decay"],
                replay_ratio=p["replay_ratio"], epochs=self.h.plan.trial_epochs,
                data_fraction=self.h.plan.data_fraction, batch_size=self.h.base.batch_size,
                grad_accumulation=self.h.base.grad_accumulation)
            results.append(self.run_trial(trial, params))

    def run(self) -> HPOOutcome:
        t0 = self.h.clock()
        h = self.h
        results = self.existing_results()
        if not self.study.trials:
            for cfg in warm_start_configs(h.space, h.base, h.beta_lo):
                self.study.enqueue_trial(cfg)
        self._resume_running(results)
        prior_seconds = sum(r.seconds for r in results)
        costs = [r.seconds for r in results if r.seconds > 0]
        n_planned: list[int] = []
        reason = "n_trials reached"
        infeasible = 0
        while True:
            h.check_cancelled()
            ran = [r for r in results if not r.metrics.get("memory_infeasible")]
            elapsed = prior_seconds + (h.clock() - t0)
            target = replan(h.plan, elapsed_hpo=elapsed, measured_costs=costs,
                            done=len(ran), n_trials_max=h.hpo.n_trials_max)
            n_planned.append(target)
            if len(ran) >= target:
                break
            next_cost = (sum(costs) / len(costs)) if costs else h.plan.expected_trial_cost
            if not should_start_trial(h.plan, elapsed_hpo=elapsed, next_cost=next_cost) \
                    and len(ran) >= 1:
                reason = "time budget"
                break
            trial = self.study.ask()
            params = sample_params(trial, h.space, beta_lo=h.beta_lo, base=h.base,
                                   epochs=h.plan.trial_epochs,
                                   data_fraction=h.plan.data_fraction)
            res = self.run_trial(trial, params)
            results.append(res)
            if res.metrics.get("memory_infeasible"):
                infeasible += 1
                if infeasible > 3 * h.hpo.n_trials_max:
                    reason = "search space infeasible for memory"
                    break
                continue
            if res.seconds > 0:
                costs.append(res.seconds)
            h.emit("trial", number=res.number, state=res.state, objective=res.objective,
                   rank=res.params.rank, lr=res.params.learning_rate,
                   target_modules=res.params.target_modules, seconds=round(res.seconds, 2))
        done = [r for r in results if r.state == "complete" and r.objective is not None]
        best = min(done, key=lambda r: (r.objective, r.number)) if done else None
        results.sort(key=lambda r: r.number)
        return HPOOutcome(trials=results, best=best, seconds=h.clock() - t0,
                          n_planned=n_planned, stopped_reason=reason)
