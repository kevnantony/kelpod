"""Time-budget planner (FR-19, DEVELOPMENT_PLAN Appendix C).

    T_train   = T_budget − T_prior_stages − T_eval − T_package − slack
    T_hpo     = T_train − T_dry_run − T_final_fit
    cost      ≈ trial_tokens / tokens_per_s + eval_overhead
    E[cost]   = cost · asha_factor   (successive halving stops most trials early)
    n_trials  = clamp(floor(T_hpo / E[cost]), n_min, n_max)
    if floor(...) < n_min: reduce the trial data fraction until n_min trials fit
                           (floor min_data_fraction); then warn and run what fits.

Re-plan after every trial with measured costs; hard stop when the next trial would cross
the HPO deadline.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Reference split of an 8 h budget (PRD §8.2), as fractions of the total.
STAGE_FRACTIONS: dict[str, float] = {
    "ingest_sanitize": 0.5 / 8,
    "synth": 1.5 / 8,
    "train": 4.5 / 8,
    "evaluate": 1.0 / 8,
    "package": 0.25 / 8,
    "slack": 0.25 / 8,
}


def train_budget_seconds(total_seconds: float, prior_elapsed: float | None = None) -> float:
    """Seconds available for the train stage.

    ``prior_elapsed`` is the measured wall time of ingest/sanitize/synth (from the manifest);
    if unknown, their reference fractions are assumed. Evaluate, package and slack are
    always reserved by fraction.
    """
    reserved_after = total_seconds * (
        STAGE_FRACTIONS["evaluate"] + STAGE_FRACTIONS["package"] + STAGE_FRACTIONS["slack"]
    )
    if prior_elapsed is None:
        prior_elapsed = total_seconds * (STAGE_FRACTIONS["ingest_sanitize"]
                                         + STAGE_FRACTIONS["synth"])
    return max(0.0, total_seconds - prior_elapsed - reserved_after)


@dataclass
class Plan:
    n_trials: int
    data_fraction: float
    trial_epochs: float
    t_train: float
    t_hpo: float
    t_final_fit: float
    trial_cost: float  # full (unpruned) trial seconds
    expected_trial_cost: float  # under ASHA
    tokens_per_s: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cost(tokens: float, tokens_per_s: float, eval_overhead_s: float) -> float:
    return tokens / max(tokens_per_s, 1e-9) + eval_overhead_s


def asha_factor(min_resource_fraction: float, reduction_factor: int) -> float:
    """Expected fraction of a full trial's cost under successive halving, assuming trials are
    stopped at rungs r0, r0·η, ... with only 1/η surviving each rung."""
    eta = max(reduction_factor, 2)
    r = min_resource_fraction
    survivors = 1.0
    cost = 0.0
    prev = 0.0
    while r < 1.0:
        cost += survivors * (r - prev)
        prev, survivors, r = r, survivors / eta, r * eta
    cost += survivors * (1.0 - prev)
    return min(1.0, cost)


def make_plan(
    *,
    t_train: float,
    t_dry_run: float,
    domain_tokens: float,
    tokens_per_s: float,
    final_epochs: float,
    trial_data_fraction: float,
    trial_epochs: float,
    n_trials_min: int,
    n_trials_max: int,
    min_resource_fraction: float,
    reduction_factor: int,
    eval_overhead_s: float = 10.0,
    min_data_fraction: float = 0.05,
) -> Plan:
    """Choose the number of HPO trials and their data fraction for the time budget."""
    warnings: list[str] = []
    t_final = run_cost(domain_tokens * final_epochs, tokens_per_s, eval_overhead_s)
    t_hpo = max(0.0, t_train - t_dry_run - t_final)
    factor = asha_factor(min_resource_fraction, reduction_factor)

    def costs(frac: float) -> tuple[float, float]:
        c = run_cost(domain_tokens * frac * trial_epochs, tokens_per_s, eval_overhead_s)
        return c, c * factor

    frac = trial_data_fraction
    cost, e_cost = costs(frac)
    n = math.floor(t_hpo / e_cost + 1e-9) if e_cost > 0 else n_trials_max
    if n < n_trials_min:
        # reduce fidelity before reducing trials (PRD §8.2)
        need = t_hpo / max(n_trials_min, 1) / factor - eval_overhead_s  # seconds per trial
        new_frac = need * tokens_per_s / max(domain_tokens * trial_epochs, 1e-9)
        new_frac = min(frac, max(min_data_fraction, new_frac))
        if new_frac < frac:
            warnings.append(f"trial data fraction reduced {frac:.2f} -> {new_frac:.2f}")
            frac = new_frac
            cost, e_cost = costs(frac)
            n = math.floor(t_hpo / e_cost + 1e-9) if e_cost > 0 else n_trials_max
        if n < n_trials_min:
            warnings.append(
                f"budget allows only {max(n, 0)} trials (< n_trials_min={n_trials_min}); "
                "consider a larger budget or fewer documents"
            )
    n = max(0, min(n, n_trials_max)) if n < n_trials_min else min(n, n_trials_max)
    if t_final > t_train:
        warnings.append("final fit alone is predicted to exceed the train budget")
    return Plan(
        n_trials=int(n), data_fraction=frac, trial_epochs=trial_epochs, t_train=t_train,
        t_hpo=t_hpo, t_final_fit=t_final, trial_cost=cost, expected_trial_cost=e_cost,
        tokens_per_s=tokens_per_s, warnings=warnings,
    )


def replan(
    plan: Plan, *, elapsed_hpo: float, measured_costs: list[float], done: int,
    n_trials_max: int | None = None,
) -> int:
    """Total trials to run given what has been spent so far, using the mean measured trial
    cost. Result is in [done, n_trials_max] (cap defaults to the original plan)."""
    cap = plan.n_trials if n_trials_max is None else n_trials_max
    if not measured_costs:
        return min(plan.n_trials, cap)
    mean_cost = sum(measured_costs) / len(measured_costs)
    remaining = max(0.0, plan.t_hpo - elapsed_hpo)
    more = math.floor(remaining / mean_cost) if mean_cost > 0 else plan.n_trials
    return max(done, min(cap, done + more))


def should_start_trial(plan: Plan, *, elapsed_hpo: float, next_cost: float) -> bool:
    """Hard stop: never start a trial predicted to end after the HPO deadline."""
    return elapsed_hpo + next_cost <= plan.t_hpo
