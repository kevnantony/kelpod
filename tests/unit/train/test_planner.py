from __future__ import annotations

import math

import pytest

from zllf.stages.train.planner import (
    STAGE_FRACTIONS,
    asha_factor,
    make_plan,
    replan,
    run_cost,
    should_start_trial,
    train_budget_seconds,
)


def test_stage_fractions_sum_to_one() -> None:
    assert math.isclose(sum(STAGE_FRACTIONS.values()), 1.0)


def test_train_budget_reference_split() -> None:
    # 8 h budget, prior stages assumed at reference → train gets 4.5 h
    assert math.isclose(train_budget_seconds(8 * 3600), 4.5 * 3600)
    # measured prior stages are used instead of the reference fractions
    t = train_budget_seconds(8 * 3600, prior_elapsed=600)
    assert math.isclose(t, 8 * 3600 - 600 - 1.5 * 3600)
    assert train_budget_seconds(100, prior_elapsed=10_000) == 0.0


def test_asha_factor() -> None:
    # rungs at 0.25 and 0.75 with η=3: 0.25 + (1/3)(0.5) + (1/9)(0.25)
    assert math.isclose(asha_factor(0.25, 3), 0.25 + 0.5 / 3 + 0.25 / 9)
    assert asha_factor(1.0, 3) == 1.0
    assert 0 < asha_factor(0.1, 2) < 1


def test_run_cost() -> None:
    assert run_cost(10_000, 1000, 5) == 15


def _plan(**kw: float) -> object:
    args = dict(t_train=3600.0, t_dry_run=60.0, domain_tokens=1e6, tokens_per_s=2000.0,
                final_epochs=2.0, trial_data_fraction=0.3, trial_epochs=1.0, n_trials_min=4,
                n_trials_max=32, min_resource_fraction=0.25, reduction_factor=3,
                eval_overhead_s=10.0)
    args.update(kw)
    return make_plan(**args)  # type: ignore[arg-type]


def test_plan_arithmetic() -> None:
    p = _plan(t_train=20_000.0)
    t_final = 1e6 * 2 / 2000 + 10  # 1010 s
    assert math.isclose(p.t_final_fit, t_final)
    assert math.isclose(p.t_hpo, 20_000 - 60 - t_final)
    cost = 1e6 * 0.3 / 2000 + 10  # 160 s
    assert math.isclose(p.trial_cost, cost)
    e = cost * asha_factor(0.25, 3)
    assert p.n_trials == min(32, math.floor(p.t_hpo / e))
    assert p.data_fraction == 0.3 and not p.warnings


def test_plan_clamps_to_max() -> None:
    assert _plan(t_train=1e9).n_trials == 32


def test_plan_reduces_fidelity_before_trials() -> None:
    p = _plan(t_train=1300.0)  # t_hpo = 230 s — 0.3 fraction allows < 4 trials
    assert p.data_fraction < 0.3
    assert p.n_trials >= 4
    assert any("fraction reduced" in w for w in p.warnings)
    # the reduced plan really fits: n_min expected trials within t_hpo
    assert 4 * p.expected_trial_cost <= p.t_hpo + 1e-6


def test_plan_warns_when_nothing_fits() -> None:
    p = _plan(t_train=1070.0)  # barely more than final fit
    assert p.data_fraction == pytest.approx(0.05)
    assert p.n_trials < 4
    assert any("only" in w for w in p.warnings)


def test_replan_and_hard_stop() -> None:
    p = _plan(t_train=20_000.0)
    assert replan(p, elapsed_hpo=0, measured_costs=[], done=0) == p.n_trials
    # trials were 4x as expensive as planned → fewer in total
    n = replan(p, elapsed_hpo=2000, measured_costs=[4 * p.trial_cost] * 3, done=3)
    assert 3 <= n < p.n_trials
    # out of time → no more than done
    assert replan(p, elapsed_hpo=p.t_hpo, measured_costs=[100.0], done=5) == 5
    assert should_start_trial(p, elapsed_hpo=0, next_cost=p.t_hpo)
    assert not should_start_trial(p, elapsed_hpo=1, next_cost=p.t_hpo)
