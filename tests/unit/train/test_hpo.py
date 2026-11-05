from __future__ import annotations

import math
from pathlib import Path

import optuna
import pytest

from zllf.core.config import LoraDefaults, SearchSpace, load_config
from zllf.core.schemas import TrialParams
from zllf.stages.train.hpo import (
    HPOContext,
    HPOController,
    create_study,
    objective_value,
    phi_halluc,
    replay_bounds,
    sample_params,
    warm_start_configs,
)
from zllf.stages.train.planner import make_plan
from zllf.stages.train.trainer import FakeTrainer, TrainJob


def test_objective_eq3() -> None:
    w = {"domain_val": 1.0, "forgetting": 2.0, "hallucination": 1.5}
    v, parts = objective_value(val_loss=1.2, anchor_loss=1.6, anchor_base=1.5,
                               halluc_loss=0.5, weights=w, epsilon=0.03)
    expected = 1.2 + 2.0 * (1.6 - 1.5 - 0.03) + 1.5 * (1 - math.exp(-0.5))
    assert v == pytest.approx(expected)
    assert parts["forgetting"] == pytest.approx(0.07)
    # no forgetting penalty inside ε; missing proxies → 0
    v2, p2 = objective_value(val_loss=1.0, anchor_loss=1.52, anchor_base=1.5,
                             halluc_loss=None, weights=w, epsilon=0.03)
    assert v2 == pytest.approx(1.0) and p2["forgetting"] == 0 and p2["phi_halluc"] == 0
    assert phi_halluc(0.0) == 0.0 and 0 < phi_halluc(3.0) < 1


def test_sampling_respects_space_and_constraints() -> None:
    space = SearchSpace()
    base = LoraDefaults()
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
    for _ in range(25):
        t = study.ask()
        p = sample_params(t, space, beta_lo=0.2, base=base, epochs=1.0, data_fraction=0.3)
        assert p.rank in space.rank
        assert p.alpha / p.rank in space.alpha_over_rank
        assert space.learning_rate[0] <= p.learning_rate <= space.learning_rate[1]
        assert p.target_modules in space.target_modules
        assert space.dropout[0] <= p.dropout <= space.dropout[1]
        assert space.weight_decay[0] <= p.weight_decay <= space.weight_decay[1]
        assert 0.2 <= p.replay_ratio <= space.replay_ratio[1]  # corpus-size lower bound
        assert p.data_fraction == 0.3
        study.tell(t, p.learning_rate * 1000 + p.rank / 100)


def test_replay_bounds_never_invert() -> None:
    s = SearchSpace()
    assert replay_bounds(s, 0.0) == (0.05, 0.30)
    assert replay_bounds(s, 0.5) == (0.30, 0.30)


def test_warm_start_snapped_into_space() -> None:
    space = SearchSpace(rank=[4, 32], alpha_over_rank=[1.0], learning_rate=(1e-5, 1e-4),
                        target_modules=["attention"])
    cfgs = warm_start_configs(space, LoraDefaults(), beta_lo=0.3)
    assert len(cfgs) == 2
    for c in cfgs:
        assert c["rank"] in [4, 32] and c["alpha_over_rank"] == 1.0
        assert c["learning_rate"] <= 1e-4 and c["target_modules"] == "attention"
        assert c["replay_ratio"] == 0.30


def _controller(tmp_path: Path, fit=None, n_max: int = 3):  # type: ignore[no-untyped-def]
    cfg = load_config("mode_a", "fake", {"hpo": {"n_trials_max": n_max, "n_trials_min": 2}})
    plan = make_plan(t_train=1e6, t_dry_run=0, domain_tokens=1e4, tokens_per_s=5000,
                     final_epochs=1, trial_data_fraction=0.3, trial_epochs=1, n_trials_min=2,
                     n_trials_max=n_max, min_resource_fraction=0.25, reduction_factor=3)

    def make_job(p: TrialParams, n: int) -> TrainJob:
        return TrainJob(params=p, run_dir=tmp_path / f"t{n}", train_file=tmp_path / "x",
                        valid_file=tmp_path / "v", model="fake", iters=40, max_seq_length=64,
                        eval_every=10)

    study = create_study(tmp_path / "study.db", seed=cfg.seed, hpo=cfg.hpo)
    hctx = HPOContext(trainer=FakeTrainer(), make_job=make_job, space=cfg.search_space,
                      hpo=cfg.hpo, base=cfg.lora, plan=plan, beta_lo=0.1, anchor_base=1.5,
                      fit_memory=fit or (lambda p: (p, False, 1e9)))
    return HPOController(hctx, study), study


def test_hpo_runs_warm_start_first_and_records(tmp_path: Path) -> None:
    ctl, _ = _controller(tmp_path)
    out = ctl.run()
    assert len(out.trials) == 3
    first = out.trials[0].params
    assert first.rank == 16 and first.alpha == 32  # enqueued known-good config
    assert out.best is not None and out.best.state == "complete"
    assert all(r.objective is not None for r in out.trials if r.state == "complete")
    # resume: a new controller on the same SQLite study sees the trials and runs no more
    ctl2, _ = _controller(tmp_path)
    out2 = ctl2.run()
    assert [t.number for t in out2.trials] == [t.number for t in out.trials]


def test_memory_infeasible_configs_are_pruned(tmp_path: Path) -> None:
    def fit(p: TrialParams):  # type: ignore[no-untyped-def]
        return None if p.rank >= 64 else (p, False, 1e9)

    ctl, _ = _controller(tmp_path, fit=fit, n_max=4)
    out = ctl.run()
    inf = [r for r in out.trials if r.metrics.get("memory_infeasible")]
    ran = [r for r in out.trials if not r.metrics.get("memory_infeasible")]
    assert len(ran) == 4
    assert all(r.params.rank >= 64 and r.state == "pruned" for r in inf)
    assert all(r.params.rank < 64 for r in ran)
