from __future__ import annotations

import json
from pathlib import Path

import pytest

from zllf.stages.train.replay import (
    approx_tokens,
    beta_min,
    load_replay,
    mix_replay,
    replay_token_fraction,
)


def _domain(n: int) -> list[dict]:
    return [{"id": f"d{i}", "messages": [{"role": "user", "content": "q" * 40},
                                         {"role": "assistant", "content": "a" * 80}]}
            for i in range(n)]


def test_fallback_set_and_anchor_split() -> None:
    rs = load_replay(Path("/nonexistent/replay.jsonl"), seed=1)
    assert rs.source == "builtin"
    assert len(rs.pool) + len(rs.anchor) >= 30
    assert len(rs.anchor) == (len(rs.pool) + len(rs.anchor)) // 4
    assert not {s["id"] for s in rs.anchor} & {s["id"] for s in rs.pool}
    assert load_replay(Path("/nonexistent"), seed=1).anchor == rs.anchor


def test_bench_file_is_used(tmp_path: Path) -> None:
    p = tmp_path / "replay.jsonl"
    with p.open("w") as f:
        for i in range(40):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                             {"role": "assistant", "content": f"a{i}"}],
                                "source": "test"}) + "\n")
    rs = load_replay(p, seed=0, n_anchor=5)
    assert rs.source == "bench" and len(rs.anchor) == 5 and len(rs.pool) == 35
    import hashlib

    assert rs.path == str(p) and rs.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(s["id"].startswith("replay-") for s in rs.pool)


@pytest.mark.parametrize("beta", [0.05, 0.15, 0.3])
def test_mixing_ratio_by_tokens(beta: float) -> None:
    rs = load_replay(Path("/nonexistent"), seed=0)
    domain = _domain(60)
    mixed, ids = mix_replay(domain, rs.pool, beta, seed=3)
    frac = replay_token_fraction(mixed, set(ids))
    one = max(approx_tokens(s["messages"]) for s in rs.pool)
    total = sum(approx_tokens(s["messages"]) for s in mixed)
    assert abs(frac - beta) <= one / total + 1e-9  # within one sample
    assert len(mixed) == 60 + len(ids)


def test_mixing_is_deterministic_and_seeded() -> None:
    rs = load_replay(Path("/nonexistent"), seed=0)
    a = mix_replay(_domain(30), rs.pool, 0.2, seed=7)
    b = mix_replay(_domain(30), rs.pool, 0.2, seed=7)
    c = mix_replay(_domain(30), rs.pool, 0.2, seed=8)
    assert a == b
    assert a[0] != c[0]


def test_zero_beta_and_bad_beta() -> None:
    mixed, ids = mix_replay(_domain(5), [], 0.0, seed=0)
    assert ids == [] and len(mixed) == 5
    with pytest.raises(ValueError):
        mix_replay(_domain(5), [], 1.0, seed=0)


def test_beta_min_rises_as_corpus_shrinks() -> None:
    assert beta_min(50) == 0.30
    assert beta_min(500) == pytest.approx(0.30)
    assert beta_min(5000) == pytest.approx(0.25)
    assert beta_min(10**9) == 0.05
    assert beta_min(100) >= beta_min(1000) >= beta_min(100000)
