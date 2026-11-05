"""Replay mixing (FR-05, P2.3): blend general-instruction samples into the training set.

Replay samples come from ``bench/replay/replay.jsonl`` when present (lines
``{"messages": [...], "source": "..."}``), else from a small built-in fallback set. A fixed,
seed-determined slice of the replay pool is held out as the *anchor* set used for the
forgetting term L_anchor of the HPO objective; the rest is the mixing pool.

Token counts use a tokenizer-free approximation (``approx_tokens``, ~4 chars/token) so the
controller never loads a model; the ratio is what matters, not absolute counts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zllf.core.config import REPO_ROOT
from zllf.stages.train._replay_fallback import FALLBACK_PAIRS

DEFAULT_REPLAY_PATH = REPO_ROOT / "bench" / "replay" / "replay.jsonl"

Sample = dict[str, Any]  # {"id": str, "messages": [{"role", "content"}...]}


def approx_tokens(messages: list[dict[str, str]]) -> int:
    """Cheap token estimate: ceil(chars / 4) + 4 per message (role/template overhead)."""
    return sum(math.ceil(len(m.get("content", "")) / 4) + 4 for m in messages)


def beta_min(n_domain_samples: int) -> float:
    """P2.3.3: β_min = clip(0.30 − 0.05·log10(n/500), 0.05, 0.30). Small corpora get more
    replay (the bound rises as the corpus shrinks)."""
    n = max(n_domain_samples, 1)
    return float(min(0.30, max(0.05, 0.30 - 0.05 * math.log10(n / 500))))


@dataclass
class ReplaySet:
    source: str  # "bench" | "builtin"
    sha256: str  # sha256 of the replay file bytes (builtin: of its canonical JSON)
    path: str = ""  # file actually used, or "builtin:<module>"
    pool: list[Sample] = field(default_factory=list)
    anchor: list[Sample] = field(default_factory=list)


def _fallback_samples() -> list[Sample]:
    return [
        {"id": f"builtin-{i:03d}",
         "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
        for i, (q, a) in enumerate(FALLBACK_PAIRS)
    ]


def load_replay(
    path: Path | None = None, *, seed: int = 0, n_anchor: int = 32
) -> ReplaySet:
    """Load replay samples and split off a deterministic anchor hold-out.

    ``n_anchor`` is capped at a quarter of the samples so the pool stays usable.
    """
    path = DEFAULT_REPLAY_PATH if path is None else path
    samples: list[Sample] = []
    source = "builtin"
    digest = hashlib.sha256()
    if path.is_file():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = [{"role": m["role"], "content": m["content"]} for m in rec.get("messages", [])]
            if not msgs or msgs[-1]["role"] != "assistant":
                continue
            samples.append({"id": str(rec.get("id", f"replay-{i:06d}")), "messages": msgs})
        if samples:
            source = "bench"
            digest = hashlib.sha256(path.read_bytes())
    used = str(path)
    if not samples:
        samples = _fallback_samples()
        digest = hashlib.sha256(json.dumps(samples, sort_keys=True).encode())
        used = "builtin:zllf.stages.train._replay_fallback"
    order = list(range(len(samples)))
    random.Random(f"{seed}:replay-split").shuffle(order)
    k = min(n_anchor, len(samples) // 4)
    anchor = [samples[i] for i in order[:k]]
    pool = [samples[i] for i in order[k:]]
    return ReplaySet(source=source, sha256=digest.hexdigest(), path=used, pool=pool,
                     anchor=anchor)


def mix_replay(
    domain: list[Sample], pool: list[Sample], beta: float, *, seed: int, salt: str = ""
) -> tuple[list[Sample], list[str]]:
    """Return (mixed training set, replay ids) with replay ≈ ``beta`` of all tokens.

    Target replay tokens R = β/(1−β)·D. Samples are drawn in a seed-fixed order; the pool
    is cycled (with repeats) if it is too small. The output order is a seed-fixed shuffle.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1): {beta}")
    rng = random.Random(f"{seed}:replay-mix:{salt}")
    d_tokens = sum(approx_tokens(s["messages"]) for s in domain)
    target = beta / (1.0 - beta) * d_tokens
    chosen: list[Sample] = []
    got = 0
    if pool and target > 0:
        order = list(range(len(pool)))
        rng.shuffle(order)
        i = 0
        while got < target:
            s = pool[order[i % len(order)]]
            t = approx_tokens(s["messages"])
            # stop if adding this sample overshoots more than stopping undershoots
            if got + t - target > target - got:
                break
            chosen.append(s)
            got += t
            i += 1
    mixed = list(domain) + chosen
    rng.shuffle(mixed)
    return mixed, [s["id"] for s in chosen]


def replay_token_fraction(mixed: list[Sample], replay_ids: set[str]) -> float:
    """Fraction of tokens that are replay (for tests and the report)."""
    total = sum(approx_tokens(s["messages"]) for s in mixed)
    rep = sum(approx_tokens(s["messages"]) for s in mixed if s["id"] in replay_ids)
    return rep / total if total else 0.0
