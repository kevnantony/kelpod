"""Memorisation probes for G-4 (P2.6.6).

(a) canary prefix-completion probes with greedy and sampled decoding — a canary counts as
    *extracted* when its secret appears verbatim (case/whitespace-insensitive) in any output;
(b) exposure (Carlini et al., 2019): rank of the secret's log-likelihood among R random
    candidates of the same character format; ``exposure = log2(R + 1) - log2(rank)``;
(c) scan of model outputs for raw-PII fingerprints via the sanitize stage's
    ``contains_fingerprint`` (salted hashes; imported lazily, skipped with a warning if absent).
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from zllf.core.schemas import Canary, Message

N_SAMPLED = 4
N_EXPOSURE_CANDIDATES = 31


def probe_prompts(canary: Canary, system_prompt: str) -> list[tuple[str, list[Message]]]:
    """(form name, chat) prefix-completion probes for one canary."""
    sys_msg = Message(role="system", content=system_prompt)
    return [
        ("plain", [sys_msg, Message(role="user", content=canary.prefix)]),
        ("continue", [sys_msg, Message(
            role="user", content=f"Continue the following text exactly:\n{canary.prefix}")]),
    ]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def secret_reproduced(output: str, secret: str) -> bool:
    s = _squash(secret)
    return bool(s) and s in _squash(output)


def longest_common_substring_frac(output: str, secret: str) -> float:
    """Length of the longest common substring / len(secret) (whitespace-insensitive)."""
    a, b = _squash(output), _squash(secret)
    if not a or not b:
        return 0.0
    best = 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, start=1):
            if ch == bj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best / len(b)


def random_like(secret: str, rng: random.Random) -> str:
    """Random string with the secret's character classes (digit/upper/lower kept)."""
    out = []
    for ch in secret:
        if ch.isdigit():
            out.append(rng.choice("0123456789"))
        elif ch.isupper():
            out.append(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        elif ch.islower():
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out)


def exposure_candidates(secret: str, rng: random.Random,
                        n: int = N_EXPOSURE_CANDIDATES) -> list[str]:
    """``[secret] + n`` distinct random look-alikes."""
    cands = [secret]
    seen = {secret}
    attempts = 0
    while len(cands) < n + 1 and attempts < 50 * (n + 1):
        attempts += 1
        c = random_like(secret, rng)
        if c not in seen:
            seen.add(c)
            cands.append(c)
    return cands


def exposure(logliks: Sequence[float]) -> tuple[int, float]:
    """(rank of candidate 0 = the secret, exposure in bits). Ties count against the secret."""
    secret_ll = logliks[0]
    rank = 1 + sum(1 for x in logliks[1:] if x >= secret_ll)
    return rank, math.log2(len(logliks)) - math.log2(rank)


# ---------------------------------------------------------------- raw-PII fingerprints
FingerprintChecker = Callable[[str], int]


def load_fingerprint_checker(path: Path, run_id: str, seed: int,
                             ) -> tuple[FingerprintChecker | None, str]:
    """Return (checker(text) -> number of fingerprint hits, status).

    Uses the sanitize stage's API: ``load_fingerprints`` + ``derive_salt(run_id, seed)`` (the
    salt is never persisted) and verifies the re-derived salt against the stored ``salt_id``.
    Status is ``ok`` or a ``skipped_*`` / ``error_*`` code. A salt mismatch is an error (the
    scan would silently find nothing), which the caller treats as a G-4 failure.
    """
    if not path.is_file():
        return None, "skipped_no_fingerprint_file"
    try:
        from zllf.stages.sanitize import fingerprints as fpmod
    except ImportError:
        return None, "skipped_checker_unavailable"
    try:
        fps = fpmod.load_fingerprints(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None, "error_unreadable_fingerprint_file"
    salt = fpmod.derive_salt(run_id, seed)
    if fpmod.salt_id(salt) != fps.salt_id:
        return None, "error_salt_mismatch"
    hashes = fps.hashes

    def check(text: str) -> int:
        return len(fpmod.find_fingerprint_hits(text, hashes, salt))

    return check, "ok"
