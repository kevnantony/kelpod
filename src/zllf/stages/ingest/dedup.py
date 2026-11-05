"""Chunk de-duplication (P1.1.8): exact hash, then MinHash LSH on word shingles.

Chunks are processed in the given (deterministic) order and the first occurrence is kept.
MinHash uses a fixed seed, so results are reproducible across runs and machines.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from zllf.core.hashing import sha256_text

_WORD = re.compile(r"\w+", re.UNICODE)
MINHASH_SEED = 1


@dataclass
class DedupResult:
    duplicate_of: dict[int, int]  # index of removed chunk -> index of the kept original
    exact: int = 0
    near: int = 0


def normalized_key(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def shingles(text: str, size: int) -> set[str]:
    """Word ``size``-grams of the normalised text (whole text if shorter)."""
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def find_duplicates(texts: Sequence[str], jaccard: float, num_perm: int = 128,
                    shingle_size: int = 5) -> DedupResult:
    """Return which texts are exact or near (estimated Jaccard >= ``jaccard``) duplicates."""
    result = DedupResult(duplicate_of={})
    first_by_hash: dict[str, int] = {}
    survivors: list[int] = []
    for i, text in enumerate(texts):
        key = sha256_text(normalized_key(text))
        if key in first_by_hash:
            result.duplicate_of[i] = first_by_hash[key]
            result.exact += 1
        else:
            first_by_hash[key] = i
            survivors.append(i)
    if not 0.0 < jaccard < 1.0:
        return result

    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
    hashes: dict[int, MinHash] = {}
    for i in survivors:
        sh = shingles(texts[i], shingle_size)
        if not sh:
            continue
        mh = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
        mh.update_batch([s.encode("utf-8") for s in sorted(sh)])
        candidates = sorted(int(c) for c in lsh.query(mh))
        match = next((c for c in candidates if hashes[c].jaccard(mh) >= jaccard), None)
        if match is not None:
            result.duplicate_of[i] = match
            result.near += 1
            continue
        lsh.insert(str(i), mh)
        hashes[i] = mh
    return result
