"""Salted fingerprints of redacted values, for leakage checks without storing PII.

``pii_fingerprints.json`` holds ``sha256(salt || normalised value)`` hex digests only. The salt
is derived from ``run_id`` + ``seed`` (:func:`derive_salt`) and never written; the file stores
``salt_id`` (hash of the salt) so a consumer can verify it re-derived the right salt.

The evaluate stage (memorisation gate, P2.6.4) uses::

    from zllf.stages.sanitize.fingerprints import (
        contains_fingerprint, derive_salt, load_fingerprints)
    fp = load_fingerprints(paths.stage("sanitize") / "pii_fingerprints.json")
    salt = derive_salt(run_id, seed)
    leaked = contains_fingerprint(model_output, fp.hashes, salt)
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINGERPRINT_VERSION = 1
MAX_NGRAM = 8  # longest value (in normalised tokens) that can be found by the n-gram scan
MIN_CHARS = 4  # values shorter than this (compact form) are not fingerprinted (false hits)
_NON_ALNUM = re.compile(r"[^\w]+|_+")


def derive_salt(run_id: str, seed: int) -> bytes:
    """Per-run salt (deterministic given run_id + seed; never persisted)."""
    return hashlib.sha256(f"zllf-pii-fingerprint:v1:{run_id}:{seed}".encode()).digest()


def salt_id(salt: bytes | str) -> str:
    """Non-reversible identifier of a salt (stored next to the fingerprints)."""
    return hashlib.sha256(b"salt-id:" + _as_bytes(salt)).hexdigest()[:16]


def _as_bytes(salt: bytes | str) -> bytes:
    return salt.encode("utf-8") if isinstance(salt, str) else salt


def normalize_tokens(text: str) -> list[str]:
    """NFKC + casefold, split on anything that is not a letter/digit."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in _NON_ALNUM.split(norm) if t]


def _hash(salt: bytes, form: str) -> str:
    return hashlib.sha256(salt + b"\0" + form.encode("utf-8")).hexdigest()


def value_forms(value: str) -> list[str]:
    """Normalised forms of a value: space-joined tokens and the compact (no separator) form."""
    tokens = normalize_tokens(value)
    if not tokens:
        return []
    compact = "".join(tokens)
    if len(compact) < MIN_CHARS:
        return []
    spaced = " ".join(tokens)
    return [spaced] if spaced == compact else [spaced, compact]


def fingerprint_value(value: str, salt: bytes | str) -> list[str]:
    """Fingerprints (hex digests) for one redacted value."""
    s = _as_bytes(salt)
    return [_hash(s, form) for form in value_forms(value)]


def _ngram_forms(tokens: list[str], max_n: int) -> Iterable[str]:
    for i in range(len(tokens)):
        for n in range(1, max_n + 1):
            if i + n > len(tokens):
                break
            window = tokens[i:i + n]
            yield " ".join(window)
            if n > 1:
                yield "".join(window)


def find_fingerprint_hits(
    text: str, fingerprints: Collection[str], salt: bytes | str, max_n: int = MAX_NGRAM,
) -> set[str]:
    """Fingerprints from ``fingerprints`` whose value occurs in ``text`` (token n-gram scan)."""
    if not fingerprints:
        return set()
    s = _as_bytes(salt)
    fp = fingerprints if isinstance(fingerprints, (set, frozenset)) else set(fingerprints)
    hits: set[str] = set()
    for form in _ngram_forms(normalize_tokens(text), max_n):
        if len(form.replace(" ", "")) < MIN_CHARS:
            continue
        h = _hash(s, form)
        if h in fp:
            hits.add(h)
    return hits


def contains_fingerprint(text: str, fingerprints: Collection[str], salt: bytes | str) -> bool:
    """True if any fingerprinted (redacted) value occurs in ``text``."""
    return bool(find_fingerprint_hits(text, fingerprints, salt))


@dataclass
class FingerprintSet:
    salt_id: str
    hashes: frozenset[str]
    by_type: dict[str, list[str]]
    max_ngram: int = MAX_NGRAM


class FingerprintBuilder:
    """Collects fingerprints during the stage; only hashes are retained."""

    def __init__(self, salt: bytes) -> None:
        self._salt = salt
        self._by_type: dict[str, set[str]] = {}

    def add(self, entity_type: str, value: str) -> None:
        hashes = fingerprint_value(value, self._salt)
        if hashes:
            self._by_type.setdefault(entity_type, set()).update(hashes)

    def to_json(self) -> dict[str, Any]:
        by_type = {k: sorted(v) for k, v in sorted(self._by_type.items())}
        all_hashes = sorted({h for v in by_type.values() for h in v})
        return {
            "version": FINGERPRINT_VERSION,
            "algorithm": "sha256(salt || 0x00 || nfkc_casefold_tokens)",
            "salt_id": salt_id(self._salt),
            "salt_derivation": "sha256('zllf-pii-fingerprint:v1:<run_id>:<seed>')",
            "max_ngram": MAX_NGRAM,
            "min_chars": MIN_CHARS,
            "count": len(all_hashes),
            "fingerprints": all_hashes,
            "by_type": by_type,
        }

    def write(self, path: Path) -> int:
        data = self.to_json()
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return int(data["count"])


def load_fingerprints(path: Path) -> FingerprintSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FingerprintSet(
        salt_id=str(data["salt_id"]),
        hashes=frozenset(data.get("fingerprints", [])),
        by_type={k: list(v) for k, v in (data.get("by_type") or {}).items()},
        max_ngram=int(data.get("max_ngram", MAX_NGRAM)),
    )
