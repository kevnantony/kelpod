"""Policy engine: thresholding, deterministic overlap resolution, typed placeholders.

Pure functions over :class:`Detection` spans; no Presidio/spaCy import, no I/O.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["keep", "mask", "drop_chunk"]


@dataclass(frozen=True)
class Detection:
    """A candidate PII span from any recogniser (text is never stored)."""

    start: int
    end: int
    entity_type: str
    score: float
    source: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ResolvedSpan:
    start: int
    end: int
    entity_type: str
    score: float
    action: Action
    members: int = 1  # number of raw detections merged into this span


@dataclass
class Policy:
    """Effective policy (entity type -> action) with a default for unknown types."""

    actions: dict[str, Action] = field(default_factory=dict)
    default_action: Action = "mask"
    score_threshold: float = 0.5

    def action_for(self, entity_type: str) -> Action:
        return self.actions.get(entity_type, self.default_action)


def _winner_key(d: Detection) -> tuple[float, int, int, str, str]:
    # highest score, then longest, then earliest, then type/source name (total order)
    return (-d.score, -d.length, d.start, d.entity_type, d.source)


def cluster_overlaps(detections: Iterable[Detection]) -> list[list[Detection]]:
    """Group detections into connected components of overlapping spans (sorted by start)."""
    items = sorted(detections, key=lambda d: (d.start, -d.end, d.entity_type, d.source))
    clusters: list[list[Detection]] = []
    current_end = -1
    for d in items:
        if d.length <= 0:
            continue
        if clusters and d.start < current_end:
            clusters[-1].append(d)
            current_end = max(current_end, d.end)
        else:
            clusters.append([d])
            current_end = d.end
    return clusters


def resolve(detections: Iterable[Detection], policy: Policy) -> list[ResolvedSpan]:
    """Threshold, then resolve overlaps deterministically.

    Within each cluster of overlapping detections the winner is the highest score, then the
    longest span. The winner's entity type decides the action. When the action is ``mask`` or
    ``drop_chunk`` the resolved span is the *union* of the cluster, so no fragment of a
    lower-ranked overlapping span can leak; a ``keep`` winner keeps its own span only.
    """
    kept = [d for d in detections if d.score >= policy.score_threshold]
    out: list[ResolvedSpan] = []
    for cluster in cluster_overlaps(kept):
        winner = min(cluster, key=_winner_key)
        action = policy.action_for(winner.entity_type)
        if action == "keep":
            start, end = winner.start, winner.end
        else:
            start = min(d.start for d in cluster)
            end = max(d.end for d in cluster)
        out.append(ResolvedSpan(start, end, winner.entity_type, round(winner.score, 4),
                                action, members=len(cluster)))
    return out


def chunk_should_drop(detections: Iterable[Detection], policy: Policy) -> bool:
    """A chunk is dropped when ANY detection above threshold maps to ``drop_chunk``.

    This is deliberately stricter than overlap resolution: a classification marking that
    lost an overlap to another entity still removes the chunk.
    """
    return any(
        d.score >= policy.score_threshold and policy.action_for(d.entity_type) == "drop_chunk"
        for d in detections
    )


# ---------------------------------------------------------------- placeholders
_COMPACT_TYPES = {"PHONE_NUMBER", "IBAN_CODE", "DE_KVNR", "DE_TAX_ID", "US_SSN", "PATIENT_ID"}


def normalize_value(value: str, entity_type: str = "") -> str:
    """Canonical form of a value for placeholder consistency (NFKC, casefold, whitespace)."""
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    if entity_type in _COMPACT_TYPES:
        return re.sub(r"[\s./()\-]+", "", text)
    return re.sub(r"\s+", " ", text)


class PlaceholderMapper:
    """Assigns ``[TYPE_n]`` consistently per document, keyed by a salted HMAC of the value.

    The mapping lives in memory only and is never persisted (P1.2.3).
    """

    def __init__(self, salt: bytes) -> None:
        self._salt = salt
        self._maps: dict[tuple[str, str], dict[str, int]] = {}

    def _key(self, doc_id: str, entity_type: str, value: str) -> str:
        msg = f"{doc_id}\0{entity_type}\0{normalize_value(value, entity_type)}".encode()
        return hmac.new(self._salt, msg, hashlib.sha256).hexdigest()

    def placeholder(self, doc_id: str, entity_type: str, value: str) -> str:
        table = self._maps.setdefault((doc_id, entity_type), {})
        key = self._key(doc_id, entity_type, value)
        if key not in table:
            table[key] = len(table) + 1
        return f"[{entity_type}_{table[key]}]"

    def clear(self) -> None:
        self._maps.clear()


def apply_masks(
    text: str,
    spans: Sequence[ResolvedSpan],
    doc_id: str,
    mapper: PlaceholderMapper,
) -> tuple[str, list[str | None]]:
    """Replace ``mask`` spans by placeholders (left to right, for deterministic numbering).

    Returns the masked text and, per input span, the placeholder used (None if not masked).
    """
    order = sorted(range(len(spans)), key=lambda i: (spans[i].start, spans[i].end))
    placeholders: list[str | None] = [None] * len(spans)
    pieces: list[str] = []
    cursor = 0
    for i in order:
        span = spans[i]
        if span.action != "mask" or span.start < cursor:
            continue
        ph = mapper.placeholder(doc_id, span.entity_type, text[span.start:span.end])
        placeholders[i] = ph
        pieces.append(text[cursor:span.start])
        pieces.append(ph)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), placeholders


def policy_from_config(
    actions: Mapping[str, str], default_action: str, score_threshold: float,
    entity_types: Iterable[str] = (),
) -> Policy:
    """Effective policy: configured actions plus the default for every other known type."""
    merged: dict[str, Action] = {t: default_action for t in entity_types}  # type: ignore[misc]
    merged.update({k: v for k, v in actions.items()})  # type: ignore[misc]
    return Policy(actions=dict(sorted(merged.items())),
                  default_action=default_action,  # type: ignore[arg-type]
                  score_threshold=score_threshold)

