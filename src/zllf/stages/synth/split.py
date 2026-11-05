"""Document-level train/val/test split (PRD §11.2): all examples of a document share a split,
so test questions never come from training documents."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from zllf.core.config import SplitConfig
from zllf.core.schemas import Split


@dataclass
class SplitPlan:
    doc_split: dict[str, Split]
    warnings: list[str] = field(default_factory=list)
    val_from_train_docs: bool = False  # val carved from train docs' examples (tiny corpora)

    def docs(self, split: Split) -> list[str]:
        return sorted(d for d, s in self.doc_split.items() if s == split)


def _n_docs(n: int, ratios: SplitConfig) -> tuple[int, int]:
    """(n_val, n_test) documents. Tiny corpora: test first, val may be 0."""
    if n <= 1:
        return 0, 0
    n_test = max(1, round(n * ratios.test)) if ratios.test > 0 else 0
    n_val = max(1, round(n * ratios.val)) if ratios.val > 0 and n >= 3 else 0
    while n - n_test - n_val < 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n_val, n_test


def plan_split(
    doc_weights: Mapping[str, int], ratios: SplitConfig, rng: random.Random
) -> SplitPlan:
    """Assign each document to a split.

    Documents are shuffled with ``rng`` (deterministic), the required number of val/test
    documents is seeded first, then remaining documents go greedily to the split whose
    example share is furthest below its target ratio (weights = examples per document).
    """
    docs = sorted(doc_weights)
    rng.shuffle(docs)
    n = len(docs)
    warnings: list[str] = []
    n_val, n_test = _n_docs(n, ratios)
    plan: dict[str, Split] = {}
    totals: dict[Split, float] = {"train": 0.0, "val": 0.0, "test": 0.0}
    queue = list(docs)
    for split, k in (("test", n_test), ("val", n_val)):
        for _ in range(k):
            doc = queue.pop()
            plan[doc] = split  # type: ignore[assignment]
            totals[split] += max(1, doc_weights[doc])  # type: ignore[index]
    if queue:
        first = queue.pop(0)
        plan[first] = "train"
        totals["train"] += max(1, doc_weights[first])
    target: dict[Split, float] = {"train": ratios.train, "val": ratios.val, "test": ratios.test}
    allowed: list[Split] = ["train"]
    if n_val:
        allowed.append("val")
    if n_test:
        allowed.append("test")
    grand = float(sum(max(1, doc_weights[d]) for d in docs)) or 1.0
    for doc in queue:
        split = max(allowed, key=lambda s: (target[s] * grand - totals[s], s == "train"))
        plan[doc] = split
        totals[split] += max(1, doc_weights[doc])
    if n < 3:
        warnings.append(f"tiny corpus: {n} document(s); document-level val/test split degraded")
    if n_test == 0 and ratios.test > 0:
        warnings.append("no test documents: held-out evaluation is not possible")
    return SplitPlan(doc_split=plan, warnings=warnings, val_from_train_docs=False)


def carve_val(
    train_ids: Sequence[str], fraction: float, rng: random.Random
) -> set[str]:
    """Example ids moved from train to val when there is no val document (tiny corpora)."""
    if not train_ids or fraction <= 0:
        return set()
    k = max(1, round(len(train_ids) * fraction))
    k = min(k, len(train_ids) - 1)
    if k <= 0:
        return set()
    return set(rng.sample(sorted(train_ids), k))
