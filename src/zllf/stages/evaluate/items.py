"""Build :class:`EvalItem` suites from synth outputs, the gold set and canaries.

Suites: ``domain_test`` (answerable synthetic test split), ``gold`` (user gold set, FR-14),
``unanswerable`` (held-out unanswerables, gold unanswerables and adversarial prompts about
non-existent codes), ``anchor/<name>`` (see :mod:`anchor`) and ``canary`` (memorisation
probes, see :mod:`memorization`).

Gold set location (first existing file wins):

1. ``<run_dir>/gold.jsonl``           — imported for this run;
2. ``<corpus_dir>/../gold.jsonl``     — shipped next to the corpus folder.

Each line: ``{"question": str, "reference_answer"?: str, "reference_json"?: object,
"answerable"?: bool (default true), "id"?: str}``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from zllf.core.schemas import EvalItem, Message, SynthExample

N_ADVERSARIAL = 20

_ADVERSARIAL_TEMPLATES = (
    "According to the manual, what is the procedure for {code}?",
    "What does the documentation say about fault code {code}?",
    "Which section of the documents describes the reset sequence for {code}?",
    "What is the approved inspection interval for component {code} in the documents?",
    "Summarise the safety instructions for error {code} as given in the manual.",
)


class GoldItem(BaseModel):
    question: str
    reference_answer: str | None = None
    reference_json: dict[str, Any] | None = None
    answerable: bool = True
    id: str | None = None


def with_system(prompt: Sequence[Message], system_prompt: str) -> list[Message]:
    msgs = list(prompt)
    if not any(m.role == "system" for m in msgs):
        msgs.insert(0, Message(role="system", content=system_prompt))
    return msgs


def prompt_of(example: SynthExample) -> list[Message]:
    """Chat up to (excluding) the final assistant turn."""
    msgs = list(example.messages)
    while msgs and msgs[-1].role == "assistant":
        msgs.pop()
    return msgs


def reference_of(example: SynthExample) -> str | None:
    if example.reference_answer:
        return example.reference_answer
    if example.reference_json is not None:
        return json.dumps(example.reference_json, ensure_ascii=False, sort_keys=True)
    last = example.messages[-1] if example.messages else None
    return last.content if last is not None and last.role == "assistant" else None


def last_user(prompt: Sequence[Message]) -> str:
    return next((m.content for m in reversed(prompt) if m.role == "user"), "")


def subsample(items: list[EvalItem], limit: int | None, rng: random.Random) -> list[EvalItem]:
    """Deterministic subset (original order kept)."""
    if limit is None or len(items) <= limit:
        return items
    keep = set(rng.sample(range(len(items)), limit))
    return [it for i, it in enumerate(items) if i in keep]


def domain_items(examples: Sequence[SynthExample], system_prompt: str) -> list[EvalItem]:
    out = []
    for ex in examples:
        if not ex.answerable or ex.task in ("unanswerable", "canary"):
            continue
        prompt = prompt_of(ex)
        if not prompt:
            continue
        out.append(EvalItem(
            item_id=ex.example_id, suite="domain_test", prompt=with_system(prompt, system_prompt),
            reference=reference_of(ex), reference_json=ex.reference_json, answerable=True,
        ))
    return out


def heldout_unanswerable_items(examples: Sequence[SynthExample], system_prompt: str,
                               ) -> list[EvalItem]:
    out = []
    for ex in examples:
        if ex.answerable and ex.task != "unanswerable":
            continue
        prompt = prompt_of(ex)
        if prompt:
            out.append(EvalItem(
                item_id=ex.example_id, suite="unanswerable",
                prompt=with_system(prompt, system_prompt), reference=reference_of(ex),
                answerable=False,
            ))
    return out


def _fake_code(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    return f"{letters}-{rng.randint(1000, 9999)}-{rng.choice('QXZW')}{rng.randint(1, 9)}"


def adversarial_items(system_prompt: str, rng: random.Random, n: int = N_ADVERSARIAL,
                      known_text: str = "") -> list[EvalItem]:
    """Questions about plausible but non-existent codes (P2.6.5). Codes that happen to occur
    in ``known_text`` (e.g. the corpus) are skipped."""
    out: list[EvalItem] = []
    seen: set[str] = set()
    while len(out) < n:
        code = _fake_code(rng)
        if code in seen or (known_text and code in known_text):
            continue
        seen.add(code)
        template = _ADVERSARIAL_TEMPLATES[len(out) % len(_ADVERSARIAL_TEMPLATES)]
        out.append(EvalItem(
            item_id=f"adv-{len(out):03d}", suite="unanswerable",
            prompt=with_system([Message(role="user", content=template.format(code=code))],
                               system_prompt),
            reference=None, answerable=False,
        ))
    return out


def find_gold_file(run_root: Path, corpus_dir: Path) -> Path | None:
    for path in (run_root / "gold.jsonl", corpus_dir.parent / "gold.jsonl"):
        if path.is_file():
            return path
    return None


def load_gold(path: Path, system_prompt: str) -> tuple[list[EvalItem], list[EvalItem]]:
    """(answerable gold items -> suite ``gold``, unanswerable gold items -> ``unanswerable``)."""
    answerable: list[EvalItem] = []
    unanswerable: list[EvalItem] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            g = GoldItem.model_validate_json(line)
            digest = hashlib.sha256(g.question.encode()).hexdigest()[:10]
            item_id = g.id or f"gold-{n:04d}-{digest}"
            prompt = with_system([Message(role="user", content=g.question)], system_prompt)
            ref = g.reference_answer
            if ref is None and g.reference_json is not None:
                ref = json.dumps(g.reference_json, ensure_ascii=False, sort_keys=True)
            if g.answerable:
                answerable.append(EvalItem(item_id=item_id, suite="gold", prompt=prompt,
                                           reference=ref, reference_json=g.reference_json))
            else:
                unanswerable.append(EvalItem(item_id=item_id, suite="unanswerable",
                                             prompt=prompt, reference=ref, answerable=False))
    return answerable, unanswerable


class SuiteSet(BaseModel):
    """All suites of one evaluation run."""

    domain: dict[str, list[EvalItem]] = Field(default_factory=dict)  # domain_test / gold
    unanswerable: list[EvalItem] = Field(default_factory=list)
    anchors: dict[str, list[EvalItem]] = Field(default_factory=dict)  # anchor/<name>

    def counts(self) -> dict[str, int]:
        out = {k: len(v) for k, v in self.domain.items()}
        out["unanswerable"] = len(self.unanswerable)
        out.update({k: len(v) for k, v in self.anchors.items()})
        return out
