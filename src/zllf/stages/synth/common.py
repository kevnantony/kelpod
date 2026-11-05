"""Shared types for the synthesis stage: candidate pairs, statistics and the batched
teacher-call helper."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from zllf.core.schemas import Chunk, Message
from zllf.llm.base import Chat, LLMBackend, SamplingParams

TEACHER_BATCH = 16  # chats per LLMBackend.chat() call
NLI_BATCH = 64
EMBED_BATCH = 256

ProgressFn = Callable[[str, int, int], None]  # (phase, done, total)


@dataclass
class Candidate:
    """A generated (not yet judged) training pair."""

    task: str
    user: str  # what the student sees as the user turn
    answer: str  # assistant target
    chunk: Chunk | None = None
    doc_id: str | None = None
    quote: str | None = None
    hypothesis: str | None = None  # NLI hypothesis (answer restated as a statement)
    premise: str | None = None  # NLI premise; None -> chunk text
    judge_question: str | None = None  # question shown to the LLM judge (default: user)
    reference_json: dict[str, Any] | None = None
    answerable: bool = True
    entailment: float | None = None
    judge_ok: bool | None = None
    needs_quote_check: bool = True
    needs_nli: bool = True
    needs_judge: bool = True

    @property
    def chunk_id(self) -> str | None:
        return self.chunk.chunk_id if self.chunk is not None else None


@dataclass
class SynthStats:
    """Counters that end up in synth_report.json (counts only, never text)."""

    candidates: Counter[str] = field(default_factory=Counter)
    kept: Counter[str] = field(default_factory=Counter)
    drops: defaultdict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    malformed: Counter[str] = field(default_factory=Counter)  # teacher phase -> count
    judge_checked: Counter[str] = field(default_factory=Counter)  # filter -> n checked
    judge_passed: Counter[str] = field(default_factory=Counter)  # filter -> n passed
    quotes_repaired: int = 0
    teacher_calls: int = 0
    teacher_batches: int = 0
    notes: list[str] = field(default_factory=list)

    def drop(self, task: str, reason: str, n: int = 1) -> None:
        self.drops[task][reason] += n

    def check(self, name: str, passed: bool) -> None:
        self.judge_checked[name] += 1
        self.judge_passed[name] += int(passed)

    def pass_rates(self) -> dict[str, float]:
        checked = sorted(self.judge_checked.items())
        return {k: round(self.judge_passed[k] / n, 4) for k, n in checked if n}


class Teacher:
    """Batches chats into ``LLMBackend.chat`` calls and reports progress/cancellation."""

    def __init__(
        self,
        llm: LLMBackend,
        params: SamplingParams,
        stats: SynthStats,
        progress: ProgressFn | None = None,
        check_cancelled: Callable[[], None] | None = None,
        batch_size: int = TEACHER_BATCH,
    ) -> None:
        self.llm = llm
        self.params = params
        self.stats = stats
        self.progress = progress
        self.check_cancelled = check_cancelled
        self.batch_size = batch_size

    def run(
        self,
        phase: str,
        chats: Sequence[Sequence[Message]],
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """One completion per chat, in order."""
        params = self.params
        if max_tokens is not None:
            params = SamplingParams(
                temperature=params.temperature, max_tokens=max_tokens, top_p=params.top_p,
                stop=params.stop, seed=params.seed,
            )
        out: list[str] = []
        total = len(chats)
        for start in range(0, total, self.batch_size):
            if self.check_cancelled is not None:
                self.check_cancelled()
            batch: list[Chat] = [list(c) for c in chats[start : start + self.batch_size]]
            out.extend(self.llm.chat(batch, params, json_schema=json_schema))
            self.stats.teacher_calls += len(batch)
            self.stats.teacher_batches += 1
            if self.progress is not None:
                self.progress(phase, min(start + self.batch_size, total), total)
        return out
