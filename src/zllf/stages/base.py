"""Stage interface and run context shared by all pipeline stages.

A stage reads committed outputs of earlier stages via ``ctx.paths.stage(name)``,
writes only into ``out_dir`` (a temp dir committed atomically by the runner), and
returns a :class:`StageResult`. Stages must be deterministic given config + seed.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from zllf.core.config import RunConfig
from zllf.core.errors import CancelledError
from zllf.core.manifest import RunPaths, append_event

if TYPE_CHECKING:
    from zllf.llm.base import Embedder, LLMBackend, NLIScorer

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class StageResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    needs_approval: bool = False  # pause the run until the user approves (sanitize)


@dataclass
class RunContext:
    run_id: str
    paths: RunPaths
    corpus_dir: Path
    config: RunConfig
    cancel_requested: Callable[[], bool] = lambda: False
    event_sink: EventSink | None = None

    # ---------------------------------------------------------------- events
    def emit(self, type_: str, **payload: Any) -> None:
        """Progress/status event for the UI. Never put document text in payload."""
        event = {"run_id": self.run_id, "type": type_, **payload}
        append_event(self.paths, event)
        if self.event_sink is not None:
            self.event_sink(event)

    def progress(self, stage: str, done: int, total: int, message: str = "") -> None:
        self.emit("progress", stage=stage, done=done, total=total, message=message)

    def check_cancelled(self) -> None:
        if self.cancel_requested():
            raise CancelledError()

    # ---------------------------------------------------------------- models
    def llm(self, role: str) -> LLMBackend:
        """role: 'teacher' | 'student' | 'judge'. Caller must call .close() when done."""
        from zllf.llm.factory import get_llm

        return get_llm(self.config, role)

    def nli(self) -> NLIScorer:
        from zllf.llm.factory import get_nli

        return get_nli(self.config)

    def embedder(self) -> Embedder:
        from zllf.llm.factory import get_embedder

        return get_embedder(self.config)

    # ---------------------------------------------------------------- rng
    def rng(self, salt: str = "") -> random.Random:
        """Deterministic RNG per (seed, salt)."""
        return random.Random(f"{self.config.seed}:{salt}")

    def np_rng(self, salt: str = "") -> np.random.Generator:
        # not hash(): str hashing is randomised per process
        digest = hashlib.sha256(f"{self.config.seed}:{salt}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class Stage(ABC):
    name: ClassVar[str]

    @abstractmethod
    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        """Do the work, writing artefacts into ``out_dir``."""
