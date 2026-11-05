"""Model interfaces used by stages. Implementations: fake (tests/CI), mlx (Apple Silicon),
vllm (Linux/CUDA, later). Stages must only depend on these interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zllf.core.schemas import Message

Chat = Sequence[Message]


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int = 0


class LLMBackend(ABC):
    """A chat model, optionally with a LoRA adapter applied."""

    model_id: str

    @abstractmethod
    def chat(
        self,
        batch: Sequence[Chat],
        params: SamplingParams = SamplingParams(),
        *,
        json_schema: dict[str, Any] | None = None,
        adapter_path: Path | None = None,
    ) -> list[str]:
        """Generate one completion per chat. With ``json_schema`` the output must be JSON
        (constrained decoding where the backend supports it, else best effort + repair)."""

    @abstractmethod
    def loglikelihood(
        self,
        prompts: Sequence[Chat],
        completions: Sequence[str],
        *,
        adapter_path: Path | None = None,
    ) -> list[float]:
        """Sum of log-probabilities of each completion given its chat prompt
        (multiple-choice scoring and validation loss)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free model memory."""

    def __enter__(self) -> LLMBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Embedder(ABC):
    model_id: str

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised embeddings, shape (n, dim), float32."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free model memory."""


class NLIScorer(ABC):
    model_id: str

    @abstractmethod
    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        """Probability in [0, 1] that each premise entails its hypothesis."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free model memory."""
