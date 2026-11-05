"""Lightweight token counting for chunk sizing.

Uses the student model's Hugging Face tokenizer from the local cache when available
(``local_files_only=True``; never downloads), else a ~4-chars-per-token heuristic
(fake/unknown models, or no cached tokenizer).
"""

from __future__ import annotations

import functools
import math
from typing import Any, Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """``ceil(len(text) / chars_per_token)`` — deterministic, dependency-free."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = chars_per_token
        self.name = f"heuristic-{chars_per_token:g}cpt"

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0


class HFCounter:
    """Counts with a cached ``transformers`` tokenizer (no special tokens)."""

    def __init__(self, tokenizer: Any, model: str) -> None:
        self._tok = tokenizer
        self.name = f"hf:{model}"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False)) if text else 0


@functools.cache
def _load_hf(model: str) -> Any | None:
    try:
        import logging

        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]
        logging.getLogger("transformers").setLevel(logging.ERROR)
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        tok.model_max_length = 10**9  # silence "sequence too long" warnings
        return tok
    except Exception:
        return None


def get_token_counter(model: str | None) -> TokenCounter:
    """Tokenizer for ``model`` if it is a cached HF repo/local path, else the heuristic."""
    if model and ("/" in model) and not model.startswith("fake"):
        tok = _load_hf(model)
        if tok is not None:
            return HFCounter(tok, model)
    return HeuristicCounter()
