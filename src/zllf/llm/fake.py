"""Deterministic fake models for unit tests and CI (no weights, no accelerator).

The outputs are crude but structurally valid, so whole pipelines can run end to end.
Tests can pass a ``responder`` to script exact outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zllf.llm.base import Chat, Embedder, LLMBackend, NLIScorer, SamplingParams

Responder = Callable[[Chat, dict[str, Any] | None, Path | None], str]

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def instance_from_schema(schema: dict[str, Any], seed_text: str = "") -> Any:
    """Smallest plausible instance of a (simple) JSON schema."""
    words = _words(seed_text) or ["value"]
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties", {})
        return {k: instance_from_schema(v, seed_text) for k, v in props.items()}
    if t == "array":
        item = schema.get("items", {"type": "string"})
        n = max(1, int(schema.get("minItems", 1)))
        return [instance_from_schema(item, seed_text) for _ in range(n)]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if t == "null":
        return None
    return " ".join(words[:6])


class FakeLLM(LLMBackend):
    def __init__(self, model_id: str = "fake", responder: Responder | None = None) -> None:
        self.model_id = model_id
        self.responder = responder
        self.calls = 0

    def chat(
        self,
        batch: Sequence[Chat],
        params: SamplingParams = SamplingParams(),
        *,
        json_schema: dict[str, Any] | None = None,
        adapter_path: Path | None = None,
    ) -> list[str]:
        out: list[str] = []
        for chat in batch:
            self.calls += 1
            if self.responder is not None:
                out.append(self.responder(chat, json_schema, adapter_path))
                continue
            user = next((m.content for m in reversed(chat) if m.role == "user"), "")
            if json_schema is not None:
                out.append(json.dumps(instance_from_schema(json_schema, user)))
            else:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user) if s.strip()]
                text = sentences[-1] if sentences else user
                out.append(" ".join(text.split()[: params.max_tokens]))
        return out

    def loglikelihood(
        self,
        prompts: Sequence[Chat],
        completions: Sequence[str],
        *,
        adapter_path: Path | None = None,
    ) -> list[float]:
        scores = []
        for chat, completion in zip(prompts, completions, strict=True):
            ctx = set(_words(" ".join(m.content for m in chat)))
            toks = _words(completion) or [""]
            overlap = sum(1 for t in toks if t in ctx) / len(toks)
            scores.append(float(np.log(0.05 + 0.95 * overlap) * len(toks)))
        return scores

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeEmbedder(Embedder):
    """Hashed bag-of-words embedding: texts sharing words have high cosine similarity."""

    def __init__(self, model_id: str = "fake-embedder", dim: int = 256) -> None:
        self.model_id = model_id
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in _words(text):
                idx = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % self.dim
                mat[i, idx] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class FakeNLI(NLIScorer):
    """Entailment ~ fraction of hypothesis content words found in the premise."""

    def __init__(self, model_id: str = "fake-nli") -> None:
        self.model_id = model_id

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        out = []
        for p, h in zip(premises, hypotheses, strict=True):
            pw = set(_words(p))
            hw = [w for w in _words(h) if len(w) > 2] or _words(h)
            out.append(sum(1 for w in hw if w in pw) / len(hw) if hw else 0.0)
        return out
