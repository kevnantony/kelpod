"""Construct model backends from config. Fakes for tests/CI; MLX on Apple Silicon."""

from __future__ import annotations

from zllf.core.config import RunConfig
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.llm.base import Embedder, LLMBackend, NLIScorer
from zllf.llm.fake import FakeEmbedder, FakeLLM, FakeNLI


def model_id_for(config: RunConfig, role: str) -> str:
    m = config.models
    if role == "teacher":
        return m.teacher
    if role == "student":
        return m.student
    if role == "judge":
        return m.judge or m.teacher
    raise ValueError(f"unknown model role {role!r}")


def get_llm(config: RunConfig, role: str) -> LLMBackend:
    model_id = model_id_for(config, role)
    backend = config.models.backend
    if backend == "fake":
        return FakeLLM(model_id)
    if backend == "mlx":
        from zllf.llm.mlx_backend import MLXBackend

        return MLXBackend(model_id, max_seq_len=config.models.max_seq_len)
    raise ZLLFError(ErrorCode.MODEL_MISSING, f"backend {backend!r} is not available yet")


def get_nli(config: RunConfig) -> NLIScorer:
    name = config.models.nli
    if name.startswith("fake"):
        return FakeNLI(name)
    from zllf.llm.aux_models import CrossEncoderNLI

    return CrossEncoderNLI(name)


def get_embedder(config: RunConfig) -> Embedder:
    name = config.models.embedder
    if name.startswith("fake"):
        return FakeEmbedder(name)
    from zllf.llm.aux_models import SentenceEmbedder

    return SentenceEmbedder(name)
