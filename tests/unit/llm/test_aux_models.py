"""NLI and embedder tests; skipped when the encoder models are not cached."""

from __future__ import annotations

import numpy as np
import pytest
from zllf_llm_helpers import cached

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.llm.aux_models import entailment_index, resolve_encoder_path

NLI = "cross-encoder/nli-deberta-v3-xsmall"
EMB = "sentence-transformers/all-MiniLM-L6-v2"


def test_entailment_index() -> None:
    assert entailment_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 1
    assert entailment_index({"0": "ENTAILMENT", "1": "NOT_ENTAILMENT"}) == 0
    with pytest.raises(ZLLFError):
        entailment_index({0: "a", 1: "b"})


def test_missing_encoder_raises() -> None:
    with pytest.raises(ZLLFError) as ei:
        resolve_encoder_path("nobody/not-cached-encoder-xyz")
    assert ei.value.code == ErrorCode.MODEL_MISSING


@pytest.mark.skipif(not cached(NLI), reason="NLI model not cached")
def test_nli_entailment() -> None:
    from zllf.llm.aux_models import CrossEncoderNLI

    nli = CrossEncoderNLI(NLI, batch_size=2)
    try:
        p = "The invoice of 3 March was paid by bank transfer."
        scores = nli.entailment(
            [p, p, p],
            [
                "The invoice was paid.",
                "The invoice was never paid.",
                "The weather is nice.",
            ],
        )
    finally:
        nli.close()
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > 0.5 > scores[1]
    assert scores[0] > scores[2]


@pytest.mark.skipif(not cached(EMB), reason="embedder not cached")
def test_embedder() -> None:
    from zllf.llm.aux_models import SentenceEmbedder

    emb = SentenceEmbedder(EMB, batch_size=2)
    try:
        m = emb.encode(["How do I reset my password?", "Password reset steps", "Banana bread"])
        empty = emb.encode([])
    finally:
        emb.close()
    assert m.dtype == np.float32 and m.shape[0] == 3
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-4)
    assert m[0] @ m[1] > m[0] @ m[2]
    assert empty.shape == (0, m.shape[1])
