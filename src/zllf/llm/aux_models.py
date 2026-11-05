"""Auxiliary encoder models: NLI cross-encoder and sentence embedder (local cache only).

Both resolve the model from the local Hugging Face cache (``local_files_only``) and raise
``ZLLFError(MODEL_MISSING)`` if it is absent. Torch device: ``mps`` when available, else CPU.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.llm.base import Embedder, NLIScorer

log = logging.getLogger(__name__)


def resolve_encoder_path(model_id: str) -> Path:
    """Local snapshot directory of an encoder model (never downloads)."""
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            path = Path(snapshot_download(model_id, local_files_only=True))
        except (LocalEntryNotFoundError, FileNotFoundError, ValueError, OSError) as exc:
            raise ZLLFError(
                ErrorCode.MODEL_MISSING, f"model {model_id!r} is not in the local cache"
            ) from exc
    if not (path / "config.json").is_file():
        raise ZLLFError(ErrorCode.MODEL_MISSING, f"{model_id!r}: config.json missing")
    weights = [*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")]
    if not weights:
        raise ZLLFError(ErrorCode.MODEL_MISSING, f"{model_id!r}: no weights found")
    return path


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def entailment_index(id2label: dict[Any, str]) -> int:
    """Index of the entailment class in a classifier's ``id2label`` mapping."""
    for idx, label in id2label.items():
        if "entail" in str(label).lower() and "not" not in str(label).lower():
            return int(idx)
    raise ZLLFError(ErrorCode.MODEL_MISSING, f"no entailment label in {dict(id2label)}")


class CrossEncoderNLI(NLIScorer):
    """NLI cross-encoder (e.g. ``cross-encoder/nli-deberta-v3-xsmall``) → P(entailment)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        path = resolve_encoder_path(model_id)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        except OSError as exc:
            raise ZLLFError(ErrorCode.MODEL_MISSING, f"cannot load {model_id!r}") from exc
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.entail_idx = entailment_index(model.config.id2label)
        self._torch = torch

    def entailment(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        torch = self._torch
        out: list[float] = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                list(premises[i : i + self.batch_size]),
                list(hypotheses[i : i + self.batch_size]),
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self.entail_idx]
            out.extend(float(p) for p in probs.cpu().tolist())
        return out

    def close(self) -> None:
        self.model = None
        _empty_torch_cache()


class SentenceEmbedder(Embedder):
    """sentence-transformers embedder → L2-normalised float32 matrix."""

    def __init__(self, model_id: str, *, device: str | None = None, batch_size: int = 64) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        path = resolve_encoder_path(model_id)
        from sentence_transformers import SentenceTransformer

        self.device = pick_device(device)
        try:
            self.model: Any = SentenceTransformer(
                str(path), device=self.device, local_files_only=True
            )
        except OSError as exc:
            raise ZLLFError(ErrorCode.MODEL_MISSING, f"cannot load {model_id!r}") from exc
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        if get_dim is None:  # sentence-transformers < 6
            get_dim = self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim() or 0)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def close(self) -> None:
        self.model = None
        _empty_torch_cache()


def _empty_torch_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover - best effort
        pass
