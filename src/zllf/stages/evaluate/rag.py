"""RAG baseline (FR-13, P2.6.2): dense retrieval over sanitised chunks + fixed prompt.

Embeddings come from ``ctx.embedder()`` (L2-normalised), so inner product == cosine.
A FAISS ``IndexFlatIP`` is used when ``faiss`` is importable and the corpus is not tiny;
otherwise an exact numpy matrix product (identical results).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from zllf.core.schemas import Chunk, Message
from zllf.llm.base import Embedder

FAISS_MIN_ROWS = 256
EMBED_BATCH = 64

RAG_INSTRUCTION = (
    "Answer the question using only the numbered context passages below. Cite the passages "
    "you use as [n]. If the passages do not contain the answer, say that the information is "
    "not available in the documents."
)


class RagIndex:
    """Exact inner-product index over chunk embeddings."""

    def __init__(self, chunk_ids: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        if len(chunk_ids) != len(texts) or len(texts) != embeddings.shape[0]:
            raise ValueError("chunk_ids, texts and embeddings must align")
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.backend = "numpy"
        self._faiss_index = None
        if len(texts) >= FAISS_MIN_ROWS:
            try:
                import faiss

                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                index.add(self.embeddings)
                self._faiss_index = index
                self.backend = "faiss"
            except ImportError:  # pragma: no cover - environment dependent
                pass

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder,
              batch_size: int = EMBED_BATCH) -> RagIndex:
        usable = [c for c in chunks if c.duplicate_of is None and c.text.strip()]
        texts = [c.text for c in usable]
        parts = [embedder.encode(texts[i : i + batch_size])
                 for i in range(0, len(texts), batch_size)]
        emb = np.concatenate(parts, axis=0) if parts else np.zeros((0, 1), dtype=np.float32)
        return cls([c.chunk_id for c in usable], texts, emb)

    def __len__(self) -> int:
        return len(self.texts)

    def search(self, query_embeddings: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        """Top-k (row index, score) per query, best first; ties broken by row index."""
        if len(self) == 0 or k <= 0:
            return [[] for _ in range(query_embeddings.shape[0])]
        k = min(k, len(self))
        q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, k)
            return [[(int(i), float(s)) for i, s in zip(ri, rs, strict=True) if i >= 0]
                    for ri, rs in zip(idx, scores, strict=True)]
        sims = q @ self.embeddings.T
        out: list[list[tuple[int, float]]] = []
        for row in sims:
            order = np.lexsort((np.arange(row.size), -row))[:k]
            out.append([(int(i), float(row[i])) for i in order])
        return out

    def save(self, directory: Path) -> None:
        """Persist embeddings + chunk ids (no text) for later inspection/serving."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunk_ids.json").write_text(json.dumps(self.chunk_ids), encoding="utf-8")
        (directory / "meta.json").write_text(
            json.dumps({"n": len(self), "dim": int(self.embeddings.shape[1]) if len(self) else 0,
                        "metric": "inner_product", "backend": self.backend}),
            encoding="utf-8",
        )


def passage_char_budget(max_seq_len: int, k: int) -> int:
    """Characters per passage so k passages use ~60 % of the context (~4 chars/token)."""
    return max(200, int(max_seq_len * 0.6 * 4 / max(1, k)))


def rag_prompt(prompt: Sequence[Message], passages: Sequence[str], char_budget: int,
               ) -> list[Message]:
    """Replace the last user turn with the fixed RAG prompt (numbered context passages)."""
    msgs = list(prompt)
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"), default=None)
    if last_user is None:
        return msgs
    ctx_lines = []
    for n, text in enumerate(passages, start=1):
        clipped = text if len(text) <= char_budget else text[:char_budget] + " ..."
        ctx_lines.append(f"[{n}] {clipped}")
    content = (
        f"{RAG_INSTRUCTION}\n\nContext:\n" + "\n\n".join(ctx_lines)
        + f"\n\nQuestion: {msgs[last_user].content}"
    )
    msgs[last_user] = Message(role="user", content=content)
    return msgs
