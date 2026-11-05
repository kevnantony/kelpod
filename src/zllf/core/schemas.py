"""Artefact record schemas shared by all stages (the run-directory contract).

Tabular artefacts are Parquet files whose rows are these models (``to_parquet`` /
``read_parquet`` helpers below). Training/eval data are JSONL files of these models.
Changing a field here is a contract change: update every stage and its tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]
Split = Literal["train", "val", "test"]


class Message(BaseModel):
    role: Role
    content: str


# ---------------------------------------------------------------- 01_ingest
class DocumentRecord(BaseModel):
    """documents.parquet"""

    doc_id: str  # first 16 hex chars of sha256
    sha256: str
    path: str  # path relative to the corpus root
    mime: str
    n_pages: int = 0
    n_ocr_pages: int = 0
    n_chars: int = 0
    status: Literal["parsed", "skipped", "failed"]
    error: str | None = None
    language: str | None = None


class Block(BaseModel):
    """A structural block inside a parsed document (pages.parquet holds these rows)."""

    doc_id: str
    block_index: int
    type: Literal["heading", "paragraph", "list", "table", "caption", "other"]
    text: str
    page: int | None = None
    level: int | None = None  # heading level
    table_markdown: str | None = None


class Chunk(BaseModel):
    """chunks.parquet (01_ingest) and chunks_clean.parquet (02_sanitize)"""

    chunk_id: str  # f"{doc_id}-{index:05d}"
    doc_id: str
    index: int
    text: str
    n_tokens: int
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    is_table: bool = False
    duplicate_of: str | None = None  # set when removed as near-duplicate


# ---------------------------------------------------------------- 02_sanitize
class Redaction(BaseModel):
    """redactions.parquet. Never contains the original value (NFR-SEC-06)."""

    chunk_id: str
    start: int
    end: int
    entity_type: str
    action: Literal["keep", "mask", "drop_chunk"]
    score: float
    placeholder: str | None = None


# ---------------------------------------------------------------- 03_synth
TaskType = Literal["qa", "unanswerable", "summarization", "definition", "drafting",
                   "extraction", "dialogue", "canary"]


class SynthExample(BaseModel):
    """train.jsonl / val.jsonl / test.jsonl / unanswerable_*.jsonl rows.

    ``messages`` is the chat sample (system?, user, assistant). The trainer uses
    ``messages`` only; everything else is metadata for evaluation and lineage.
    """

    example_id: str
    task: TaskType
    split: Split
    messages: list[Message]
    doc_id: str | None = None
    chunk_id: str | None = None
    quote: str | None = None
    reference_answer: str | None = None
    reference_json: dict[str, Any] | None = None
    answerable: bool = True
    entailment: float | None = None
    judge_ok: bool | None = None


class Canary(BaseModel):
    canary_id: str
    prefix: str  # context shown when probing
    secret: str  # random string that must never be reproduced


# ---------------------------------------------------------------- 04_train
class TrialParams(BaseModel):
    rank: int
    alpha: float
    learning_rate: float
    target_modules: Literal["attention", "mlp", "all"]
    dropout: float
    weight_decay: float
    replay_ratio: float
    epochs: float = 1.0
    data_fraction: float = 1.0
    batch_size: int = 4
    grad_accumulation: int = 4


class TrialResult(BaseModel):
    number: int
    params: TrialParams
    state: Literal["complete", "pruned", "failed", "failed_oom", "running"]
    objective: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    adapter_path: str | None = None
    seconds: float = 0.0


class HardwareProfile(BaseModel):
    platform: str  # e.g. "darwin-arm64", "linux-cuda"
    accelerator: Literal["mps", "cuda", "cpu"]
    device_name: str
    n_devices: int = 1
    memory_total_gb: float
    memory_available_gb: float
    cpu_cores: int
    disk_free_gb: float
    bf16: bool = False
    fp8: bool = False
    backend: str  # llm backend chosen for this machine
    quantization: Literal["4bit", "8bit", "none"] = "4bit"


# ---------------------------------------------------------------- 05_evaluate
Variant = Literal["base", "base_rag", "adapter", "adapter_rag"]


class EvalItem(BaseModel):
    item_id: str
    suite: str  # "domain_test", "gold", "anchor/<name>", "unanswerable", "canary"
    prompt: list[Message]
    reference: str | None = None
    reference_json: dict[str, Any] | None = None
    answerable: bool = True
    choices: list[str] | None = None  # multiple-choice anchor items


class EvalPrediction(BaseModel):
    item_id: str
    suite: str
    variant: Variant
    output: str
    scores: dict[str, float] = Field(default_factory=dict)


class GateResult(BaseModel):
    gate: Literal["G1_domain", "G2_forgetting", "G3_hallucination", "G4_memorization",
                  "G5_schema"]
    passed: bool
    value: float
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def to_parquet(rows: Iterable[BaseModel], path: Path) -> int:
    records = [r.model_dump(mode="json") for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(path, index=False)
    return len(records)


def read_parquet[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    df = pd.read_parquet(path)
    out: list[T] = []
    for rec in df.to_dict(orient="records"):
        clean = {k: _from_arrow(v) for k, v in rec.items()}
        out.append(model.model_validate(clean))
    return out


def _from_arrow(value: Any) -> Any:
    # numpy arrays / NaN from parquet -> python lists / None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def write_jsonl(rows: Iterable[BaseModel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def messages(*pairs: tuple[Role, str]) -> list[Message]:
    return [Message(role=r, content=c) for r, c in pairs]


__all__: Sequence[str] = [
    "Block", "Canary", "Chunk", "DocumentRecord", "EvalItem", "EvalPrediction", "GateResult",
    "HardwareProfile", "Message", "Redaction", "SynthExample", "TrialParams", "TrialResult",
    "messages", "read_jsonl", "read_parquet", "to_parquet", "write_jsonl",
]
