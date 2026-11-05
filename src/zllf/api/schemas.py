"""Request/response models mirroring the TypeScript types in docs/API.md."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["admin", "user"]
Sensitivity = Literal["public", "internal", "confidential", "restricted"]
RunState = Literal["queued", "running", "awaiting_approval", "completed", "failed", "cancelled"]
StageName = Literal["ingest", "sanitize", "synth", "train", "evaluate", "package"]
StageStatus = Literal["pending", "running", "completed", "failed", "awaiting_approval"]


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class UserOut(BaseModel):
    id: str
    username: str
    role: Role


class SetupStatus(BaseModel):
    needs_setup: bool


class RunSummary(BaseModel):
    id: str
    workspace_id: str
    mode: Literal["A", "B", "C"]
    state: RunState
    created_at: str
    current_stage: StageName | None


class StageInfo(BaseModel):
    name: StageName
    status: StageStatus
    started_at: str | None
    finished_at: str | None
    metrics: dict[str, Any]
    error: str | None


class RunOut(RunSummary):
    hardware: str
    stages: list[StageInfo]
    error: str | None
    eta_seconds: float | None
    adapter_id: str | None = None  # extension: registered adapter of this run, if any


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    sensitivity: Sensitivity
    base_model: str = Field(min_length=1, max_length=256)


class WorkspaceOut(BaseModel):
    id: str
    name: str
    sensitivity: Sensitivity
    base_model: str
    created_at: str
    latest_run: RunSummary | None
    deployed_adapter_id: str | None


class DocumentInfo(BaseModel):
    id: str
    name: str
    mime: str
    size_bytes: int
    sha256: str
    status: Literal["uploaded", "parsed", "skipped", "failed"]
    pages: int | None
    error: str | None


class ModelInfo(BaseModel):
    id: str
    role: Literal["student", "teacher", "aux"]
    license: str | None
    installed: bool


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["A", "B", "C"]
    hardware: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    json_schema: dict[str, Any] | None = None
    budget_seconds: int | None = Field(default=None, ge=60, le=7 * 24 * 3600)


class Span(BaseModel):
    start: int
    end: int
    entity_type: str
    action: str


class RedactionSample(BaseModel):
    chunk_id: str
    text_redacted: str
    spans: list[Span]


class RedactionSummary(BaseModel):
    counts_by_type: dict[str, int]
    chunks_dropped: int
    chunks_total: int
    samples: list[RedactionSample]


class TrialOut(BaseModel):
    number: int
    state: str
    objective: float | None
    params: dict[str, Any]
    metrics: dict[str, float]
    seconds: float


class GateOut(BaseModel):
    gate: str
    passed: bool
    value: float
    threshold: float
    details: dict[str, Any]


class ScorecardExample(BaseModel):
    prompt: str
    base: str
    adapter: str
    reference: str | None


class Scorecard(BaseModel):
    run_id: str
    gates: list[GateOut]
    all_passed: bool
    metrics: dict[str, dict[str, float]]
    examples: list[ScorecardExample]
    adapter_id: str | None = None  # extension: registered adapter of this run, if any


class GoldItemIO(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    reference_answer: str | None = None
    reference_json: dict[str, Any] | None = None
    answerable: bool = True


class AdapterOut(BaseModel):
    id: str
    run_id: str
    workspace_id: str
    base_model: str
    created_at: str
    gates_passed: bool
    size_bytes: int
    deployed: bool


class DeploymentOut(BaseModel):
    id: str
    adapter_id: str
    model_name: str
    endpoint: str
    state: Literal["starting", "ready", "stopped", "failed"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32000)


class ChatRequest(BaseModel):
    adapter_id: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class Citation(BaseModel):
    chunk_id: str
    text: str


class ChatResponse(BaseModel):
    output: str
    citations: list[Citation]
    json_valid: bool | None


class Health(BaseModel):
    status: Literal["ok"]
    version: str
