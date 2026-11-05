"""SQLModel tables (DEVELOPMENT_PLAN §1.5). Timestamps are UNIX epoch floats (UTC).

The run manifest (``runs/<id>/manifest.json``) is the source of truth for stage state;
``stage_run`` mirrors it for queries. Keep this file and the Alembic migrations in sync
(``tests/unit/api/test_db.py`` checks it).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


class User(SQLModel, table=True):
    __tablename__ = "user"

    id: str = Field(default_factory=new_id, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "user"  # admin | user
    disabled: bool = False
    created_at: float = Field(default_factory=now)


class AuthSession(SQLModel, table=True):
    """Server-side session. ``id`` is the sha256 of the cookie token (token never stored)."""

    __tablename__ = "auth_session"

    id: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    created_at: float = Field(default_factory=now)
    expires_at: float
    last_seen_at: float = Field(default_factory=now)


class Workspace(SQLModel, table=True):
    __tablename__ = "workspace"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str = Field(index=True, unique=True)
    sensitivity: str = "internal"
    base_model: str
    created_by: str | None = Field(default=None, foreign_key="user.id")
    created_at: float = Field(default_factory=now)


class WorkspaceMember(SQLModel, table=True):
    """Workspace-level access for non-admin users (admins see every workspace)."""

    __tablename__ = "workspace_member"

    workspace_id: str = Field(foreign_key="workspace.id", primary_key=True)
    user_id: str = Field(foreign_key="user.id", primary_key=True)
    role: str = "owner"  # owner | member


class Document(SQLModel, table=True):
    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("workspace_id", "path"),)

    id: str = Field(default_factory=new_id, primary_key=True)
    workspace_id: str = Field(foreign_key="workspace.id", index=True)
    path: str  # relative to workspaces/<id>/corpus
    sha256: str = Field(index=True)
    mime: str = "application/octet-stream"
    size_bytes: int = 0
    pages: int | None = None
    status: str = "uploaded"  # uploaded | parsed | skipped | failed
    error: str | None = None
    created_at: float = Field(default_factory=now)


class Run(SQLModel, table=True):
    __tablename__ = "run"

    id: str = Field(primary_key=True)  # == run directory name
    workspace_id: str = Field(foreign_key="workspace.id", index=True)
    mode: str  # A | B | C
    profile: str
    hardware: str
    state: str = Field(default="queued", index=True)
    budget_s: int | None = None
    seed: int | None = None
    run_dir: str
    created_by: str | None = Field(default=None, foreign_key="user.id")
    created_at: float = Field(default_factory=now)
    started_at: float | None = None
    finished_at: float | None = None
    current_stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    lease_owner: str | None = None
    lease_expires_at: float | None = None


class StageRun(SQLModel, table=True):
    """Mirror of the manifest's per-stage entries (manifest stays the source of truth)."""

    __tablename__ = "stage_run"
    __table_args__ = (UniqueConstraint("run_id", "stage"),)

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    stage: str
    state: str = "pending"
    started_at: float | None = None
    finished_at: float | None = None
    output_hash: str | None = None
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None


class Adapter(SQLModel, table=True):
    __tablename__ = "adapter"

    id: str = Field(default_factory=new_id, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True, unique=True)
    workspace_id: str = Field(foreign_key="workspace.id", index=True)
    base_model: str
    path: str
    sha256: str
    size_bytes: int = 0
    gates_passed: bool = False
    created_at: float = Field(default_factory=now)


class Deployment(SQLModel, table=True):
    __tablename__ = "deployment"

    id: str = Field(default_factory=new_id, primary_key=True)
    adapter_id: str = Field(foreign_key="adapter.id", index=True)
    model_name: str
    endpoint: str = ""
    state: str = "starting"  # starting | ready | stopped | failed
    created_at: float = Field(default_factory=now)
    stopped_at: float | None = None


class GoldItem(SQLModel, table=True):
    __tablename__ = "gold_item"

    id: str = Field(default_factory=new_id, primary_key=True)
    workspace_id: str = Field(foreign_key="workspace.id", index=True)
    position: int = 0
    question: str
    reference_answer: str | None = None
    reference_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    answerable: bool = True


class ApiKey(SQLModel, table=True):
    __tablename__ = "api_key"

    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    name: str = ""
    key_hash: str = Field(index=True, unique=True)
    created_at: float = Field(default_factory=now)
    revoked_at: float | None = None


class AuditEvent(SQLModel, table=True):
    __tablename__ = "audit_event"

    id: str = Field(default_factory=new_id, primary_key=True)
    ts: float = Field(default_factory=now, index=True)
    actor: str  # user id, "worker" or "system"
    action: str
    object_type: str | None = None
    object_id: str | None = None
    details_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


ACTIVE_RUN_STATES = ("queued", "running", "awaiting_approval")
TERMINAL_RUN_STATES = ("completed", "failed", "cancelled")
