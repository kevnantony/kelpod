"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlmodel.sql.sqltypes import AutoString

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

S = AutoString


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", S(), primary_key=True),
        sa.Column("username", S(), nullable=False),
        sa.Column("password_hash", S(), nullable=False),
        sa.Column("role", S(), nullable=False),
        sa.Column("disabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_user_username", "user", ["username"], unique=True)

    op.create_table(
        "auth_session",
        sa.Column("id", S(), primary_key=True),
        sa.Column("user_id", S(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("last_seen_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_auth_session_user_id", "auth_session", ["user_id"])

    op.create_table(
        "workspace",
        sa.Column("id", S(), primary_key=True),
        sa.Column("name", S(), nullable=False),
        sa.Column("sensitivity", S(), nullable=False),
        sa.Column("base_model", S(), nullable=False),
        sa.Column("created_by", S(), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_workspace_name", "workspace", ["name"], unique=True)

    op.create_table(
        "workspace_member",
        sa.Column("workspace_id", S(), sa.ForeignKey("workspace.id"), primary_key=True),
        sa.Column("user_id", S(), sa.ForeignKey("user.id"), primary_key=True),
        sa.Column("role", S(), nullable=False),
    )

    op.create_table(
        "document",
        sa.Column("id", S(), primary_key=True),
        sa.Column("workspace_id", S(), sa.ForeignKey("workspace.id"), nullable=False),
        sa.Column("path", S(), nullable=False),
        sa.Column("sha256", S(), nullable=False),
        sa.Column("mime", S(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("status", S(), nullable=False),
        sa.Column("error", S(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("workspace_id", "path"),
    )
    op.create_index("ix_document_workspace_id", "document", ["workspace_id"])
    op.create_index("ix_document_sha256", "document", ["sha256"])

    op.create_table(
        "run",
        sa.Column("id", S(), primary_key=True),
        sa.Column("workspace_id", S(), sa.ForeignKey("workspace.id"), nullable=False),
        sa.Column("mode", S(), nullable=False),
        sa.Column("profile", S(), nullable=False),
        sa.Column("hardware", S(), nullable=False),
        sa.Column("state", S(), nullable=False),
        sa.Column("budget_s", sa.Integer(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("run_dir", S(), nullable=False),
        sa.Column("created_by", S(), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=True),
        sa.Column("finished_at", sa.Float(), nullable=True),
        sa.Column("current_stage", S(), nullable=True),
        sa.Column("error_code", S(), nullable=True),
        sa.Column("error_message", S(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("lease_owner", S(), nullable=True),
        sa.Column("lease_expires_at", sa.Float(), nullable=True),
    )
    op.create_index("ix_run_workspace_id", "run", ["workspace_id"])
    op.create_index("ix_run_state", "run", ["state"])

    op.create_table(
        "stage_run",
        sa.Column("id", S(), primary_key=True),
        sa.Column("run_id", S(), sa.ForeignKey("run.id"), nullable=False),
        sa.Column("stage", S(), nullable=False),
        sa.Column("state", S(), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=True),
        sa.Column("finished_at", sa.Float(), nullable=True),
        sa.Column("output_hash", S(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("error", S(), nullable=True),
        sa.UniqueConstraint("run_id", "stage"),
    )
    op.create_index("ix_stage_run_run_id", "stage_run", ["run_id"])

    op.create_table(
        "adapter",
        sa.Column("id", S(), primary_key=True),
        sa.Column("run_id", S(), sa.ForeignKey("run.id"), nullable=False),
        sa.Column("workspace_id", S(), sa.ForeignKey("workspace.id"), nullable=False),
        sa.Column("base_model", S(), nullable=False),
        sa.Column("path", S(), nullable=False),
        sa.Column("sha256", S(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("gates_passed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_adapter_run_id", "adapter", ["run_id"], unique=True)
    op.create_index("ix_adapter_workspace_id", "adapter", ["workspace_id"])

    op.create_table(
        "deployment",
        sa.Column("id", S(), primary_key=True),
        sa.Column("adapter_id", S(), sa.ForeignKey("adapter.id"), nullable=False),
        sa.Column("model_name", S(), nullable=False),
        sa.Column("endpoint", S(), nullable=False),
        sa.Column("state", S(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("stopped_at", sa.Float(), nullable=True),
    )
    op.create_index("ix_deployment_adapter_id", "deployment", ["adapter_id"])

    op.create_table(
        "gold_item",
        sa.Column("id", S(), primary_key=True),
        sa.Column("workspace_id", S(), sa.ForeignKey("workspace.id"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("question", S(), nullable=False),
        sa.Column("reference_answer", S(), nullable=True),
        sa.Column("reference_json", sa.JSON(), nullable=True),
        sa.Column("answerable", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_gold_item_workspace_id", "gold_item", ["workspace_id"])

    op.create_table(
        "api_key",
        sa.Column("id", S(), primary_key=True),
        sa.Column("user_id", S(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("name", S(), nullable=False),
        sa.Column("key_hash", S(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("revoked_at", sa.Float(), nullable=True),
    )
    op.create_index("ix_api_key_user_id", "api_key", ["user_id"])
    op.create_index("ix_api_key_key_hash", "api_key", ["key_hash"], unique=True)

    op.create_table(
        "audit_event",
        sa.Column("id", S(), primary_key=True),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("actor", S(), nullable=False),
        sa.Column("action", S(), nullable=False),
        sa.Column("object_type", S(), nullable=True),
        sa.Column("object_id", S(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=True),
    )
    op.create_index("ix_audit_event_ts", "audit_event", ["ts"])


def downgrade() -> None:
    for table in (
        "audit_event", "api_key", "gold_item", "deployment", "adapter", "stage_run", "run",
        "document", "workspace_member", "workspace", "auth_session", "user",
    ):
        op.drop_table(table)
