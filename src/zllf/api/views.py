"""Convert DB rows + run manifests into API response models."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, col, select

from zllf.api.schemas import (
    AdapterOut,
    DeploymentOut,
    DocumentInfo,
    RunOut,
    RunSummary,
    StageInfo,
    WorkspaceOut,
)
from zllf.core.errors import USER_MESSAGES, ErrorCode
from zllf.core.manifest import STAGE_ORDER, Manifest, RunPaths, read_manifest
from zllf.db.models import Adapter, Deployment, Document, Run, Workspace

LIVE_DEPLOYMENT_STATES = ("starting", "ready")


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def iso_req(ts: float) -> str:
    out = iso(ts)
    assert out is not None
    return out


def error_message(code: str | None) -> str | None:
    """User-facing text for a stage/run error code (or exception class name)."""
    if not code:
        return None
    if code == "cancelled":
        return USER_MESSAGES[ErrorCode.CANCELLED]
    try:
        return USER_MESSAGES[ErrorCode(code)]
    except ValueError:
        return f"{USER_MESSAGES[ErrorCode.INTERNAL]} ({code})"


def load_manifest(run: Run) -> Manifest | None:
    try:
        return read_manifest(RunPaths(Path(run.run_dir)))
    except (OSError, ValueError):
        return None


def current_stage(run: Run, manifest: Manifest | None) -> str | None:
    if manifest is None:
        return run.current_stage
    statuses = {name: manifest.stages[name].status for name in STAGE_ORDER}
    for wanted in ("running", "awaiting_approval", "failed"):
        for name in STAGE_ORDER:
            if statuses[name] == wanted:
                return name
    if run.state == "queued":
        return None
    return run.current_stage


def run_summary(run: Run, manifest: Manifest | None = None) -> RunSummary:
    return RunSummary(
        id=run.id,
        workspace_id=run.workspace_id,
        mode=run.mode,  # type: ignore[arg-type]
        state=run.state,  # type: ignore[arg-type]
        created_at=iso_req(run.created_at),
        current_stage=current_stage(run, manifest),  # type: ignore[arg-type]
    )


def stage_infos(run: Run, manifest: Manifest | None) -> list[StageInfo]:
    out: list[StageInfo] = []
    for name in STAGE_ORDER:
        entry = manifest.stages[name] if manifest else None
        status = entry.status if entry else "pending"
        # A crashed/cancelled worker can leave "running" in the manifest.
        if status == "running" and run.state not in ("running",):
            status = "failed" if run.state in ("failed", "cancelled") else "pending"
        out.append(
            StageInfo(
                name=name,  # type: ignore[arg-type]
                status=status,
                started_at=iso(entry.started_at) if entry else None,
                finished_at=iso(entry.finished_at) if entry else None,
                metrics=dict(entry.metrics) if entry else {},
                error=error_message(entry.error) if entry else None,
            )
        )
    return out


def _tail_events(paths: RunPaths, max_bytes: int = 65536) -> list[dict[str, Any]]:
    try:
        with paths.events.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    events = []
    for line in data.splitlines()[1 if size > max_bytes else 0:]:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def eta_seconds(run: Run, manifest: Manifest | None) -> float | None:
    """Estimate for the current stage from its latest progress event (None when unknown)."""
    if run.state != "running" or manifest is None:
        return None
    stage = current_stage(run, manifest)
    entry = manifest.stages.get(stage or "")
    if not stage or entry is None or entry.started_at is None:
        return None
    for event in reversed(_tail_events(RunPaths(Path(run.run_dir)))):
        if isinstance(event.get("eta_seconds"), int | float):
            return float(event["eta_seconds"])
        if event.get("type") == "progress" and event.get("stage") == stage:
            done, total = event.get("done") or 0, event.get("total") or 0
            if done <= 0 or total <= 0:
                return None
            elapsed = max(0.0, time.time() - entry.started_at)
            return round(elapsed * (total - done) / done, 1)
    return None


def run_adapter_id(db: Session, run_id: str) -> str | None:
    return db.exec(select(Adapter.id).where(Adapter.run_id == run_id)).first()


def run_out(run: Run, db: Session | None = None) -> RunOut:
    manifest = load_manifest(run)
    summary = run_summary(run, manifest)
    return RunOut(
        **summary.model_dump(),
        hardware=run.hardware,
        stages=stage_infos(run, manifest),
        error=run.error_message,
        eta_seconds=eta_seconds(run, manifest),
        adapter_id=run_adapter_id(db, run.id) if db is not None else None,
    )


def latest_run(db: Session, workspace_id: str) -> Run | None:
    return db.exec(
        select(Run).where(Run.workspace_id == workspace_id).order_by(col(Run.created_at).desc())
    ).first()


def deployed_adapter_id(db: Session, workspace_id: str) -> str | None:
    row = db.exec(
        select(Adapter.id)
        .join(Deployment, col(Deployment.adapter_id) == col(Adapter.id))
        .where(Adapter.workspace_id == workspace_id)
        .where(col(Deployment.state).in_(LIVE_DEPLOYMENT_STATES))
        .order_by(col(Deployment.created_at).desc())
    ).first()
    return row


def workspace_out(db: Session, ws: Workspace) -> WorkspaceOut:
    run = latest_run(db, ws.id)
    return WorkspaceOut(
        id=ws.id,
        name=ws.name,
        sensitivity=ws.sensitivity,  # type: ignore[arg-type]
        base_model=ws.base_model,
        created_at=iso_req(ws.created_at),
        latest_run=run_summary(run, load_manifest(run)) if run else None,
        deployed_adapter_id=deployed_adapter_id(db, ws.id),
    )


def document_info(doc: Document) -> DocumentInfo:
    return DocumentInfo(
        id=doc.id,
        name=doc.path,
        mime=doc.mime,
        size_bytes=doc.size_bytes,
        sha256=doc.sha256,
        status=doc.status,  # type: ignore[arg-type]
        pages=doc.pages,
        error=doc.error,
    )


def is_deployed(db: Session, adapter_id: str) -> bool:
    return db.exec(
        select(Deployment.id)
        .where(Deployment.adapter_id == adapter_id)
        .where(col(Deployment.state).in_(LIVE_DEPLOYMENT_STATES))
    ).first() is not None


def adapter_out(db: Session, adapter: Adapter) -> AdapterOut:
    return AdapterOut(
        id=adapter.id,
        run_id=adapter.run_id,
        workspace_id=adapter.workspace_id,
        base_model=adapter.base_model,
        created_at=iso_req(adapter.created_at),
        gates_passed=adapter.gates_passed,
        size_bytes=adapter.size_bytes,
        deployed=is_deployed(db, adapter.id),
    )


def deployment_out(dep: Deployment) -> DeploymentOut:
    return DeploymentOut(
        id=dep.id,
        adapter_id=dep.adapter_id,
        model_name=dep.model_name,
        endpoint=dep.endpoint,
        state=dep.state,  # type: ignore[arg-type]
    )
