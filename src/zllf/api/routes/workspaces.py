"""Workspaces, gold set and the model catalogue."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Response
from sqlalchemy import func
from sqlmodel import col, delete, select

from zllf.api.catalog import list_models
from zllf.api.deps import DB, CurrentUser, SettingsDep, accessible_workspace_ids, get_workspace
from zllf.api.errors import ApiError
from zllf.api.schemas import GoldItemIO, ModelInfo, WorkspaceCreate, WorkspaceOut
from zllf.api.views import LIVE_DEPLOYMENT_STATES, workspace_out
from zllf.db.engine import audit
from zllf.db.models import (
    Adapter,
    Deployment,
    Document,
    GoldItem,
    Run,
    StageRun,
    Workspace,
    WorkspaceMember,
)

router = APIRouter(prefix="/api", tags=["workspaces"])


@router.get("/models", response_model=list[ModelInfo])
def models(_user: CurrentUser, settings: SettingsDep) -> list[ModelInfo]:
    return list_models(settings)


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(db: DB, user: CurrentUser) -> list[WorkspaceOut]:
    allowed = accessible_workspace_ids(db, user)
    rows = db.exec(select(Workspace).order_by(col(Workspace.created_at).desc())).all()
    return [workspace_out(db, ws) for ws in rows if allowed is None or ws.id in allowed]


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace(
    body: WorkspaceCreate, db: DB, user: CurrentUser, settings: SettingsDep
) -> WorkspaceOut:
    name = body.name.strip()
    if not name:
        raise ApiError(422, "E_VALIDATION", "Give the workspace a name.")
    model = next((m for m in list_models(settings)
                  if m.id == body.base_model and m.role == "student"), None)
    if model is None:
        raise ApiError(422, "E_VALIDATION", "Choose one of the installed base models.")
    if not model.installed:
        raise ApiError(
            422, "E_MODEL_NOT_INSTALLED",
            f"{model.id} is not installed on this appliance. Pick another model or ask an "
            "administrator to import it.",
        )
    exists = db.exec(select(Workspace.id).where(func.lower(Workspace.name) == name.lower()))
    if exists.first():
        raise ApiError(409, "E_CONFLICT",
                       "A workspace with this name already exists. Choose another name.")
    ws = Workspace(name=name, sensitivity=body.sensitivity, base_model=model.id,
                   created_by=user.id)
    db.add(ws)
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="owner"))
    settings.corpus_dir(ws.id).mkdir(parents=True, exist_ok=True)
    audit(db, user.id, "workspace.create", "workspace", ws.id, sensitivity=ws.sensitivity)
    return workspace_out(db, ws)


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceOut)
def read_workspace(workspace_id: str, db: DB, user: CurrentUser) -> WorkspaceOut:
    return workspace_out(db, get_workspace(db, user, workspace_id))


@router.delete("/workspaces/{workspace_id}", status_code=204)
def delete_workspace(
    workspace_id: str, db: DB, user: CurrentUser, settings: SettingsDep
) -> Response:
    """Delete a workspace and all its data (corpus, runs, adapters). Irreversible."""
    ws = get_workspace(db, user, workspace_id)
    if user.role != "admin" and ws.created_by != user.id:
        raise ApiError(403, "E_FORBIDDEN", "Only the owner or an administrator can delete this.")
    runs = db.exec(select(Run).where(Run.workspace_id == ws.id)).all()
    if any(r.state in ("queued", "running") for r in runs):
        raise ApiError(409, "E_RUN_ACTIVE",
                       "A run is in progress for this workspace. Cancel it first.")
    adapter_ids = list(db.exec(select(Adapter.id).where(Adapter.workspace_id == ws.id)).all())
    if adapter_ids and db.exec(
        select(Deployment.id).where(col(Deployment.adapter_id).in_(adapter_ids))
        .where(col(Deployment.state).in_(LIVE_DEPLOYMENT_STATES))
    ).first():
        raise ApiError(409, "E_DEPLOYED", "A model of this workspace is deployed. Stop it first.")
    run_ids = [r.id for r in runs]
    if adapter_ids:
        db.exec(delete(Deployment).where(col(Deployment.adapter_id).in_(adapter_ids)))
        db.exec(delete(Adapter).where(col(Adapter.id).in_(adapter_ids)))
    if run_ids:
        db.exec(delete(StageRun).where(col(StageRun.run_id).in_(run_ids)))
        db.exec(delete(Run).where(col(Run.id).in_(run_ids)))
    db.exec(delete(Document).where(col(Document.workspace_id) == ws.id))
    db.exec(delete(GoldItem).where(col(GoldItem.workspace_id) == ws.id))
    db.exec(delete(WorkspaceMember).where(col(WorkspaceMember.workspace_id) == ws.id))
    db.delete(ws)
    audit(db, user.id, "workspace.delete", "workspace", workspace_id, runs=len(run_ids))
    db.flush()
    for run in runs:
        shutil.rmtree(Path(run.run_dir), ignore_errors=True)
    shutil.rmtree(settings.workspace_dir(workspace_id), ignore_errors=True)
    return Response(status_code=204)


# ------------------------------------------------------------------ gold set
def _gold_rows(db: DB, workspace_id: str) -> list[GoldItem]:
    return list(db.exec(select(GoldItem).where(GoldItem.workspace_id == workspace_id)
                        .order_by(col(GoldItem.position))).all())


def _gold_io(item: GoldItem) -> GoldItemIO:
    return GoldItemIO(question=item.question, reference_answer=item.reference_answer,
                      reference_json=item.reference_json, answerable=item.answerable)


@router.get("/workspaces/{workspace_id}/gold", response_model=list[GoldItemIO])
def get_gold(workspace_id: str, db: DB, user: CurrentUser) -> list[GoldItemIO]:
    get_workspace(db, user, workspace_id)
    return [_gold_io(g) for g in _gold_rows(db, workspace_id)]


@router.post("/workspaces/{workspace_id}/gold", response_model=list[GoldItemIO])
def replace_gold(
    workspace_id: str, items: list[GoldItemIO], db: DB, user: CurrentUser, settings: SettingsDep
) -> list[GoldItemIO]:
    """Replace the gold set; also written to ``workspaces/<id>/gold.jsonl`` for the evaluator."""
    get_workspace(db, user, workspace_id)
    if len(items) > 10000:
        raise ApiError(422, "E_VALIDATION", "The question set is too large (max 10 000).")
    if any(not i.question.strip() for i in items):
        raise ApiError(422, "E_VALIDATION", "Every entry needs a non-empty question.")
    db.exec(delete(GoldItem).where(col(GoldItem.workspace_id) == workspace_id))
    for pos, item in enumerate(items):
        db.add(GoldItem(workspace_id=workspace_id, position=pos, question=item.question.strip(),
                        reference_answer=item.reference_answer,
                        reference_json=item.reference_json, answerable=item.answerable))
    path = settings.gold_path(workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(i.model_dump()) + "\n" for i in items), encoding="utf-8")
    tmp.replace(path)
    audit(db, user.id, "gold.replace", "workspace", workspace_id, items=len(items))
    db.flush()
    return [_gold_io(g) for g in _gold_rows(db, workspace_id)]
