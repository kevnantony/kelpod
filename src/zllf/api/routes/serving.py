"""Adapters registry, deployments and the playground."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy.engine import Engine
from sqlmodel import col, select

from zllf.api.deps import DB, CurrentUser, accessible_workspace_ids, can_access, get_adapter
from zllf.api.errors import ApiError, not_found
from zllf.api.schemas import AdapterOut, ChatRequest, ChatResponse, DeploymentOut
from zllf.api.serving import ServingHub
from zllf.api.views import LIVE_DEPLOYMENT_STATES, adapter_out, deployment_out, load_manifest
from zllf.core.logging import get_logger
from zllf.db.engine import audit, session_scope
from zllf.db.models import Adapter, Deployment, Run, Workspace

router = APIRouter(prefix="/api", tags=["serving"])
log = get_logger("zllf.api.serving")


@router.get("/adapters", response_model=list[AdapterOut])
def list_adapters(db: DB, user: CurrentUser) -> list[AdapterOut]:
    allowed = accessible_workspace_ids(db, user)
    rows = db.exec(select(Adapter).order_by(col(Adapter.created_at).desc())).all()
    return [adapter_out(db, a) for a in rows if allowed is None or a.workspace_id in allowed]


@router.get("/deployments", response_model=list[DeploymentOut])
def list_deployments(db: DB, user: CurrentUser) -> list[DeploymentOut]:
    allowed = accessible_workspace_ids(db, user)
    rows = db.exec(select(Deployment, Adapter).join(Adapter)
                   .order_by(col(Deployment.created_at).desc())).all()
    return [deployment_out(d) for d, a in rows if allowed is None or a.workspace_id in allowed]


def _model_name(db: DB, adapter: Adapter) -> str:
    ws = db.get(Workspace, adapter.workspace_id)
    ids = list(db.exec(select(Adapter.id).where(Adapter.workspace_id == adapter.workspace_id)
                       .order_by(col(Adapter.created_at))).all())
    version = ids.index(adapter.id) + 1 if adapter.id in ids else len(ids)
    try:
        from zllf.serve import deployment_model_name

        return deployment_model_name(ws.name if ws else "ws", version)
    except ImportError:
        import re

        slug = re.sub(r"[^a-z0-9]+", "-", (ws.name if ws else "ws").lower()).strip("-") or "ws"
        return f"ws-{slug}-v{version}"


def _start_deployment(hub: ServingHub, engine: Engine, dep_id: str, base_model: str,
                      name: str, adapter_path: Path) -> None:
    """Background: start the server (slow model load) and register the adapter."""
    try:
        endpoint = hub.deploy(base_model, name, adapter_path)
        state, err = "ready", None
    except Exception as exc:
        endpoint, state, err = "", "failed", type(exc).__name__
        log.error("deploy_failed", deployment_id=dep_id, error=err)
    with session_scope(engine) as db:
        dep = db.get(Deployment, dep_id)
        if dep is not None and dep.state == "starting":
            dep.state, dep.endpoint = state, endpoint or dep.endpoint
            audit(db, "system", f"deployment.{state}", "deployment", dep_id, error=err)


@router.post("/adapters/{adapter_id}/deploy", response_model=DeploymentOut)
def deploy_adapter(adapter_id: str, request: Request, db: DB, user: CurrentUser
                   ) -> DeploymentOut:
    adapter = get_adapter(db, user, adapter_id)
    if not adapter.gates_passed:
        raise ApiError(409, "E_GATES_FAILED",
                       "This model did not pass all quality checks, so it cannot be deployed.")
    if not Path(adapter.path).is_dir():
        raise ApiError(409, "E_ADAPTER_MISSING", "The adapter files are missing on disk.")
    hub: ServingHub = request.app.state.serving
    # one live deployment per workspace: stop the previous one
    live = db.exec(select(Deployment, Adapter).join(Adapter)
                   .where(Adapter.workspace_id == adapter.workspace_id)
                   .where(col(Deployment.state).in_(LIVE_DEPLOYMENT_STATES))).all()
    for dep, other in live:
        if other.id == adapter.id:
            return deployment_out(dep)
        hub.undeploy(other.base_model, dep.model_name)
        dep.state, dep.stopped_at = "stopped", time.time()
        db.add(dep)
    name = _model_name(db, adapter)
    dep = Deployment(adapter_id=adapter.id, model_name=name, state="starting", endpoint="")
    db.add(dep)
    audit(db, user.id, "adapter.deploy", "adapter", adapter.id, deployment_id=dep.id)
    db.commit()
    threading.Thread(
        target=_start_deployment,
        args=(hub, request.app.state.engine, dep.id, adapter.base_model, name, Path(adapter.path)),
        daemon=True, name=f"deploy-{dep.id[:8]}",
    ).start()
    return deployment_out(dep)


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentOut)
def stop_deployment(deployment_id: str, request: Request, db: DB, user: CurrentUser
                    ) -> DeploymentOut:
    dep = db.get(Deployment, deployment_id)
    adapter = db.get(Adapter, dep.adapter_id) if dep else None
    if dep is None or adapter is None or not can_access(db, user, adapter.workspace_id):
        raise not_found("deployment")
    if dep.state in LIVE_DEPLOYMENT_STATES:
        hub: ServingHub = request.app.state.serving
        hub.undeploy(adapter.base_model, dep.model_name)
        dep.state, dep.stopped_at = "stopped", time.time()
        db.add(dep)
        audit(db, user.id, "deployment.stop", "deployment", dep.id)
    return deployment_out(dep)


def _json_valid(output: str, schema: dict[str, Any] | None) -> bool | None:
    if schema is None:
        return None
    try:
        value = json.loads(output)
    except ValueError:
        return False
    try:
        import jsonschema

        jsonschema.validate(value, schema)
    except ImportError:
        return True
    except Exception:
        return False
    return True


@router.post("/playground/chat", response_model=ChatResponse)
def playground_chat(body: ChatRequest, request: Request, db: DB, user: CurrentUser
                    ) -> ChatResponse:
    hub: ServingHub = request.app.state.serving
    settings = request.app.state.settings
    messages = [m.model_dump() for m in body.messages]
    max_tokens = body.max_tokens or 512
    schema: dict[str, Any] | None = None
    if body.adapter_id is None:
        from zllf.core.config import load_config

        base_model = load_config(hardware=settings.hardware,
                                 config_dir=settings.config_dir).models.student
        model_name = None
    else:
        adapter = get_adapter(db, user, body.adapter_id)
        dep = db.exec(select(Deployment).where(Deployment.adapter_id == adapter.id)
                      .where(Deployment.state == "ready")).first()
        if dep is None:
            raise ApiError(409, "E_NOT_DEPLOYED",
                           "This model is not running. Deploy it on the Models page first.")
        base_model, model_name = adapter.base_model, dep.model_name
        run = db.get(Run, adapter.run_id)
        manifest = load_manifest(run) if run else None
        if manifest is not None and manifest.config.get("mode") == "C":
            schema = manifest.config.get("json_schema")
    try:
        output = hub.chat(base_model, model_name, messages, max_tokens, json_schema=schema)
    except ApiError:
        raise
    except Exception as exc:
        log.error("playground_failed", error=type(exc).__name__)
        raise ApiError(503, "E_SERVING_FAILED",
                       "The model could not answer right now. Try again in a moment.") from exc
    audit(db, user.id, "playground.chat", "adapter", body.adapter_id, turns=len(messages))
    return ChatResponse(output=output, citations=[], json_valid=_json_valid(output, schema))
