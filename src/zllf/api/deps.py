"""FastAPI dependencies: settings, DB session, current user, workspace/run access checks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from zllf.api.errors import ApiError, not_found
from zllf.api.security import SESSION_COOKIE, lookup_session
from zllf.api.settings import Settings
from zllf.db.engine import session_scope
from zllf.db.models import Adapter, Run, User, Workspace, WorkspaceMember


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine_dep(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def get_db(request: Request) -> Iterator[Session]:
    with session_scope(request.app.state.engine) as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DB = Annotated[Session, Depends(get_db)]


def current_user(request: Request, db: DB) -> User:
    user = lookup_session(db, request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise ApiError(401, "E_AUTH", "Please sign in to continue.")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_admin(user: CurrentUser) -> User:
    if user.role != "admin":
        raise ApiError(403, "E_FORBIDDEN", "Only an administrator can do this.")
    return user


AdminUser = Annotated[User, Depends(require_admin)]


def can_access(db: Session, user: User, workspace_id: str) -> bool:
    if user.role == "admin":
        return True
    member = db.get(WorkspaceMember, (workspace_id, user.id))
    return member is not None


def accessible_workspace_ids(db: Session, user: User) -> set[str] | None:
    """None means 'all' (admin)."""
    if user.role == "admin":
        return None
    rows = db.exec(select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id))
    return set(rows.all())


def get_workspace(db: Session, user: User, workspace_id: str) -> Workspace:
    ws = db.get(Workspace, workspace_id)
    if ws is None or not can_access(db, user, workspace_id):
        raise not_found("workspace")  # 404, not 403: do not reveal other workspaces
    return ws


def get_run(db: Session, user: User, run_id: str) -> Run:
    run = db.get(Run, run_id)
    if run is None or not can_access(db, user, run.workspace_id):
        raise not_found("run")
    return run


def get_adapter(db: Session, user: User, adapter_id: str) -> Adapter:
    adapter = db.get(Adapter, adapter_id)
    if adapter is None or not can_access(db, user, adapter.workspace_id):
        raise not_found("model")
    return adapter
