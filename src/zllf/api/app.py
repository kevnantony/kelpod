"""FastAPI application factory: routers, errors, CSRF, DB init and the SPA."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlmodel import col, select

from zllf import __version__
from zllf.api.errors import error_body, install_error_handlers
from zllf.api.routes import auth, documents, runs, serving, system, workspaces
from zllf.api.security import CSRFMiddleware, LoginRateLimiter
from zllf.api.serving import ServingHub
from zllf.api.settings import Settings, get_settings
from zllf.db.engine import audit, init_db, session_scope
from zllf.db.models import Deployment


def _stop_stale_deployments(app: FastAPI) -> None:
    """Chat services are in-memory: after a restart nothing is being served."""
    with session_scope(app.state.engine) as db:
        for dep in db.exec(select(Deployment).where(
                col(Deployment.state).in_(("starting", "ready")))).all():
            dep.state, dep.stopped_at = "stopped", time.time()
            audit(db, "system", "deployment.stopped_on_restart", "deployment", dep.id)


def create_app(settings: Settings | None = None, *, start_worker: bool = False) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    engine = init_db(settings.db_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _stop_stale_deployments(app)
        worker = None
        if start_worker:
            from zllf.worker.loop import WorkerThread

            worker = WorkerThread(settings, engine)
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
            app.state.serving.shutdown()

    app = FastAPI(title="ZLLF", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.engine = engine
    app.state.login_limiter = LoginRateLimiter(settings.login_max_failures,
                                               settings.login_window_seconds)
    app.state.serving = ServingHub(settings.root / "logs")
    app.add_middleware(CSRFMiddleware)
    install_error_handlers(app)
    for module in (auth, workspaces, documents, runs, serving, system):
        app.include_router(module.router)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    _mount_spa(app, settings.static_dir)
    return app


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve ``web/dist`` with client-side routing fallback to index.html."""
    root = static_dir.resolve()

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   include_in_schema=False)
    def api_not_found(rest: str) -> JSONResponse:
        return JSONResponse(error_body("E_NOT_FOUND", "No such API endpoint."), status_code=404)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        index = root / "index.html"
        if path:
            candidate = (root / path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return JSONResponse(error_body("E_NOT_FOUND", "The web UI is not built (web/dist)."),
                            status_code=404)


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory zllf.api.app:app_factory``."""
    return create_app()
