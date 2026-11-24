"""HTTP API (FastAPI): ``POST /convert`` a plan, get the IFC model and a report.

Needs the optional extra: ``uv sync --extra api``. Run with ``uv run --extra api python -m
kelpod.api --port 8000`` or ``uvicorn kelpod.api:app``. The module-level ``app`` reads
``KELPOD_JOBS_DIR`` (default: a temp directory deleted on shutdown), ``KELPOD_MAX_UPLOAD_MB``
(default 50), ``KELPOD_KEEP_JOBS`` (default 20) and ``KELPOD_JOB_TTL_S`` (default 3600).
"""

from __future__ import annotations

import os

from kelpod.api.app import create_app
from kelpod.api.jobs import JobStore

__all__ = ["JobStore", "app", "create_app", "app_from_env"]


def app_from_env():
    return create_app(
        jobs_dir=os.environ.get("KELPOD_JOBS_DIR") or None,
        max_upload_bytes=int(float(os.environ.get("KELPOD_MAX_UPLOAD_MB", "50")) * 1024 * 1024),
        keep_last=int(os.environ.get("KELPOD_KEEP_JOBS", "20")),
        ttl_s=float(os.environ.get("KELPOD_JOB_TTL_S", "3600")),
    )


app = app_from_env()
