"""Kelpod sandbox API for the Hugging Face Space.

Wraps ``kelpod.api.create_app`` with what the web sandbox at kevinantony.de/kelpod needs:
CORS for that site, ``GET /jobs/{id}/meshes.json`` (triangle meshes of the IFC model for the 3D
viewer) and ``GET /jobs/{id}/plan.png`` (a preview of the uploaded plan). Jobs expire after
``KELPOD_JOB_TTL_S`` seconds (15 minutes here); nothing is kept longer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from kelpod.api import create_app
from kelpod.viz import ifc_meshes

ORIGINS = os.environ.get(
    "KELPOD_CORS_ORIGINS",
    "https://kevinantony.de,https://www.kevinantony.de,http://localhost:8765,http://127.0.0.1:8765",
).split(",")
PLAN_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

app = create_app(
    jobs_dir=os.environ.get("KELPOD_JOBS_DIR") or None,
    max_upload_bytes=int(float(os.environ.get("KELPOD_MAX_UPLOAD_MB", "25")) * 1024 * 1024),
    keep_last=int(os.environ.get("KELPOD_KEEP_JOBS", "20")),
    ttl_s=float(os.environ.get("KELPOD_JOB_TTL_S", "900")),
    max_concurrent=int(os.environ.get("KELPOD_MAX_CONCURRENT", "1")),
)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])
store = app.state.store


def _job_dir(job_id: str) -> Path:
    d = store.path(job_id)
    if d is None or not (d / "model.ifc").is_file():
        raise HTTPException(404, "Unknown or expired job.")
    return d


@app.get("/jobs/{job_id}/meshes.json")
def meshes(job_id: str) -> Response:
    d = _job_dir(job_id)
    cached = d / "meshes.json"
    if not cached.is_file():
        data = {"meshes": ifc_meshes(d / "model.ifc")}
        cached.write_text(json.dumps(data, separators=(",", ":")))
    return Response(cached.read_bytes(), media_type="application/json")


@app.get("/jobs/{job_id}/plan.png")
def plan_preview(job_id: str) -> Response:
    d = _job_dir(job_id)
    cached = d / "plan_preview.png"
    if not cached.is_file():
        from kelpod.clean import load

        src = next((p for p in d.iterdir() if p.suffix.lower() in PLAN_SUFFIXES
                    and p.name != cached.name), None)
        if src is None:
            raise HTTPException(404, "The uploaded plan is no longer available.")
        img = load(src, dpi=120)[0]
        h, w = img.shape[:2]
        f = min(1.0, 1600 / max(h, w))
        if f < 1.0:
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cached), img)
    return Response(cached.read_bytes(), media_type="image/png")
