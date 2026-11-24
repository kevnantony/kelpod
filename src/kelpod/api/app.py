"""FastAPI app: upload a plan, get a closed IFC model and a plain-language report.

Design (same shape as e.g. Gotenberg, unstructured-api and docling-serve): one synchronous
multipart ``POST /convert`` that runs the whole pipeline (target: under 3 minutes per plan) and
returns JSON, plus job resources to fetch the result again and download the file. The pipeline
runs in a worker thread with a concurrency limit, so the event loop stays free.
"""

from __future__ import annotations

import importlib.util
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import anyio.to_thread
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from kelpod import __version__
from kelpod.api.jobs import JobStore
from kelpod.contracts import CONTRACT_VERSION, PipelineResult

ALLOWED_SUFFIXES = {
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
}
_MAGIC = {
    "pdf": (b"%PDF-",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),
}
NOTICE = (
    "Automatically generated model, not a certified measurement. A qualified expert must check "
    "it before use."
)
_CHUNK = 1 << 20


def _ocr_available() -> bool:
    """True if the default OCR engine (RapidOCR) can be imported (the model is not loaded)."""
    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or "plan").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:64]
    return stem or "plan"


def _summary(res: PipelineResult) -> str:
    rep = res.graph.report
    if rep is None:
        return "No consistency report was produced."
    head = "Closed model" if rep.is_closed else "Model is NOT fully closed"
    ends = len(rep.dangling_nodes)
    return (
        f"{head}: {rep.n_walls} walls, {rep.n_rooms} rooms, {rep.n_components} connected "
        f"part(s), {ends} loose wall end(s); {rep.node_closure:.0%} of wall nodes and "
        f"{rep.room_closure:.0%} of rooms closed."
    )


def _issues(res: PipelineResult) -> list[str]:
    out: list[str] = []
    rep = res.graph.report
    if rep is not None:
        out += rep.issues
    if res.export is not None:
        if not res.export.reopened_ok:
            out.append("The IFC file could not be re-opened after writing; do not use it.")
        out += [f"Export: {w}" for w in res.export.warnings]
    if res.graph.scale_source != "dimensions":
        out.append(
            f"Scale taken from {res.graph.scale_source.replace('_', ' ')}, not from dimension "
            "chains; check one known length."
        )
    return out


def create_app(
    *,
    jobs_dir: str | Path | None = None,
    max_upload_bytes: int = 50 * 1024 * 1024,
    keep_last: int = 20,
    ttl_s: float = 3600.0,
    max_concurrent: int = 2,
) -> FastAPI:
    """Build the API. ``jobs_dir=None`` uses a fresh temp directory that is deleted on shutdown."""
    from kelpod import ocr, segment
    from kelpod.pipeline import convert

    store = JobStore(jobs_dir, keep_last=keep_last, ttl_s=ttl_s)
    limiter = anyio.CapacityLimiter(max_concurrent)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(
        title="Kelpod",
        version=__version__,
        summary="Old German floor-plan scans to closed IFC models.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.max_upload_bytes = max_upload_bytes

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "contract_version": CONTRACT_VERSION,
            "ocr_available": _ocr_available(),
            "ocr_backends": ocr.available_engines(),
            "segment_backends": segment.available(),
            "max_upload_bytes": max_upload_bytes,
        }

    async def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
        suffix = Path(upload.filename or "").suffix.lower()
        kind = ALLOWED_SUFFIXES.get(suffix)
        if kind is None:
            raise HTTPException(
                415, f"Unsupported file type {suffix or '(none)'!r}; send PDF, PNG, JPG or TIFF."
            )
        dest = dest_dir / f"{_safe_stem(upload.filename)}{suffix}"
        size = 0
        head = b""
        with dest.open("wb") as fh:
            while chunk := await upload.read(_CHUNK):
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(
                        413, f"File larger than the limit of {max_upload_bytes / 1e6:.1f} MB."
                    )
                if len(head) < 16:
                    head += chunk[:16]
                fh.write(chunk)
        if size == 0:
            raise HTTPException(422, "The uploaded file is empty.")
        if not head.startswith(_MAGIC[kind]):
            raise HTTPException(
                415, f"File content does not look like {kind.upper()} (extension {suffix})."
            )
        return dest

    @app.post("/convert")
    async def convert_plan(
        request: Request,
        file: Annotated[UploadFile, File(description="Floor plan: PDF, PNG, JPG or TIFF")],
        ifc_schema: Annotated[Literal["IFC4", "IFC2X3"], Form(alias="schema")] = "IFC4",
        dpi: Annotated[float, Form(gt=50, le=600, description="Working resolution")] = 200.0,
        plan_scale: Annotated[
            float, Form(gt=0, le=10000, description="Scale denominator, 100 = 1:100")
        ] = 100.0,
        storey_height_m: Annotated[float, Form(gt=1.5, le=10, description="Room height")] = 2.60,
        page: Annotated[int, Form(ge=0, description="Page of a multi-page PDF (0-based)")] = 0,
        ocr_backend: Annotated[str, Form()] = "rapidocr",
        segment_backend: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        if not segment_backend:  # the run configuration's default (learned)
            from kelpod.runconfig import default_segment_backend

            segment_backend = default_segment_backend()
        if ocr_backend not in ocr.available_engines():
            raise HTTPException(
                422, f"Unknown ocr_backend {ocr_backend!r}; use one of {ocr.available_engines()}."
            )
        if segment_backend not in segment.available():
            raise HTTPException(
                422,
                f"Unknown segment_backend {segment_backend!r}; use one of {segment.available()}.",
            )
        job_id, job_dir = store.new()
        options = {
            "schema": ifc_schema,
            "dpi": dpi,
            "plan_scale": plan_scale,
            "storey_height_m": storey_height_m,
            "page": page,
            "ocr_backend": ocr_backend,
            "segment_backend": segment_backend,
        }
        try:
            src = await _save_upload(file, job_dir)

            def work() -> PipelineResult:
                return convert(
                    src,
                    job_dir / "model.ifc",
                    schema=ifc_schema,
                    dpi=dpi,
                    plan_scale=plan_scale,
                    storey_height_m=storey_height_m,
                    page_index=page,
                    ocr_backend=ocr_backend,
                    segment_backend=segment_backend,
                )

            try:
                res = await anyio.to_thread.run_sync(work, limiter=limiter)
            except Exception as exc:  # any pipeline failure is reported as unprocessable input
                from kelpod.runconfig import CheckpointError

                if isinstance(exc, CheckpointError | ImportError):  # server set-up, not input
                    raise HTTPException(503, f"Segmentation model unavailable: {exc}") from exc
                hint = ""
                if isinstance(exc, IndexError) and page > 0:
                    hint = f" Does the file have a page {page} (0-based)?"
                raise HTTPException(
                    422, f"Conversion failed: {type(exc).__name__}: {exc}.{hint}"
                ) from exc
            if res.export is None or not (job_dir / "model.ifc").is_file():
                raise HTTPException(422, "Conversion produced no IFC file.")
        except BaseException:
            store.discard(job_id)
            raise

        rep = res.graph.report
        body = {
            "job_id": job_id,
            "status": "done",
            "filename": file.filename,
            "options": options,
            "closed": bool(rep and rep.is_closed),
            "summary": _summary(res),
            "issues": _issues(res),
            "report": ({**rep.model_dump(), "is_closed": rep.is_closed} if rep else None),
            "scale": {"m_per_px": res.graph.m_per_px, "source": res.graph.scale_source},
            "export": {
                "schema": res.export.schema_,
                "reopened_ok": res.export.reopened_ok,
                "entity_counts": res.export.entity_counts,
                "warnings": res.export.warnings,
            },
            "timings": [t.model_dump() for t in res.timings],
            "total_seconds": res.total_seconds,
            "download_url": str(request.url_for("download_ifc", job_id=job_id)),
            "result_url": str(request.url_for("get_job", job_id=job_id)),
            "notice": NOTICE,
        }
        store.save_result(job_id, body)
        return body

    @app.get("/jobs/{job_id}", name="get_job")
    def get_job(job_id: str) -> dict[str, Any]:
        body = store.load_result(job_id)
        if body is None:
            raise HTTPException(404, "Unknown or expired job.")
        return body

    @app.get("/jobs/{job_id}/model.ifc", name="download_ifc", response_class=FileResponse)
    def download_ifc(job_id: str) -> FileResponse:
        d = store.path(job_id)
        if d is None or not (d / "model.ifc").is_file():
            raise HTTPException(404, "Unknown or expired job.")
        body = store.load_result(job_id) or {}
        name = f"{_safe_stem(body.get('filename'))}.ifc"
        return FileResponse(d / "model.ifc", media_type="application/x-step", filename=name)

    return app
