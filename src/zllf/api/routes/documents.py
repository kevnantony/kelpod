"""Corpus documents: list, streaming upload (files and folders), delete."""

from __future__ import annotations

import mimetypes
import os
import shutil

from fastapi import APIRouter, Request, Response
from sqlmodel import col, select

from zllf.api.deps import DB, CurrentUser, SettingsDep, get_workspace
from zllf.api.errors import ApiError, not_found
from zllf.api.schemas import DocumentInfo
from zllf.api.uploads import new_tmp_dir, receive_files, safe_relative_path
from zllf.api.views import document_info
from zllf.db.engine import audit
from zllf.db.models import Document

router = APIRouter(prefix="/api/workspaces/{workspace_id}/documents", tags=["documents"])

_EXTRA_MIME = {
    ".md": "text/markdown", ".docx": "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document", ".jsonl": "application/x-ndjson",
}


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _EXTRA_MIME.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"


@router.get("", response_model=list[DocumentInfo])
def list_documents(workspace_id: str, db: DB, user: CurrentUser) -> list[DocumentInfo]:
    get_workspace(db, user, workspace_id)
    docs = db.exec(select(Document).where(Document.workspace_id == workspace_id)
                   .order_by(col(Document.path))).all()
    return [document_info(d) for d in docs]


@router.post("", response_model=list[DocumentInfo], status_code=201)
async def upload_documents(
    workspace_id: str, request: Request, db: DB, user: CurrentUser, settings: SettingsDep
) -> list[DocumentInfo]:
    get_workspace(db, user, workspace_id)
    corpus = settings.corpus_dir(workspace_id)
    corpus.mkdir(parents=True, exist_ok=True)
    tmp_dir = new_tmp_dir(settings.workspace_dir(workspace_id))
    try:
        files, paths = await receive_files(
            request, tmp_dir, settings.max_file_mb * 1024 * 1024,
            settings.max_request_mb * 1024 * 1024,
        )
        if not files:
            raise ApiError(422, "E_VALIDATION", "No files were received. Try again.")
        if paths and len(paths) != len(files):
            raise ApiError(422, "E_VALIDATION", "Each file needs exactly one path.")
        rel_paths = [safe_relative_path(paths[i] if paths else f.filename)
                     for i, f in enumerate(files)]
        if len(set(rel_paths)) != len(rel_paths):
            raise ApiError(422, "E_INVALID_PATH", "The same file path appears twice.")
        root = corpus.resolve()
        out: list[Document] = []
        existing = {d.path: d for d in db.exec(
            select(Document).where(Document.workspace_id == workspace_id)).all()}
        by_sha = {d.sha256: d for d in existing.values() if d.status != "skipped"}
        for upload, rel in zip(files, rel_paths, strict=True):
            target = (corpus / rel).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ApiError(422, "E_INVALID_PATH",
                               "File paths must stay inside the uploaded folder.")
            doc = existing.get(rel)
            dup = by_sha.get(upload.sha256)
            if doc is None:
                doc = Document(workspace_id=workspace_id, path=rel, sha256=upload.sha256)
                db.add(doc)
                existing[rel] = doc
            doc.sha256, doc.size_bytes, doc.mime = upload.sha256, upload.size, guess_mime(rel)
            doc.pages, doc.error = None, None
            if dup is not None and dup.path != rel:
                doc.status = "skipped"
                doc.error = f"Same content as “{dup.path}” — skipped as a duplicate."
                target.unlink(missing_ok=True)
            else:
                doc.status = "uploaded"
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(upload.tmp_path, target)
                by_sha[upload.sha256] = doc
            out.append(doc)
        audit(db, user.id, "document.upload", "workspace", workspace_id, files=len(out),
              bytes=sum(f.size for f in files))
        db.flush()
        return [document_info(d) for d in out]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    workspace_id: str, doc_id: str, db: DB, user: CurrentUser, settings: SettingsDep
) -> Response:
    get_workspace(db, user, workspace_id)
    doc = db.get(Document, doc_id)
    if doc is None or doc.workspace_id != workspace_id:
        raise not_found("document")
    corpus = settings.corpus_dir(workspace_id).resolve()
    target = (corpus / doc.path).resolve()
    if target.is_relative_to(corpus) and target != corpus and doc.status != "skipped":
        target.unlink(missing_ok=True)
        parent = target.parent
        while parent != corpus and parent.is_relative_to(corpus):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    db.delete(doc)
    audit(db, user.id, "document.delete", "document", doc_id, workspace_id=workspace_id)
    return Response(status_code=204)
