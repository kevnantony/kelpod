"""Streaming multipart upload into a workspace corpus (no spooling to the system temp dir).

Uploaded bytes go straight to ``workspaces/<id>/.upload-<uuid>/`` (same filesystem as the
corpus, so the final move is an atomic rename) while their sha256 is computed.
"""

from __future__ import annotations

import hashlib
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any

from fastapi import Request

from zllf.api.errors import ApiError

try:  # python-multipart >= 0.0.13
    from python_multipart.multipart import MultipartParser, parse_options_header
except ImportError:  # pragma: no cover - older python-multipart
    from multipart.multipart import MultipartParser, parse_options_header  # type: ignore

FILE_FIELDS = frozenset({"files", "files[]", "file"})
PATH_FIELDS = frozenset({"paths", "paths[]"})
MAX_TEXT_FIELD = 4096
MAX_PARTS = 10000
MAX_DEPTH = 32


@dataclass
class UploadedFile:
    filename: str
    tmp_path: Path
    size: int = 0
    sha256: str = ""


@dataclass
class _Part:
    headers: dict[bytes, bytes] = field(default_factory=dict)
    name: str = ""
    filename: str | None = None
    fh: IO[bytes] | None = None
    hasher: Any = None
    size: int = 0
    text: bytearray = field(default_factory=bytearray)


def safe_relative_path(raw: str) -> str:
    """Normalise a client-supplied relative path; reject absolute paths and ``..``."""
    value = unicodedata.normalize("NFC", raw).replace("\\", "/").strip()
    if not value or "\x00" in value or any(ord(c) < 32 for c in value):
        raise ApiError(422, "E_INVALID_PATH",
                       "A file name is empty or contains invalid characters.")
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise ApiError(422, "E_INVALID_PATH", "File paths must be relative to the folder.")
    parts = [p for p in PurePosixPath(value).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ApiError(422, "E_INVALID_PATH", "File paths must stay inside the uploaded folder.")
    if len(parts) > MAX_DEPTH or any(len(p.encode("utf-8")) > 255 for p in parts):
        raise ApiError(422, "E_INVALID_PATH", "A file path is too long.")
    return "/".join(parts)


async def receive_files(
    request: Request, tmp_dir: Path, max_file_bytes: int, max_request_bytes: int
) -> tuple[list[UploadedFile], list[str]]:
    """Parse the multipart body, writing file parts to ``tmp_dir``. Returns (files, paths)."""
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise ApiError(422, "E_VALIDATION", "Upload must be multipart form data.")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > max_request_bytes:
        raise ApiError(413, "E_TOO_LARGE", "The upload is too large. Try fewer files at once.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    files: list[UploadedFile] = []
    paths: list[str] = []
    state: dict[str, Any] = {"part": None, "field": b"", "value": b"", "total": 0, "n": 0}
    errors: list[ApiError] = []

    def on_part_begin() -> None:
        state["n"] += 1
        if state["n"] > MAX_PARTS:
            errors.append(ApiError(413, "E_TOO_LARGE", "Too many files in one upload."))
        state["part"] = _Part()

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        state["part"].headers[state["field"].lower()] = state["value"]
        state["field"], state["value"] = b"", b""

    def on_headers_finished() -> None:
        part: _Part = state["part"]
        _disp, opts = parse_options_header(part.headers.get(b"content-disposition", b""))
        part.name = opts.get(b"name", b"").decode("utf-8", "replace")
        if b"filename" in opts and part.name in FILE_FIELDS:
            part.filename = opts[b"filename"].decode("utf-8", "replace")
            tmp = tmp_dir / f"{len(files):06d}.part"
            part.fh = tmp.open("wb")
            part.hasher = hashlib.sha256()
            files.append(UploadedFile(filename=part.filename, tmp_path=tmp))

    def on_part_data(data: bytes, start: int, end: int) -> None:
        part: _Part = state["part"]
        chunk = data[start:end]
        if part.fh is not None:
            part.size += len(chunk)
            if part.size > max_file_bytes and not errors:
                errors.append(ApiError(413, "E_TOO_LARGE",
                                       "A file is larger than the allowed upload size."))
            if not errors:
                part.fh.write(chunk)
                part.hasher.update(chunk)
        elif part.name in PATH_FIELDS:
            part.text += chunk
            if len(part.text) > MAX_TEXT_FIELD and not errors:
                errors.append(ApiError(422, "E_INVALID_PATH", "A file path is too long."))

    def on_part_end() -> None:
        part: _Part = state["part"]
        if part.fh is not None:
            part.fh.close()
            files[-1].size = part.size
            files[-1].sha256 = part.hasher.hexdigest()
        elif part.name in PATH_FIELDS:
            paths.append(part.text.decode("utf-8", "replace"))

    parser = MultipartParser(boundary, {
        "on_part_begin": on_part_begin, "on_part_data": on_part_data,
        "on_part_end": on_part_end, "on_header_field": on_header_field,
        "on_header_value": on_header_value, "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished,
    })
    try:
        async for chunk in request.stream():
            state["total"] += len(chunk)
            if state["total"] > max_request_bytes:
                raise ApiError(413, "E_TOO_LARGE", "The upload is too large. Try fewer files.")
            parser.write(chunk)
            if errors:
                raise errors[0]
        parser.finalize()
    except ApiError:
        _close(state)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as exc:
        _close(state)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ApiError(422, "E_VALIDATION", "The upload could not be read. Try again.") from exc
    return files, paths


def _close(state: dict[str, Any]) -> None:
    part = state.get("part")
    if part is not None and part.fh is not None and not part.fh.closed:
        part.fh.close()


def new_tmp_dir(workspace_dir: Path) -> Path:
    return workspace_dir / f".upload-{uuid.uuid4().hex}"
