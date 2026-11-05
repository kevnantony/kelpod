"""Loader registry: file extension / magic bytes -> (MIME type, loader kind) (P1.1.1)."""

from __future__ import annotations

import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["pdf", "docx", "txt", "md", "html", "xml", "csv", "parquet", "image"]


@dataclass(frozen=True)
class FileType:
    mime: str
    kind: Kind | None  # None = unsupported
    reason: str | None = None


_BY_EXT: dict[str, tuple[str, Kind]] = {
    ".pdf": ("application/pdf", "pdf"),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "docx"),
    ".txt": ("text/plain", "txt"),
    ".text": ("text/plain", "txt"),
    ".md": ("text/markdown", "md"),
    ".markdown": ("text/markdown", "md"),
    ".html": ("text/html", "html"),
    ".htm": ("text/html", "html"),
    ".xhtml": ("application/xhtml+xml", "html"),
    ".xml": ("application/xml", "xml"),
    ".csv": ("text/csv", "csv"),
    ".tsv": ("text/tab-separated-values", "csv"),
    ".parquet": ("application/vnd.apache.parquet", "parquet"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
    ".tif": ("image/tiff", "image"),
    ".tiff": ("image/tiff", "image"),
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_BY_EXT)


def _sniff(head: bytes, path: Path) -> tuple[str, Kind] | None:
    if head.startswith(b"%PDF-"):
        return _BY_EXT[".pdf"]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return _BY_EXT[".png"]
    if head.startswith(b"\xff\xd8\xff"):
        return _BY_EXT[".jpg"]
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return _BY_EXT[".tif"]
    if head.startswith(b"PAR1"):
        return _BY_EXT[".parquet"]
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                if "word/document.xml" in zf.namelist():
                    return _BY_EXT[".docx"]
        except (zipfile.BadZipFile, OSError):
            return None
    return None


def detect(path: Path) -> FileType:
    """Classify a file. Unknown extensions are sniffed by magic bytes before giving up."""
    ext = path.suffix.lower()
    if ext in _BY_EXT:
        mime, kind = _BY_EXT[ext]
        return FileType(mime, kind)
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        head = b""
    sniffed = _sniff(head, path)
    if sniffed is not None:
        return FileType(sniffed[0], sniffed[1])
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    label = ext or "(no extension)"
    return FileType(mime, None, f"unsupported file type {label}")
