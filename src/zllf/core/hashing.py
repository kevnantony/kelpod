"""Streaming SHA-256 helpers for files, directories and JSON-able objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_BUF = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_BUF):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def sha256_dir(path: Path) -> str:
    """Hash of a directory tree: relative paths + file hashes, in sorted order."""
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(file).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
