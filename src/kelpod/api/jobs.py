"""A small on-disk job store: one directory per conversion, pruned by count and age.

Each job directory holds the uploaded file, ``model.ifc`` and ``result.json`` (the JSON the API
returned). Nothing is kept longer than ``ttl_s`` seconds or beyond the newest ``keep_last`` jobs,
and a store that created its own temp directory deletes it on :meth:`JobStore.close`.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class JobStore:
    def __init__(
        self, root: str | Path | None = None, *, keep_last: int = 20, ttl_s: float = 3600.0
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._owns_root = root is None
        self.keep_last = keep_last
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._created: dict[str, float] = {}

    @property
    def root(self) -> Path:
        """Base directory; a fresh temp directory is made on first use if none was given."""
        with self._lock:
            if self._root is None:
                self._root = Path(tempfile.mkdtemp(prefix="kelpod-api-"))
            self._root.mkdir(parents=True, exist_ok=True)
            return self._root

    def new(self) -> tuple[str, Path]:
        """Create an empty job directory (and prune old ones first)."""
        self.prune(reserve=1)
        job_id = uuid.uuid4().hex
        d = self.root / job_id
        d.mkdir()
        with self._lock:
            self._created[job_id] = time.time()
        return job_id, d

    def path(self, job_id: str) -> Path | None:
        """Directory of a finished job, or ``None`` if unknown, expired or malformed."""
        if not _JOB_ID.match(job_id):
            return None
        self.prune()
        with self._lock:
            if job_id not in self._created:
                return None
        d = self.root / job_id
        return d if (d / "result.json").is_file() else None

    def save_result(self, job_id: str, result: dict[str, Any]) -> None:
        (self.root / job_id / "result.json").write_text(json.dumps(result, indent=1), "utf-8")

    def load_result(self, job_id: str) -> dict[str, Any] | None:
        d = self.path(job_id)
        if d is None:
            return None
        return json.loads((d / "result.json").read_text("utf-8"))

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._created.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def prune(self, *, reserve: int = 0) -> None:
        """Drop jobs older than ``ttl_s`` and all but the newest ``keep_last - reserve``."""
        now = time.time()
        with self._lock:
            by_age = sorted(self._created.items(), key=lambda kv: kv[1], reverse=True)
            keep = max(self.keep_last - reserve, 0)
            drop = [j for i, (j, t) in enumerate(by_age) if i >= keep or now - t > self.ttl_s]
            for j in drop:
                self._created.pop(j, None)
            root = self._root
        if root is not None:
            for j in drop:
                shutil.rmtree(root / j, ignore_errors=True)

    def close(self) -> None:
        """Forget all jobs; delete the base directory if this store created it."""
        with self._lock:
            ids, self._created = list(self._created), {}
            root = self._root
        if root is None:
            return
        if self._owns_root:
            shutil.rmtree(root, ignore_errors=True)
            with self._lock:
                self._root = None
        else:
            for j in ids:
                shutil.rmtree(root / j, ignore_errors=True)
