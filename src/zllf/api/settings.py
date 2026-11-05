"""Service settings for the API, worker and CLI (env prefix ``ZLLF_``).

Data layout under ``ZLLF_DATA_DIR``::

    zllf.db                       SQLite (WAL) database
    workspaces/<id>/corpus/       uploaded documents (folder structure kept)
    workspaces/<id>/gold.jsonl    gold evaluation set
    runs/<run_id>/                run directories (see zllf.core.manifest.RunPaths)
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from zllf.core.config import CONFIG_DIR, REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZLLF_", extra="ignore")

    data_dir: Path = Path("~/.local/share/zllf")
    host: str = "127.0.0.1"  # loopback by default; set 0.0.0.0 explicitly for LAN access
    port: int = 8000
    hardware: str = "toy"  # default hardware profile for new runs: fake | toy | mac | ...
    config_dir: Path = CONFIG_DIR
    static_dir: Path = REPO_ROOT / "web" / "dist"

    # auth
    session_ttl_hours: float = 12.0
    cookie_secure: bool = False  # set True behind TLS
    login_max_failures: int = 5
    login_window_seconds: int = 300
    min_password_length: int = 8

    # uploads
    max_file_mb: int = 512
    max_request_mb: int = 4096

    # worker
    worker_poll_seconds: float = 1.0
    lease_seconds: float = 60.0

    @property
    def root(self) -> Path:
        return self.data_dir.expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.root / "zllf.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def workspace_dir(self, workspace_id: str) -> Path:
        return self.workspaces_dir / workspace_id

    def corpus_dir(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "corpus"

    def gold_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "gold.jsonl"

    def ensure_dirs(self) -> None:
        for path in (self.root, self.workspaces_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            self.root.chmod(0o700)  # documents are sensitive: owner-only

    def hardware_names(self) -> list[str]:
        return sorted(p.stem for p in (self.config_dir / "hardware").glob("*.yaml"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["REPO_ROOT", "Settings", "get_settings"]
