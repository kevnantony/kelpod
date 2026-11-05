"""Alembic migrations match the SQLModel tables; WAL is enabled."""

from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlmodel import SQLModel

import zllf.db.models  # noqa: F401
from zllf.db.engine import init_db


def test_migrations_match_models(tmp_path: Path) -> None:
    engine = init_db(f"sqlite:///{tmp_path / 'z.db'}")
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), SQLModel.metadata)
        assert diff == [], diff
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    init_db(f"sqlite:///{tmp_path / 'z.db'}")  # idempotent
