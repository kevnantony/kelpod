"""SQLite engine (WAL), Alembic-driven ``init_db`` and small session helpers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine

from zllf.db.models import AuditEvent

DB_DIR = Path(__file__).resolve().parent
_engines: dict[str, Engine] = {}
_lock = threading.Lock()


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=10000")
    cur.close()


def get_engine(url: str) -> Engine:
    """One engine per URL per process (WAL, foreign keys, busy timeout)."""
    with _lock:
        engine = _engines.get(url)
        if engine is None:
            engine = create_engine(
                url, connect_args={"check_same_thread": False, "timeout": 30}
            )
            if url.startswith("sqlite"):
                event.listen(engine, "connect", _sqlite_pragmas)
            _engines[url] = engine
        return engine


def dispose_engines() -> None:
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


def alembic_config(url: str) -> Any:
    from alembic.config import Config

    cfg = Config(str(DB_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(DB_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def init_db(url: str) -> Engine:
    """Create/upgrade the schema to the latest Alembic revision and return the engine."""
    from alembic import command

    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(url)
    cfg = alembic_config(url)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Session that commits on success and rolls back on error."""
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise


def audit(
    session: Session,
    actor: str,
    action: str,
    object_type: str | None = None,
    object_id: str | None = None,
    **details: Any,
) -> None:
    """Append an audit event. ``details`` must hold IDs/counts only — never document text."""
    session.add(
        AuditEvent(
            actor=actor,
            action=action,
            object_type=object_type,
            object_id=object_id,
            details_json=details,
        )
    )
