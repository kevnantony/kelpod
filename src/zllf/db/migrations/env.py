"""Alembic environment. Uses a connection passed via ``config.attributes`` when present."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

import zllf.db.models  # noqa: F401  (register tables)

config = context.config
target_metadata = SQLModel.metadata


def _run(connection: object) -> None:
    context.configure(
        connection=connection,  # type: ignore[arg-type]
        target_metadata=target_metadata,
        render_as_batch=True,  # SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    url = context.get_x_argument(as_dictionary=True).get("url")
    section = config.get_section(config.config_ini_section) or {}
    if url:
        section["sqlalchemy.url"] = url
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as conn:
        _run(conn)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
