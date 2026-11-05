"""Structured JSON logging with run/stage context.

Never log document text or redacted values (NFR-SEC-06). Log counts, IDs and hashes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", json_file: Path | None = None) -> None:
    global _configured
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(json_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    # chatty third-party loggers (fontTools logs every PDF font subset at INFO)
    for noisy in ("fontTools", "matplotlib", "PIL", "httpx", "presidio-analyzer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str, **context: Any) -> Any:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name).bind(**context)
