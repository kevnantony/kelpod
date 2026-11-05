"""Offline guard: disable telemetry/network in every library we use (NFR-SEC-01).

Importing ``zllf`` calls :func:`apply_offline_env`. The egress test is the real
guarantee; these variables are defence in depth.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "DO_NOT_TRACK": "1",
    "RAY_USAGE_STATS_ENABLED": "0",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "TOKENIZERS_PARALLELISM": "false",
    "DOCLING_DISABLE_TELEMETRY": "1",
}


def apply_offline_env(*, allow_online: bool | None = None) -> None:
    """Set offline env vars unless ZLLF_ALLOW_ONLINE=1 (build host / model download only)."""
    if allow_online is None:
        allow_online = os.environ.get("ZLLF_ALLOW_ONLINE") == "1"
    if allow_online:
        return
    for key, value in OFFLINE_ENV.items():
        os.environ.setdefault(key, value)


class EgressBlockedError(RuntimeError):
    """Raised when code tries to open a non-loopback connection under :func:`block_egress`."""


_LOOPBACK = ("127.", "::1", "localhost", "0.0.0.0")


def _is_loopback(address: Any) -> bool:
    host = str(address[0]) if isinstance(address, tuple) and address else str(address)
    return host.startswith(_LOOPBACK) or host == ""


@contextmanager
def block_egress() -> Iterator[None]:
    """Make any non-loopback ``socket.connect`` raise. Used by tests."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise EgressBlockedError(f"blocked outbound connection to {address!r}")
        original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise EgressBlockedError(f"blocked outbound connection to {address!r}")
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
