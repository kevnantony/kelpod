"""Serving of deployed adapters (FR-15, FR-21 basic) on Apple Silicon via ``mlx_lm.server``.

``ServingManager`` runs ``mlx_lm server`` (OpenAI-compatible HTTP) as a subprocess bound to
127.0.0.1 for one base model, with offline/telemetry env vars, a readiness probe (``/health``
plus a 1-token warm-up chat that forces the model load) and logs to a file.

Adapters are served by name: ``deploy(name, adapter_dir)`` registers a stable model name
(e.g. ``ws-<workspace>-v<n>``, see :func:`deployment_model_name`) and requests for that name
are sent with mlx-lm's per-request ``"adapters": <dir>`` field.

Limitation (mlx-lm 0.32): the server holds ONE (model, adapter) combination at a time. A
request for a different adapter makes it reload the base weights from disk and apply the
new adapter (~0.5 s for 0.5B, seconds for 7B), and requests for different adapters are
served one combination after another. There is no concurrent multi-LoRA batching like
vLLM's ``--enable-lora``. The request ``model`` field must be the server's model *path*
(``"default_model"`` ignores ``adapters``), so the manager never forwards client-supplied
model strings: only registered names are accepted.

``InProcessChat`` offers the same ``chat`` API on top of :class:`MLXBackend` for
environments where a server subprocess cannot be started.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.offline import OFFLINE_ENV
from zllf.core.schemas import Message

LOOPBACK = "127.0.0.1"
BASE = "base"  # model name that selects the base model without an adapter
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DeployState = Literal["starting", "ready", "stopped", "failed"]


def deployment_model_name(workspace: str, version: int) -> str:
    """Stable served model name for a workspace adapter version (P1.7.3)."""
    slug = re.sub(r"[^a-z0-9]+", "-", workspace.lower()).strip("-") or "ws"
    return f"ws-{slug}-v{version}"


def free_port(host: str = LOOPBACK) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _as_dicts(messages: Sequence[Message | Mapping[str, str]]) -> list[dict[str, str]]:
    out = []
    for m in messages:
        if isinstance(m, Message):
            out.append({"role": m.role, "content": m.content})
        else:
            out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


class ChatService(Protocol):
    def chat(
        self,
        model: str | None,
        messages: Sequence[Message | Mapping[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...


@dataclass(frozen=True)
class Deployment:
    name: str
    adapter_path: Path


class ServingManager:
    """Owns one ``mlx_lm server`` subprocess for ``base_model`` plus named adapters."""

    def __init__(
        self,
        base_model: str,
        *,
        log_path: Path,
        port: int | None = None,
        host: str = LOOPBACK,
        max_tokens_default: int = 512,
        startup_timeout: float = 120.0,
        request_timeout: float = 300.0,
    ) -> None:
        if host not in (LOOPBACK, "localhost", "::1"):
            raise ValueError("the model server may only bind to loopback")
        self.base_model = base_model
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.max_tokens_default = max_tokens_default
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.state: DeployState = "stopped"
        self._model_path: Path | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None
        self._deployments: dict[str, Deployment] = {}
        self._lock = threading.Lock()
        # trust_env=False: never route through HTTP(S)_PROXY, loopback only.
        self._client: httpx.Client | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def command(self) -> list[str]:
        assert self._model_path is not None and self.port is not None
        return [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(self._model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--allowed-origins",
            f"http://{self.host}",
            "--max-tokens",
            str(self.max_tokens_default),
            "--log-level",
            "WARNING",
        ]

    @staticmethod
    def server_env() -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if "PROXY" not in k.upper()}
        env.update(OFFLINE_ENV)
        env["NO_PROXY"] = "*"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def start(self) -> None:
        """Start the server and block until it answers a warm-up chat."""
        from zllf.llm.mlx_backend import resolve_model_path

        if self.running:
            return
        self._model_path = resolve_model_path(self.base_model)
        self.port = self.port or free_port(self.host)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("ab")
        self.state = "starting"
        self._proc = subprocess.Popen(
            self.command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.server_env(),
            start_new_session=True,
        )
        self._client = httpx.Client(
            base_url=self.url, trust_env=False, timeout=self.request_timeout
        )
        try:
            self._wait_ready()
        except Exception:
            self.state = "failed"
            self.stop()
            self.state = "failed"
            raise
        self.state = "ready"

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        assert self._client is not None
        while time.monotonic() < deadline:
            if not self.running:
                raise ZLLFError(ErrorCode.INTERNAL, f"model server exited; see {self.log_path}")
            try:
                r = self._client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        else:
            raise ZLLFError(ErrorCode.INTERNAL, "model server did not become healthy")
        remaining = max(5.0, deadline - time.monotonic())
        self._post_chat(None, [{"role": "user", "content": "hi"}], 1, 0.0, timeout=remaining)

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the server (SIGTERM, then SIGKILL) and close the log."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=timeout)
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.state = "stopped"

    def __enter__(self) -> ServingManager:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ adapters
    def deploy(self, name: str, adapter_path: Path) -> Deployment:
        """Register ``adapter_path`` under the served model name ``name``."""
        from zllf.llm.mlx_backend import check_adapter_dir

        if not _NAME.match(name) or name == BASE:
            raise ValueError(f"invalid model name {name!r}")
        dep = Deployment(name, check_adapter_dir(adapter_path))
        self._deployments[name] = dep
        return dep

    def undeploy(self, name: str) -> None:
        self._deployments.pop(name, None)

    def deployments(self) -> list[Deployment]:
        return list(self._deployments.values())

    def _adapter_for(self, model: str | None) -> Path | None:
        if model in (None, "", BASE, self.base_model):
            return None
        dep = self._deployments.get(str(model))
        if dep is None:
            raise ZLLFError(ErrorCode.MODEL_MISSING, f"no deployment named {model!r}")
        return dep.adapter_path

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        model: str | None,
        messages: Sequence[Message | Mapping[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Chat completion for a deployment name (or ``None``/``"base"`` for the base)."""
        if not self.running:
            raise ZLLFError(ErrorCode.INTERNAL, "model server is not running")
        return self._post_chat(
            self._adapter_for(model), _as_dicts(messages), max_tokens, temperature
        )

    def _post_chat(
        self,
        adapter: Path | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        assert self._client is not None and self._model_path is not None
        body: dict[str, Any] = {
            # must be the path: "default_model" would ignore the adapters field
            "model": str(self._model_path),
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if adapter is not None:
            body["adapters"] = str(adapter)
        with self._lock:  # one adapter combination at a time; avoid reload thrash
            r = self._client.post(
                "/v1/chat/completions",
                json=body,
                timeout=timeout if timeout is not None else self.request_timeout,
            )
        if r.status_code != 200:
            raise ZLLFError(ErrorCode.INTERNAL, f"model server returned {r.status_code}")
        return str(r.json()["choices"][0]["message"]["content"])


class InProcessChat:
    """``ChatService`` fallback using :class:`MLXBackend` in this process."""

    def __init__(self, base_model: str, max_seq_len: int = 2048) -> None:
        from zllf.llm.mlx_backend import MLXBackend, check_adapter_dir

        self._check = check_adapter_dir
        self.backend = MLXBackend(base_model, max_seq_len=max_seq_len)
        self.base_model = base_model
        self._deployments: dict[str, Path] = {}

    def deploy(self, name: str, adapter_path: Path) -> Deployment:
        if not _NAME.match(name) or name == BASE:
            raise ValueError(f"invalid model name {name!r}")
        self._deployments[name] = self._check(adapter_path)
        return Deployment(name, self._deployments[name])

    def undeploy(self, name: str) -> None:
        self._deployments.pop(name, None)

    def chat(
        self,
        model: str | None,
        messages: Sequence[Message | Mapping[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        from zllf.llm.base import SamplingParams

        adapter: Path | None = None
        if model not in (None, "", BASE, self.base_model):
            if str(model) not in self._deployments:
                raise ZLLFError(ErrorCode.MODEL_MISSING, f"no deployment named {model!r}")
            adapter = self._deployments[str(model)]
        msgs = [Message.model_validate(m) for m in _as_dicts(messages)]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        return self.backend.chat([msgs], params, adapter_path=adapter)[0]

    def close(self) -> None:
        self.backend.close()


def start_chat_service(
    base_model: str, *, log_path: Path, prefer_server: bool = True, max_seq_len: int = 2048
) -> ServingManager | InProcessChat:
    """Start the HTTP server; fall back to in-process MLX if it cannot be started."""
    if prefer_server:
        mgr = ServingManager(base_model, log_path=log_path)
        try:
            mgr.start()
            return mgr
        except ZLLFError as exc:
            if exc.code == ErrorCode.MODEL_MISSING:
                raise
        except OSError:
            pass
    return InProcessChat(base_model, max_seq_len=max_seq_len)
