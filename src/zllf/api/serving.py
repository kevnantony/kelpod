"""Bridge between the API and ``zllf.serve`` (imported lazily; 503 when unavailable).

One chat service per base model (``ServingManager`` = mlx_lm.server subprocess on loopback,
or ``InProcessChat``). Fake models (``fake-*``) get an in-process fake so the whole flow
works in tests and on machines without MLX. Deployment state lives in the DB; services are
in-memory, so on API start every live deployment is marked ``stopped``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zllf.api.errors import ApiError
from zllf.core.logging import get_logger

log = get_logger("zllf.api.serving")


class FakeChatService:
    """Deterministic in-process chat over :class:`zllf.llm.fake.FakeLLM`."""

    url = ""

    def __init__(self, base_model: str) -> None:
        from zllf.llm.fake import FakeLLM

        self.base_model = base_model
        self.llm = FakeLLM(base_model)
        self._deployments: dict[str, Path] = {}

    def deploy(self, name: str, adapter_path: Path) -> Any:
        self._deployments[name] = Path(adapter_path)
        return name

    def undeploy(self, name: str) -> None:
        self._deployments.pop(name, None)

    def chat(self, model: str | None, messages: Sequence[Mapping[str, str]],
             max_tokens: int = 512, temperature: float = 0.0,
             json_schema: dict[str, Any] | None = None) -> str:
        from zllf.core.schemas import Message
        from zllf.llm.base import SamplingParams

        msgs = [Message(role=m["role"], content=m["content"]) for m in messages]  # type: ignore[arg-type]
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        return self.llm.chat([msgs], params, json_schema=json_schema,
                             adapter_path=self._deployments.get(model or ""))[0]

    def stop(self) -> None:
        self._deployments.clear()


class ServingHub:
    """Starts/stops chat services and registers adapters on them."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._services: dict[str, Any] = {}
        self._lock = threading.Lock()

    def service(self, base_model: str, *, start: bool = True) -> Any:
        with self._lock:
            svc = self._services.get(base_model)
            if svc is not None or not start:
                return svc
            svc = self._start(base_model)
            self._services[base_model] = svc
            return svc

    def _start(self, base_model: str) -> Any:
        if base_model.startswith("fake"):
            return FakeChatService(base_model)
        try:
            from zllf.serve import start_chat_service
        except ImportError as exc:
            raise ApiError(503, "E_SERVING_UNAVAILABLE",
                           "Model serving is not available on this appliance.") from exc
        safe = base_model.replace("/", "--")
        return start_chat_service(base_model, log_path=self.log_dir / f"serve-{safe}.log")

    @staticmethod
    def endpoint(svc: Any) -> str:
        url = getattr(svc, "url", "") or ""
        return f"{url}/v1" if url else ""  # in-process services have no HTTP endpoint

    def deploy(self, base_model: str, name: str, adapter_path: Path) -> str:
        svc = self.service(base_model)
        svc.deploy(name, adapter_path)
        return self.endpoint(svc)

    def undeploy(self, base_model: str, name: str) -> None:
        svc = self.service(base_model, start=False)
        if svc is not None:
            svc.undeploy(name)

    def chat(self, base_model: str, model_name: str | None, messages: list[dict[str, str]],
             max_tokens: int, json_schema: dict[str, Any] | None = None) -> str:
        svc = self.service(base_model)
        if isinstance(svc, FakeChatService):
            return svc.chat(model_name, messages, max_tokens=max_tokens, json_schema=json_schema)
        return str(svc.chat(model_name, messages, max_tokens=max_tokens))

    def shutdown(self) -> None:
        with self._lock:
            for svc in self._services.values():
                for method in ("stop", "close"):
                    if hasattr(svc, method):
                        try:
                            getattr(svc, method)()
                        except Exception as exc:  # pragma: no cover - best effort
                            log.warning("serving_stop_failed", error=type(exc).__name__)
                        break
            self._services.clear()
