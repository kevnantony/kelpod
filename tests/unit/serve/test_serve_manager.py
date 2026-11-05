"""Serving manager tests. Pure tests always run; the server test needs MLX + the 0.5B model."""

from __future__ import annotations

from pathlib import Path

import pytest

from zllf.core.errors import ZLLFError
from zllf.serve.manager import ServingManager, deployment_model_name

TOY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_deployment_model_name() -> None:
    assert deployment_model_name("Legal Team", 3) == "ws-legal-team-v3"


def test_refuses_non_loopback(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ServingManager(TOY_MODEL, log_path=tmp_path / "s.log", host="0.0.0.0")


def test_server_env_is_offline() -> None:
    env = ServingManager.server_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert not any(k.upper() in ("HTTP_PROXY", "HTTPS_PROXY") for k in env)


def test_command_binds_loopback(tmp_path: Path) -> None:
    mgr = ServingManager(TOY_MODEL, log_path=tmp_path / "s.log", port=12345)
    mgr._model_path = Path("/models/x")
    cmd = mgr.command()
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--allowed-origins" in cmd


def test_unknown_deployment_and_bad_names(tmp_path: Path) -> None:
    mgr = ServingManager(TOY_MODEL, log_path=tmp_path / "s.log")
    with pytest.raises(ZLLFError):
        mgr._adapter_for("ws-unknown-v1")
    assert mgr._adapter_for(None) is None and mgr._adapter_for("base") is None
    with pytest.raises(ValueError):
        mgr.deploy("../etc", tmp_path)
    with pytest.raises(ZLLFError):
        mgr.deploy("ws-a-v1", tmp_path)  # not an adapter dir


@pytest.mark.mlx
@pytest.mark.slow
def test_server_roundtrip_with_adapter(tmp_path: Path) -> None:
    pytest.importorskip("mlx_lm")
    from zllf.llm.mlx_backend import resolve_model_path

    try:
        resolve_model_path(TOY_MODEL)
    except ZLLFError:
        pytest.skip("toy model not cached")
    from zllf.llm.mlx_backend import MLXBackend, write_random_lora_adapter

    with MLXBackend(TOY_MODEL, max_seq_len=512) as b:
        adapter = write_random_lora_adapter(b, tmp_path / "adapter", num_layers=4, std=0.05)
    log = tmp_path / "server.log"
    msgs = [{"role": "user", "content": "Describe the moon in one sentence."}]
    with ServingManager(TOY_MODEL, log_path=log, startup_timeout=180) as mgr:
        assert mgr.state == "ready" and mgr.url.startswith("http://127.0.0.1:")
        mgr.deploy("ws-test-v1", adapter)
        base = mgr.chat(None, msgs, max_tokens=16)
        tuned = mgr.chat("ws-test-v1", msgs, max_tokens=16)
        base_again = mgr.chat("base", msgs, max_tokens=16)
    assert base.strip() and tuned.strip()
    assert tuned != base
    assert base_again == base
    assert mgr.state == "stopped" and not mgr.running
    assert log.exists()
