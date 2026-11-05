"""CLI via Typer's CliRunner with dummy stages: exit codes, --auto-approve, resume/status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from zllf_testkit import Behaviour, install_dummy_stages

from zllf.cli import app
from zllf.core.offline import block_egress

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    install_dummy_stages(monkeypatch)
    monkeypatch.setenv("ZLLF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ZLLF_HARDWARE", "fake")
    from zllf.api.settings import get_settings

    get_settings.cache_clear()
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.txt").write_text("hello")
    return d


def _invoke(*args: str) -> tuple[int, dict]:
    with block_egress():
        result = runner.invoke(app, list(args))
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("{")]
    return result.exit_code, json.loads(lines[-1]) if lines else {}


def test_run_pauses_for_approval_then_resume(corpus: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    code, out = _invoke("run", "--corpus", str(corpus), "--runs-dir", str(runs), "-q")
    assert code == 2 and out["outcome"] == "awaiting_approval"
    run_dir = out["run_dir"]
    code, out = _invoke("resume", run_dir, "-q")
    assert code == 2  # still not approved
    with block_egress():
        assert runner.invoke(app, ["approve", run_dir]).exit_code == 0
    code, out = _invoke("resume", run_dir, "-q")
    assert code == 0 and out["outcome"] == "completed"
    with block_egress():
        status = runner.invoke(app, ["status", run_dir, "--json"])
    assert status.exit_code == 0
    stages = json.loads(status.stdout)["stages"]
    assert all(s["status"] == "completed" for s in stages.values())


def test_auto_approve_completes(corpus: Path, tmp_path: Path) -> None:
    code, out = _invoke("run", "--corpus", str(corpus), "--runs-dir", str(tmp_path / "r"),
                        "--hardware", "fake", "--auto-approve")
    assert code == 0 and out["outcome"] == "completed"
    assert (Path(out["run_dir"]) / "06_package" / "report.json").exists()


def test_until_and_failure_exit_codes(corpus: Path, tmp_path: Path) -> None:
    code, out = _invoke("run", "--corpus", str(corpus), "--runs-dir", str(tmp_path / "r"),
                        "--until", "ingest", "-q")
    assert code == 0
    Behaviour.fail_stage = "ingest"
    code, out = _invoke("run", "--corpus", str(corpus), "--runs-dir", str(tmp_path / "r"), "-q")
    assert code == 1 and out["outcome"] == "failed"
    with block_egress():
        bad = runner.invoke(app, ["run", "--corpus", str(corpus), "--until", "nope"])
    assert bad.exit_code != 0


def test_cancel_exit_code(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zllf_testkit import DUMMIES

    from zllf.core.errors import CancelledError

    def boom(self, ctx, out_dir):  # type: ignore[no-untyped-def]
        raise CancelledError()

    monkeypatch.setattr(DUMMIES["ingest"], "produce", boom)
    code, out = _invoke("run", "--corpus", str(corpus), "--runs-dir", str(tmp_path / "r"), "-q")
    assert code == 130 and out["outcome"] == "cancelled"


def test_create_admin_and_hardware(corpus: Path) -> None:
    with block_egress():
        r = runner.invoke(app, ["create-admin", "--username", "root", "--password",
                                "longpassword"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["create-admin", "--username", "root", "--password", "short"])
        assert r.exit_code == 1
        hw = runner.invoke(app, ["hardware"])
    assert hw.exit_code == 0
    assert "accelerator" in json.loads(hw.stdout)
