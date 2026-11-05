"""API tests: auth/CSRF, workspaces, uploads, run lifecycle with dummy stages, SSE, results."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from zllf_testkit import (
    CALLS,
    CSRF,
    Behaviour,
    create_workspace,
    install_dummy_stages,
    make_settings,
    setup_admin,
    upload,
)

from zllf.api.app import create_app
from zllf.core.offline import block_egress
from zllf.worker.loop import Worker


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    install_dummy_stages(monkeypatch)
    try:  # sse-starlette keeps a module-global exit event bound to the first event loop
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit_event = None
    except ImportError:
        pass
    settings = make_settings(tmp_path)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<html>spa</html>")
    with block_egress():
        app = create_app(settings)
        with TestClient(app) as client:
            yield {"client": client, "app": app, "settings": settings,
                   "worker": Worker(settings, app.state.engine)}


@pytest.fixture
def authed(env: dict[str, Any]) -> dict[str, Any]:
    setup_admin(env["client"])
    return env


def _ready_workspace(env: dict[str, Any]) -> dict:
    ws = create_workspace(env["client"])
    r = upload(env["client"], ws["id"], {"notes/a.txt": b"alpha", "b.txt": b"beta"})
    assert r.status_code == 201, r.text
    return ws


def _start_run(env: dict[str, Any], ws_id: str, **body: Any) -> dict:
    r = env["client"].post(f"/api/workspaces/{ws_id}/runs", json={"mode": "A", **body},
                           headers=CSRF)
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------------ auth
def test_health_and_setup_flow(env: dict[str, Any]) -> None:
    c = env["client"]
    assert c.get("/api/system/health").json()["status"] == "ok"
    assert c.get("/api/setup").json() == {"needs_setup": True}
    assert c.get("/api/me").status_code == 401
    assert c.get("/api/me").json()["error"]["code"] == "E_AUTH"
    weak = c.post("/api/setup", json={"username": "a", "password": "short"}, headers=CSRF)
    assert weak.status_code == 422 and weak.json()["error"]["code"] == "E_WEAK_PASSWORD"
    user = setup_admin(c)
    assert user["role"] == "admin"
    assert c.get("/api/setup").json() == {"needs_setup": False}
    assert c.get("/api/me").json()["username"] == "admin"  # setup also logs in
    again = c.post("/api/setup", json={"username": "x", "password": "longenough"},
                   headers=CSRF)
    assert again.status_code == 409 and again.json()["error"]["code"] == "E_SETUP_DONE"


def test_login_logout_cookie_flags_and_rate_limit(env: dict[str, Any]) -> None:
    c = env["client"]
    setup_admin(c)
    assert c.post("/api/auth/logout", headers=CSRF).status_code == 204
    assert c.get("/api/me").status_code == 401
    r = c.post("/api/auth/login", json={"username": "admin", "password": "correct horse"},
               headers=CSRF)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "zllf_session=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/api/me").status_code == 200
    for _ in range(env["settings"].login_max_failures):
        bad = c.post("/api/auth/login", json={"username": "admin", "password": "nope"},
                     headers=CSRF)
        assert bad.status_code == 401 and bad.json()["error"]["code"] == "E_BAD_CREDENTIALS"
    blocked = c.post("/api/auth/login", json={"username": "admin", "password": "correct horse"},
                     headers=CSRF)
    assert blocked.status_code == 429


def test_csrf_header_required_for_mutations(authed: dict[str, Any]) -> None:
    c = authed["client"]
    body = {"name": "W", "sensitivity": "public", "base_model": "fake-student"}
    r = c.post("/api/workspaces", json=body)
    assert r.status_code == 403 and r.json()["error"]["code"] == "E_CSRF"
    r = c.post("/api/workspaces", json=body,
               headers={**CSRF, "Origin": "http://evil.example"})
    assert r.status_code == 403
    assert c.post("/api/workspaces", json=body, headers=CSRF).status_code == 201


def test_non_admin_sees_only_own_workspaces(authed: dict[str, Any]) -> None:
    from zllf.api.security import hash_password
    from zllf.db.engine import session_scope
    from zllf.db.models import User

    c = authed["client"]
    ws = create_workspace(c, "AdminOnly")
    with session_scope(authed["app"].state.engine) as db:
        db.add(User(username="bob", password_hash=hash_password("bobpassword"), role="user"))
    c.post("/api/auth/logout", headers=CSRF)
    assert c.post("/api/auth/login", json={"username": "bob", "password": "bobpassword"},
                  headers=CSRF).status_code == 200
    assert c.get("/api/workspaces").json() == []
    assert c.get(f"/api/workspaces/{ws['id']}").status_code == 404
    mine = create_workspace(c, "Bobs")
    assert [w["id"] for w in c.get("/api/workspaces").json()] == [mine["id"]]


# ------------------------------------------------------------------ workspaces & documents
def test_workspace_crud_and_models(authed: dict[str, Any]) -> None:
    c = authed["client"]
    models = c.get("/api/models").json()
    assert {"id": "fake-student", "role": "student", "license": None, "installed": True} in models
    bad = c.post("/api/workspaces", headers=CSRF, json={
        "name": "X", "sensitivity": "public", "base_model": "nope"})
    assert bad.status_code == 422
    ws = create_workspace(c)
    assert ws["latest_run"] is None and ws["deployed_adapter_id"] is None
    assert ws["created_at"].endswith("Z")
    dup = c.post("/api/workspaces", headers=CSRF, json={
        "name": "clinic", "sensitivity": "public", "base_model": "fake-student"})
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "E_CONFLICT"
    assert [w["id"] for w in c.get("/api/workspaces").json()] == [ws["id"]]
    assert c.get(f"/api/workspaces/{ws['id']}").json()["name"] == "Clinic"
    assert c.delete(f"/api/workspaces/{ws['id']}", headers=CSRF).status_code == 204
    assert c.get(f"/api/workspaces/{ws['id']}").status_code == 404
    assert not authed["settings"].workspace_dir(ws["id"]).exists()


def test_upload_folder_paths_hash_and_duplicates(authed: dict[str, Any]) -> None:
    import hashlib

    c = authed["client"]
    ws = create_workspace(c)
    r = upload(c, ws["id"], {"reports/2024/q1.txt": b"one", "top.md": b"two",
                             "copy/q1.txt": b"one"})
    assert r.status_code == 201, r.text
    docs = {d["name"]: d for d in r.json()}
    assert docs["reports/2024/q1.txt"]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert docs["reports/2024/q1.txt"]["size_bytes"] == 3
    assert docs["reports/2024/q1.txt"]["status"] == "uploaded"
    assert docs["top.md"]["mime"] == "text/markdown"
    assert docs["copy/q1.txt"]["status"] == "skipped"
    corpus = authed["settings"].corpus_dir(ws["id"])
    assert (corpus / "reports" / "2024" / "q1.txt").read_bytes() == b"one"
    assert not (corpus / "copy" / "q1.txt").exists()
    listed = c.get(f"/api/workspaces/{ws['id']}/documents").json()
    assert len(listed) == 3
    doc_id = docs["top.md"]["id"]
    assert c.delete(f"/api/workspaces/{ws['id']}/documents/{doc_id}",
                    headers=CSRF).status_code == 204
    assert not (corpus / "top.md").exists()
    assert not list(authed["settings"].workspace_dir(ws["id"]).glob(".upload-*"))


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "a/../../b.txt",
                                 "..\\win.txt", "C:\\x.txt", ""])
def test_upload_rejects_path_traversal(authed: dict[str, Any], bad: str) -> None:
    c = authed["client"]
    ws = create_workspace(c)
    r = c.post(f"/api/workspaces/{ws['id']}/documents", headers=CSRF,
               files=[("files[]", ("x.txt", b"data", "text/plain")), ("paths[]", (None, bad))])
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] in ("E_INVALID_PATH", "E_VALIDATION")
    root = authed["settings"].root
    assert not (root.parent / "escape.txt").exists()
    assert c.get(f"/api/workspaces/{ws['id']}/documents").json() == []


def test_upload_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_dummy_stages(monkeypatch)
    settings = make_settings(tmp_path, max_file_mb=1)
    with TestClient(create_app(settings)) as c:
        setup_admin(c)
        ws = create_workspace(c)
        r = upload(c, ws["id"], {"big.bin": b"x" * (1024 * 1024 + 10)})
        assert r.status_code == 413 and r.json()["error"]["code"] == "E_TOO_LARGE"
        assert c.get(f"/api/workspaces/{ws['id']}/documents").json() == []


# ------------------------------------------------------------------ runs
def test_run_lifecycle_with_worker(authed: dict[str, Any]) -> None:
    c, worker = authed["client"], authed["worker"]
    ws = _ready_workspace(authed)
    run = _start_run(authed, ws["id"])
    assert run["state"] == "queued" and run["hardware"] == "fake"
    assert [s["name"] for s in run["stages"]] == [
        "ingest", "sanitize", "synth", "train", "evaluate", "package"]
    again = c.post(f"/api/workspaces/{ws['id']}/runs", json={"mode": "A"}, headers=CSRF)
    assert again.status_code == 409 and again.json()["error"]["code"] == "E_RUN_ACTIVE"
    assert c.get(f"/api/runs/{run['id']}/redactions").status_code == 404
    assert c.get(f"/api/runs/{run['id']}/scorecard").status_code == 404

    assert worker.run_once()
    r = c.get(f"/api/runs/{run['id']}").json()
    assert r["state"] == "awaiting_approval"
    assert r["current_stage"] == "sanitize"
    assert {s["name"]: s["status"] for s in r["stages"]}["sanitize"] == "awaiting_approval"
    docs = c.get(f"/api/workspaces/{ws['id']}/documents").json()
    assert {d["status"] for d in docs} == {"parsed"} and docs[0]["pages"] == 1

    red = c.get(f"/api/runs/{run['id']}/redactions").json()
    assert red["counts_by_type"] == {"PERSON": 1}
    assert red["chunks_total"] == 1 and red["chunks_dropped"] == 0
    sample = red["samples"][0]
    assert "Alice" not in sample["text_redacted"]
    span = sample["spans"][0]
    assert sample["text_redacted"][span["start"]:span["end"]] == "<PERSON>"
    assert c.get(f"/api/runs/{run['id']}/scorecard").status_code == 404

    assert c.post(f"/api/runs/{run['id']}/resume", headers=CSRF).status_code == 409
    approved = c.post(f"/api/runs/{run['id']}/approve", headers=CSRF).json()
    assert approved["state"] == "queued"
    assert worker.run_once()
    r = c.get(f"/api/runs/{run['id']}").json()
    assert r["state"] == "completed", r
    assert all(s["status"] == "completed" for s in r["stages"])
    assert CALLS["ingest"] == 1 and CALLS["sanitize"] == 1

    trials = c.get(f"/api/runs/{run['id']}/trials").json()
    assert trials[0]["number"] == 0 and trials[0]["params"]["rank"] == 8
    card = c.get(f"/api/runs/{run['id']}/scorecard").json()
    assert card["all_passed"] is True and card["gates"][0]["gate"] == "G1_domain"
    assert set(card["metrics"]) == {"base", "base_rag", "adapter", "adapter_rag"}
    assert card["metrics"]["adapter"] == {"f1": 0.6}
    assert card["examples"] == [{"prompt": "What happened?", "base": "No idea.",
                                 "adapter": "A visit.", "reference": "A visit."}]
    assert c.get(f"/api/runs/{run['id']}/report.pdf").content.startswith(b"%PDF")
    assert c.get(f"/api/runs/{run['id']}/report.json").json()["run_id"] == run["id"]

    runs = c.get(f"/api/workspaces/{ws['id']}/runs").json()
    assert runs[0]["id"] == run["id"] and runs[0]["state"] == "completed"
    assert c.get(f"/api/workspaces/{ws['id']}").json()["latest_run"]["id"] == run["id"]

    adapters = c.get("/api/adapters").json()
    assert len(adapters) == 1 and adapters[0]["run_id"] == run["id"]
    assert c.get(f"/api/runs/{run['id']}").json()["adapter_id"] == adapters[0]["id"]
    assert card["adapter_id"] == adapters[0]["id"]
    with c.stream("GET", f"/api/runs/{run['id']}/events") as resp:
        types = [json.loads(ln[5:])["type"] for ln in "".join(resp.iter_text()).splitlines()
                 if ln.startswith("data:")]
    assert "approved" in types and types[-1] == "run_completed"
    assert adapters[0]["gates_passed"] is True and adapters[0]["size_bytes"] > 0


def test_deploy_and_playground_with_fake_model(authed: dict[str, Any]) -> None:
    import time

    c, worker = authed["client"], authed["worker"]
    ws = _ready_workspace(authed)
    run = _start_run(authed, ws["id"])
    worker.run_once()
    c.post(f"/api/runs/{run['id']}/approve", headers=CSRF)
    worker.run_once()
    adapter = c.get("/api/adapters").json()[0]
    chat = {"adapter_id": adapter["id"], "messages": [{"role": "user", "content": "Hi there."}]}
    r = c.post("/api/playground/chat", json=chat, headers=CSRF)
    assert r.status_code == 409 and r.json()["error"]["code"] == "E_NOT_DEPLOYED"
    dep = c.post(f"/api/adapters/{adapter['id']}/deploy", headers=CSRF).json()
    assert dep["state"] in ("starting", "ready") and dep["model_name"] == "ws-clinic-v1"
    for _ in range(100):
        if c.get("/api/deployments").json()[0]["state"] == "ready":
            break
        time.sleep(0.02)
    assert c.get("/api/deployments").json()[0]["state"] == "ready"
    assert c.get(f"/api/workspaces/{ws['id']}").json()["deployed_adapter_id"] == adapter["id"]
    r = c.post("/api/playground/chat", json=chat, headers=CSRF)
    assert r.status_code == 200, r.text
    assert r.json() == {"output": "Hi there.", "citations": [], "json_valid": None}
    base = c.post("/api/playground/chat", headers=CSRF,
                  json={"adapter_id": None, "messages": [{"role": "user", "content": "Yo."}]})
    assert base.status_code == 200
    stopped = c.post(f"/api/deployments/{dep['id']}/stop", headers=CSRF).json()
    assert stopped["state"] == "stopped"
    assert c.get("/api/adapters").json()[0]["deployed"] is False


def test_failed_run_can_resume(authed: dict[str, Any]) -> None:
    c, worker = authed["client"], authed["worker"]
    ws = _ready_workspace(authed)
    run = _start_run(authed, ws["id"])
    Behaviour.fail_stage = "ingest"
    worker.run_once()
    r = c.get(f"/api/runs/{run['id']}").json()
    assert r["state"] == "failed"
    assert r["error"] == "Training failed. Your data is safe; you can resume the job."
    assert r["stages"][0]["status"] == "failed"
    Behaviour.fail_stage = None
    assert c.post(f"/api/runs/{run['id']}/resume", headers=CSRF).json()["state"] == "queued"
    worker.run_once()
    assert c.get(f"/api/runs/{run['id']}").json()["state"] == "awaiting_approval"


def test_cancel_queued_and_running(authed: dict[str, Any]) -> None:
    c, worker = authed["client"], authed["worker"]
    ws = _ready_workspace(authed)
    run = _start_run(authed, ws["id"])
    r = c.post(f"/api/runs/{run['id']}/cancel", headers=CSRF).json()
    assert r["state"] == "cancelled"
    assert c.post(f"/api/runs/{run['id']}/cancel", headers=CSRF).status_code == 409
    assert worker.run_once() is False  # nothing queued

    c.post(f"/api/runs/{run['id']}/resume", headers=CSRF)
    Behaviour.block_stage = "ingest"
    t = threading.Thread(target=worker.run_once)
    t.start()
    assert Behaviour.started.wait(10)
    assert c.get(f"/api/runs/{run['id']}").json()["state"] == "running"
    r = c.post(f"/api/runs/{run['id']}/cancel", headers=CSRF).json()
    assert r["state"] == "running"  # worker stops it cooperatively
    t.join(15)
    r = c.get(f"/api/runs/{run['id']}").json()
    assert r["state"] == "cancelled" and r["stages"][0]["status"] == "failed"


def test_mode_c_requires_schema_and_no_documents(authed: dict[str, Any]) -> None:
    c = authed["client"]
    ws = create_workspace(c)
    r = c.post(f"/api/workspaces/{ws['id']}/runs", json={"mode": "A"}, headers=CSRF)
    assert r.status_code == 409 and r.json()["error"]["code"] == "E_NO_DOCUMENTS"
    upload(c, ws["id"], {"a.txt": b"x"})
    r = c.post(f"/api/workspaces/{ws['id']}/runs", json={"mode": "C"}, headers=CSRF)
    assert r.status_code == 422 and r.json()["error"]["code"] == "E_SCHEMA_REQUIRED"
    r = c.post(f"/api/workspaces/{ws['id']}/runs", json={"mode": "A", "hardware": "nope"},
               headers=CSRF)
    assert r.status_code == 422
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    run = _start_run(authed, ws["id"], mode="C", json_schema=schema, budget_seconds=900)
    manifest = json.loads((Path(authed["settings"].runs_dir) / run["id"] / "manifest.json")
                          .read_text())
    assert manifest["config"]["json_schema"] == schema
    assert manifest["config"]["budget"]["total_seconds"] == 900


def test_sse_replays_events(authed: dict[str, Any]) -> None:
    c, worker = authed["client"], authed["worker"]
    ws = _ready_workspace(authed)
    run = _start_run(authed, ws["id"])
    Behaviour.fail_stage = "sanitize"
    worker.run_once()  # -> failed (terminal), so the stream ends after the replay
    with c.stream("GET", f"/api/runs/{run['id']}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    data = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
    types = [e["type"] for e in data]
    assert types[0] == "stage_started" and "stage_failed" in types
    assert all(e["run_id"] == run["id"] and "ts" in e for e in data)
    assert "event: message" in body
    with c.stream("GET", f"/api/runs/{run['id']}/events",
                  headers={"Last-Event-ID": str(len(data) - 1)}) as resp:
        tail = [ln for ln in "".join(resp.iter_text()).splitlines() if ln.startswith("data:")]
    assert len(tail) == 1


def test_gold_set_roundtrip(authed: dict[str, Any]) -> None:
    c = authed["client"]
    ws = create_workspace(c)
    items = [{"question": "Q1?", "reference_answer": "A1", "reference_json": None,
              "answerable": True},
             {"question": "Q2?", "reference_answer": None, "reference_json": {"k": 1},
              "answerable": False}]
    r = c.post(f"/api/workspaces/{ws['id']}/gold", json=items, headers=CSRF)
    assert r.status_code == 200 and r.json() == items
    assert c.get(f"/api/workspaces/{ws['id']}/gold").json() == items
    lines = authed["settings"].gold_path(ws["id"]).read_text().splitlines()
    assert [json.loads(x)["question"] for x in lines] == ["Q1?", "Q2?"]
    upload(c, ws["id"], {"a.txt": b"x"})
    run = _start_run(authed, ws["id"])
    snap = Path(authed["settings"].runs_dir) / run["id"] / "gold.jsonl"
    assert snap.read_text().splitlines() == lines
    assert c.post(f"/api/workspaces/{ws['id']}/gold", json=[{"question": " "}],
                  headers=CSRF).status_code == 422


def test_system_hardware_spa_and_unknown_api(authed: dict[str, Any]) -> None:
    c = authed["client"]
    hw = c.get("/api/system/hardware").json()
    assert {"platform", "accelerator", "memory_total_gb", "backend"} <= set(hw)
    assert c.get("/workspaces/abc").text == "<html>spa</html>"
    r = c.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "E_NOT_FOUND"


def test_audit_log_has_no_document_names(authed: dict[str, Any]) -> None:
    from sqlmodel import Session, select

    from zllf.db.models import AuditEvent

    c = authed["client"]
    ws = create_workspace(c)
    upload(c, ws["id"], {"secret-patient-name.txt": b"x"})
    with Session(authed["app"].state.engine) as db:
        events = db.exec(select(AuditEvent)).all()
    actions = {e.action for e in events}
    assert {"setup.admin_created", "workspace.create", "document.upload"} <= actions
    assert "secret-patient-name" not in json.dumps([e.details_json for e in events])


def test_scorecard_flattens_nested_results(tmp_path: Path) -> None:
    from zllf.api.artefacts import scorecard
    from zllf.core.manifest import RunPaths

    paths = RunPaths(tmp_path)
    ev = paths.stage("evaluate")
    ev.mkdir(parents=True)
    (ev / "gates.json").write_text(json.dumps([{"gate": "G1_domain", "passed": True,
                                                "value": 0.3, "threshold": 0.12}]))
    (ev / "results.json").write_text(json.dumps({"variants": {"adapter": {"domain_test": {
        "rouge_l": {"value": 0.5, "ci": [0.4, 0.6], "n": 10}}}}}))
    (ev / "items.jsonl").write_text(json.dumps({"item_id": "i1", "suite": "gold", "prompt": [
        {"role": "user", "content": "Q?"}], "reference": "R"}) + "\n")
    (ev / "predictions.jsonl").write_text("".join(json.dumps(
        {"item_id": "i1", "suite": "gold", "variant": v, "output": v}) + "\n"
        for v in ("base", "adapter")))
    card = scorecard("r1", paths)
    assert card.metrics["adapter"] == {"domain_test.rouge_l": 0.5,
                                       "domain_test.rouge_l.ci_low": 0.4,
                                       "domain_test.rouge_l.ci_high": 0.6}
    assert card.metrics["base"] == {}
    assert card.examples[0].prompt == "Q?" and card.examples[0].reference == "R"
