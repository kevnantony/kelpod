from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import ifcopenshell  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from kelpod import __version__  # noqa: E402
from kelpod.api import create_app  # noqa: E402
from kelpod.evaluation.synth import generate, save  # noqa: E402


@pytest.fixture(scope="module")
def plan_dir(tmp_path_factory):
    return save(generate(0, degrade=0), tmp_path_factory.mktemp("synth"))


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", keep_last=5)) as c:
        yield c


def _upload(client, path, name=None, **form):
    with open(path, "rb") as fh:
        return client.post("/convert", files={"file": (name or path.name, fh)}, data=form)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert isinstance(body["ocr_available"], bool)
    assert "none" in body["ocr_backends"] and "classical" in body["segment_backends"]


def test_unsupported_type_415(client, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    r = _upload(client, p)
    assert r.status_code == 415 and "PDF" in r.json()["detail"]


def test_wrong_magic_415(client, tmp_path):
    p = tmp_path / "fake.png"
    p.write_bytes(b"this is not a png at all")
    assert _upload(client, p).status_code == 415


def test_too_large_413(tmp_path, plan_dir):
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", max_upload_bytes=1000)) as c:
        r = _upload(c, plan_dir / "plan.pdf")
    assert r.status_code == 413 and "limit" in r.json()["detail"]
    assert not any((tmp_path / "jobs").iterdir())  # failed upload leaves no job behind


def test_bad_options_422(client, plan_dir):
    assert _upload(client, plan_dir / "plan.pdf", schema="IFC5").status_code == 422
    r = _upload(client, plan_dir / "plan.pdf", ocr_backend="tesseract9")
    assert r.status_code == 422 and "ocr_backend" in r.json()["detail"]


def test_pipeline_error_422(client, tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nnot really a pdf\n")
    r = _upload(client, p, ocr_backend="none")
    assert r.status_code == 422
    assert r.json()["detail"].startswith("Conversion failed:")


def test_missing_page_422(client, plan_dir):
    r = _upload(client, plan_dir / "plan.pdf", ocr_backend="none", page="3")
    assert r.status_code == 422 and "page 3" in r.json()["detail"]


def test_convert_and_download(client, plan_dir, tmp_path):
    r = _upload(
        client, plan_dir / "plan.pdf", "Bestand EG.pdf", ocr_backend="none", schema="IFC2X3"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done" and body["options"]["schema"] == "IFC2X3"
    rep = body["report"]
    assert rep["n_walls"] > 0 and rep["n_rooms"] > 0
    assert body["closed"] == rep["is_closed"]
    assert isinstance(body["issues"], list) and body["summary"]
    assert {t["step"] for t in body["timings"]} >= {"clean", "segment", "geometry", "export"}
    assert body["export"]["reopened_ok"] is True
    assert body["download_url"].endswith(f"/jobs/{body['job_id']}/model.ifc")

    again = client.get(f"/jobs/{body['job_id']}")
    assert again.status_code == 200 and again.json() == body

    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert 'filename="Bestand_EG.ifc"' in dl.headers["content-disposition"]
    out = tmp_path / "down.ifc"
    out.write_bytes(dl.content)
    model = ifcopenshell.open(str(out))
    assert model.schema == "IFC2X3"
    assert len(model.by_type("IfcWall")) == rep["n_walls"]
    assert len(model.by_type("IfcSpace")) == rep["n_rooms"]


def test_unknown_job_404(client):
    assert client.get("/jobs/" + "0" * 32).status_code == 404
    assert client.get("/jobs/..%2F..%2Fetc/model.ifc").status_code == 404


def test_cleanup_keeps_last_n(tmp_path, plan_dir):
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", keep_last=1)) as c:
        png = plan_dir / "plan.png"
        a = _upload(c, png, ocr_backend="none", dpi="100").json()["job_id"]
        b = _upload(c, png, ocr_backend="none", dpi="100").json()["job_id"]
        assert c.get(f"/jobs/{a}").status_code == 404
        assert c.get(f"/jobs/{b}").status_code == 200
        assert [d.name for d in (tmp_path / "jobs").iterdir()] == [b]


def test_temp_dir_removed_on_shutdown(plan_dir):
    app = create_app()
    with TestClient(app) as c:
        assert _upload(c, plan_dir / "plan.png", ocr_backend="none", dpi="100").status_code == 200
        root = app.state.store.root
        assert root.is_dir()
    assert not root.exists()


@pytest.mark.slow
def test_convert_with_rapidocr(client, plan_dir):
    r = _upload(client, plan_dir / "plan.pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["options"]["ocr_backend"] == "rapidocr"
    assert body["export"]["reopened_ok"] is True
    assert body["report"]["n_rooms"] > 0
