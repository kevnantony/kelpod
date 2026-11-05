from __future__ import annotations

import json
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths
from zllf.stages.package import artefacts as A
from zllf.stages.package.pdf import font_paths, loss_chart_png, render_scorecard, scrub
from zllf.stages.package.report import build_report
from zllf.stages.package.sbom import (
    build_sbom,
    hf_cache_files,
    resolve_model,
    sbom_json,
    tree_hash,
    validate_sbom_json,
    write_sbom,
)


def _pdf_text(path: Path) -> str:
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    finally:
        doc.close()


def _report(paths: RunPaths, bench: Path) -> dict:
    return build_report(paths, load_config(hardware="fake"), signing_key_id="k",
                        bench_dir=bench)


def test_fonts_ship_with_matplotlib() -> None:
    assert all(p.exists() for p in font_paths().values())


@pytest.mark.parametrize("ready", [True, False])
def test_pdf_scorecard(run_factory, empty_bench: Path, tmp_path: Path, ready: bool,
                       canary_secret: str) -> None:
    paths = run_factory(gates_pass=ready)
    report = _report(paths, empty_bench)
    examples = A.load_examples(paths)
    assert examples and all(e.suite != "canary" for e in examples)
    out = tmp_path / f"scorecard_{ready}.pdf"
    pages = render_scorecard(report, examples, out, secrets=A.load_canary_secrets(paths))
    assert pages >= 2
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 5000
    text = _pdf_text(out)
    assert ("NOT READY" in text) is (not ready)
    assert "READY" in text
    assert "Accuracy on your documents" in text and "Keeps general skills" in text
    assert "Prüfung" in text and "µg/m³" in text  # Unicode survives
    assert canary_secret not in text and "kontakt@example.org" not in text
    assert "Loss curves" in text


def test_pdf_without_optional_parts(run_factory, empty_bench: Path, tmp_path: Path) -> None:
    paths = run_factory(omit=["train/train_report.json", "train/trials.json",
                              "evaluate/predictions.jsonl", "evaluate/results.json", "ingest"])
    report = _report(paths, empty_bench)
    pages = render_scorecard(report, A.load_examples(paths), tmp_path / "s.pdf")
    assert pages >= 2
    text = _pdf_text(tmp_path / "s.pdf")
    assert "No example predictions" in text and "No loss curves" in text


def test_scrub_and_chart() -> None:
    assert scrub("hi SECRET x@y.de", ["SECRET"]) == "hi [removed] [EMAIL]"
    assert scrub("a" * 500).endswith("…")
    png = loss_chart_png({"train": [[0, 1.0], [1, 0.5]]})
    assert png is not None and png.startswith(b"\x89PNG")
    assert loss_chart_png({}) is None


def test_sbom_validates_with_ml_components(fake_run: RunPaths, tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    (bench / "anchor").mkdir(parents=True)
    (bench / "anchor" / "mmlu.jsonl").write_text('{"q": 1}\n')
    report = _report(fake_run, bench)
    bom = build_sbom(fake_run, load_config(hardware="fake"), report,
                     bench={"anchor": bench / "anchor"})
    text = sbom_json(bom)
    try:
        assert validate_sbom_json(text) is None
    except RuntimeError as exc:  # optional validator deps missing
        pytest.skip(str(exc))
    data = json.loads(text)
    assert data["specVersion"] == "1.6"
    types = {c["type"] for c in data["components"]}
    assert {"machine-learning-model", "data", "library"} <= types
    names = {c["name"] for c in data["components"]}
    assert {"fake-student", "fake-teacher", "synthetic-train", "bench-anchor", "pydantic"} <= names
    assert data["metadata"]["component"]["type"] == "machine-learning-model"
    adapter_hash = data["metadata"]["component"]["hashes"][0]["content"]
    assert adapter_hash == report["adapter"]["sha256"]
    deps = {d["ref"]: d.get("dependsOn", []) for d in data["dependencies"]}
    assert "model:fake-student" in deps["adapter"]
    assert "data:synth/train.jsonl" in deps["adapter"]
    stats = write_sbom(bom, tmp_path / "sbom.cdx.json")
    assert stats["validated"] is True


def test_model_resolution_from_manifest_and_cache(tmp_path: Path) -> None:
    entry = {"revision": "abc", "license": "apache-2.0",
             "files": [{"path": "model.safetensors", "sha256": "a" * 64}]}
    info = resolve_model("org/model", entry)
    assert info["resolved"] and info["sha256"] == tree_hash([("model.safetensors", "a" * 64)])
    assert resolve_model("fake-x", None)["resolved"] is False
    # HF cache layout: snapshot symlink -> blobs/<sha256>
    blobs, snap = tmp_path / "blobs", tmp_path / "snapshots" / "rev1"
    blobs.mkdir()
    snap.mkdir(parents=True)
    (blobs / ("c" * 64)).write_bytes(b"weights")
    (snap / "w.safetensors").symlink_to(blobs / ("c" * 64))
    (snap / "config.json").write_text("{}")
    files = dict(hf_cache_files(snap))
    assert files["w.safetensors"] == "c" * 64  # LFS blob name reused, not re-hashed
    assert len(files["config.json"]) == 64
