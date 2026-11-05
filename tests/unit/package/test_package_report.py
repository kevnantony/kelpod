from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.manifest import RunPaths
from zllf.stages.package import artefacts as A
from zllf.stages.package.report import build_report, validate_report, write_report


def _report(paths: RunPaths, bench: Path) -> dict:
    return build_report(paths, load_config(hardware="fake"), signing_key_id="abc",
                        bench_dir=bench, now=1_790_000_000.0)


def test_full_report_validates_and_has_all_sections(fake_run: RunPaths, empty_bench: Path,
                                                    tmp_path: Path) -> None:
    report = _report(fake_run, empty_bench)
    validate_report(report)
    write_report(report, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["run_id"] == report["run_id"]

    assert report["ready"] is True
    assert report["inputs"]["n_documents"] == 3
    assert report["inputs"]["by_status"] == {"failed": 1, "parsed": 2}
    assert report["redaction"]["counts"]["n_redactions"] == 9
    assert report["redaction"]["counts"]["by_entity_type"]["PERSON"] == 3
    assert report["synthesis"]["splits"]["train"]["n"] == 8
    assert report["synthesis"]["splits"]["test"]["n_unanswerable"] == 1
    assert report["training"]["n_trials"] == 3
    assert report["training"]["final_params"]["rank"] == 16  # best complete trial
    assert set(report["training"]["loss_curves"]) == {"train", "val"}
    metrics = report["evaluation"]["metrics"]
    assert set(metrics) == {"base", "base_rag", "adapter", "adapter_rag"}
    rouge = metrics["adapter"]["domain_test/rougeL"]
    assert rouge["value"] == 0.5 and rouge["ci_low"] == pytest.approx(0.45)
    assert metrics["base"]["unanswerable/abstention"]["ci_high"] == 1.0
    assert report["evaluation"]["gates"][0]["label"] == "Accuracy on your documents"
    assert report["timings"]["train"]["seconds"] == 42.0
    assert report["adapter"]["size_bytes"] == 4096 + len(
        (fake_run.stage("train") / "adapter" / "adapter_config.json").read_bytes())
    assert report["signing"]["key_id"] == "abc"
    assert report["licenses"]["models"][0]["source"] == "fake-test-model"


def test_report_contains_no_text_or_paths(fake_run: RunPaths, empty_bench: Path,
                                          raw_secret_name: str) -> None:
    text = json.dumps(_report(fake_run, empty_bench), ensure_ascii=False)
    assert raw_secret_name not in text  # free-form report strings are dropped
    assert "secret_0.pdf" not in text  # document file names are not recorded
    assert "Grenzwert" not in text  # no synthetic Q/A text
    assert "text dropped" not in text  # gate details are numeric-only


@pytest.mark.parametrize("omit", [
    ("ingest",), ("sanitize",), ("synth",),
    ("train/trials.json", "train/hw_profile.json", "train/train_report.json"),
    ("evaluate/results.json", "evaluate/predictions.jsonl", "evaluate/eval_report.json"),
])
def test_missing_optional_artefacts_are_recorded_not_fatal(run_factory, empty_bench: Path,
                                                           omit: tuple[str, ...]) -> None:
    paths = run_factory(omit=omit)
    report = _report(paths, empty_bench)
    validate_report(report)
    section = {"ingest": "inputs", "sanitize": "redaction", "synth": "synthesis",
               "train/trials.json": "training"}.get(omit[0])
    if section:
        assert report[section] == {"present": False}
    absent = [k for k, v in report["artefacts"].items() if not v]
    assert absent


@pytest.mark.parametrize("omit", ["train/adapter", "evaluate/gates.json"])
def test_missing_required_artefacts_fail(run_factory, empty_bench: Path, omit: str) -> None:
    paths = run_factory(omit=[omit])
    with pytest.raises(ZLLFError) as err:
        _report(paths, empty_bench)
    assert err.value.code is ErrorCode.PACKAGE_FAILED


def test_works_without_manifest(fake_run: RunPaths, empty_bench: Path) -> None:
    fake_run.manifest.unlink()
    report = _report(fake_run, empty_bench)
    validate_report(report)
    assert report["timings"]["ingest"]["status"] == "unknown"


def test_schema_rejects_bad_report(fake_run: RunPaths, empty_bench: Path) -> None:
    report = _report(fake_run, empty_bench)
    bad = copy.deepcopy(report)
    bad["inputs"]["documents"][0]["path"] = "leak.pdf"
    with pytest.raises(ZLLFError):
        validate_report(bad)
    bad = copy.deepcopy(report)
    bad["redaction"]["report"] = {"value": "Erika"}
    with pytest.raises(ZLLFError):
        validate_report(bad)
    bad = copy.deepcopy(report)
    del bad["adapter"]
    with pytest.raises(ZLLFError):
        validate_report(bad)


def test_failed_gate_means_not_ready(run_factory, empty_bench: Path) -> None:
    report = _report(run_factory(gates_pass=False), empty_bench)
    assert report["ready"] is False


def test_models_manifest_and_bench_licences(fake_run: RunPaths, tmp_path: Path) -> None:
    bench = tmp_path / "bench2"
    (bench / "replay").mkdir(parents=True)
    (bench / "replay" / "r.jsonl").write_text("{}\n")
    (bench / "replay" / "manifest.json").write_text(json.dumps({"license": "CC-BY-4.0"}))
    mm = tmp_path / "models.json"
    mm.write_text(json.dumps({"models": [{"name": "fake-nli", "license": "mit",
                                          "revision": "r1", "files": []}]}))
    cfg = load_config(hardware="fake")
    report = build_report(fake_run, cfg, models_manifest=mm, bench_dir=bench)
    validate_report(report)
    datasets = {d["name"]: d["license"] for d in report["licenses"]["datasets"]}
    assert datasets["bench/replay"] == "CC-BY-4.0"


def test_metric_normalisation_shapes() -> None:
    flat = A.normalize_metrics({"rougeL": {"base": 0.1, "adapter": {"value": 0.2, "lo": 0.1,
                                                                    "hi": 0.3}}})
    assert flat["adapter"]["rougeL"].ci_high == 0.3
    assert A.normalize_metrics(None) == {}
    assert A.normalize_metrics({"base": "n/a"}) == {}


def test_loss_curve_shapes() -> None:
    assert A.loss_curves({"loss_curve": [3.0, 2.0]}) == {"train": [[0.0, 3.0], [1.0, 2.0]]}
    hist = A.loss_curves({"history": [{"step": 5, "train_loss": 1.0, "val_loss": 1.5}]})
    assert hist == {"train": [[5.0, 1.0]], "val": [[5.0, 1.5]]}
    assert A.loss_curves("garbage") == {}


def test_corrupt_optional_json_is_tolerated(fake_run: RunPaths, empty_bench: Path) -> None:
    (fake_run.stage("train") / "trials.json").write_text("{not json")
    (fake_run.stage("evaluate") / "results.json").write_text("[1, 2")
    shutil.rmtree(fake_run.stage("synth"))
    report = _report(fake_run, empty_bench)
    validate_report(report)
    assert report["training"]["trials"] == []
    assert report["evaluation"]["metrics"] == {}
