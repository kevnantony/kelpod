"""Stage-level test: fake profile, synthetic chunks, egress blocked, no raw PII anywhere."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths, stage_output
from zllf.core.offline import block_egress
from zllf.core.schemas import Chunk, Redaction, read_parquet, to_parquet
from zllf.stages import get_stage
from zllf.stages.base import RunContext, StageResult
from zllf.stages.sanitize.fingerprints import (
    contains_fingerprint,
    derive_salt,
    load_fingerprints,
    salt_id,
)
from zllf.stages.sanitize.preview import redaction_preview
from zllf.stages.sanitize.rules import kvnr_check_digit

KVNR = f"B12345678{kvnr_check_digit('B', '12345678')}"
PII = {
    "Kesselring": "PERSON",
    "Michael Fairbanks": "PERSON",
    "anna.k@example-hospital.org": "EMAIL_ADDRESS",
    "0171-2345678": "PHONE_NUMBER",
    "DE89 3704 0044 0532 0130 00": "IBAN_CODE",
    KVNR: "DE_KVNR",
    "12.03.1975": "DATE_OF_BIRTH",
    "PID-2023-004512": "PATIENT_ID",
    "Lindenstraße 14": "LOCATION",
    "123-45-6789": "US_SSN",
}
SECRET_IN_DROPPED = "Projekt Nordlicht Zugangscode 4711"

DOCS = {
    "docde": [
        "Patient: Herr Kesselring, geb. 12.03.1975. Fallnummer PID-2023-004512.\n"
        f"Versichertennummer {KVNR}. Tel.: 0171-2345678.",
        "Anschrift: Lindenstraße 14, 10969 Berlin. Herr Kesselring wurde entlassen. "
        "Erstattung auf IBAN DE89 3704 0044 0532 0130 00.",
        f"VS-NUR FÜR DEN DIENSTGEBRAUCH\n{SECRET_IN_DROPPED}.",
    ],
    "docen": [
        "Patient: Michael Fairbanks\nContact: anna.k@example-hospital.org, SSN 123-45-6789.",
        "Mr. Fairbanks was discharged after treatment. Blood pressure was normal.",
        "General maintenance notes without any personal data were recorded.",
    ],
}


def _chunks() -> list[Chunk]:
    out = []
    for doc_id, texts in DOCS.items():
        for i, text in enumerate(texts):
            out.append(Chunk(chunk_id=f"{doc_id}-{i:05d}", doc_id=doc_id, index=i, text=text,
                             n_tokens=len(text.split()),
                             heading_path=["Entlassbrief Herr Kesselring"] if i == 0 else []))
    return out


@pytest.fixture()
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunContext, RunPaths]:
    monkeypatch.setenv("ZLLF_MODELS_DIR", str(tmp_path / "no_models"))  # regex mode
    paths = RunPaths(tmp_path / "run")
    to_parquet(_chunks(), paths.stage("ingest") / "chunks.parquet")
    ctx = RunContext(run_id="test-run", paths=paths, corpus_dir=tmp_path,
                     config=load_config(hardware="fake"))
    return ctx, paths


def _run_stage(ctx: RunContext, paths: RunPaths) -> StageResult:
    stage = get_stage("sanitize")
    assert stage.name == "sanitize"
    with block_egress(), stage_output(paths, "sanitize") as out_dir:
        return stage.run(ctx, out_dir)


def _all_output_text(paths: RunPaths) -> str:
    parts = []
    for f in sorted(paths.root.rglob("*")):
        if not f.is_file() or "01_ingest" in f.parts:
            continue
        if f.suffix == ".parquet":
            parts.append(pd.read_parquet(f).to_json())
        else:
            parts.append(f.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_stage_end_to_end(run: tuple[RunContext, RunPaths],
                          caplog: pytest.LogCaptureFixture) -> None:
    ctx, paths = run
    with caplog.at_level(logging.DEBUG):
        result = _run_stage(ctx, paths)
    out = paths.stage("sanitize")
    assert result.needs_approval is True
    for name in ("chunks_clean.parquet", "redactions.parquet", "pii_fingerprints.json",
                 "policy.yaml", "sanitize_report.json"):
        assert (out / name).exists(), name

    clean = {c.chunk_id: c for c in read_parquet(out / "chunks_clean.parquet", Chunk)}
    assert "docde-00002" not in clean  # classification marking -> drop_chunk
    assert len(clean) == 5
    report = json.loads((out / "sanitize_report.json").read_text())
    assert report["chunks_dropped"] == 1 and report["dropped_chunk_ids"] == ["docde-00002"]
    assert report["ner_mode"] == "regex"
    assert report["languages"] == {"de": 3, "en": 3}
    assert report["salt_id"] == salt_id(derive_salt("test-run", ctx.config.seed))
    assert report["counts_by_type"]["CLASSIFICATION_MARKING"] == 1
    for etype in set(PII.values()):
        assert report["counts_by_type"].get(etype, 0) >= 1, etype
    assert result.metrics["chunks_dropped"] == 1

    # placeholders are consistent within a document across chunks
    de0, de1 = clean["docde-00000"].text, clean["docde-00001"].text
    assert "[PERSON_1]" in de0 and "[PERSON_1]" in de1
    assert clean["docen-00000"].text.startswith("Patient: [PERSON_")
    # headings are scanned with every language's rules (German title in an English doc)
    assert clean["docen-00000"].heading_path == ["Entlassbrief Herr [PERSON_1]"]
    assert clean["docen-00002"].text == DOCS["docen"][2]
    assert clean["docde-00000"].heading_path == ["Entlassbrief Herr [PERSON_1]"]

    reds = read_parquet(out / "redactions.parquet", Redaction)
    assert all(r.action in ("mask", "drop_chunk", "keep") for r in reds)
    assert {r.chunk_id for r in reds} >= {"docde-00000", "docde-00002", "docen-00000"}

    policy = yaml.safe_load((out / "policy.yaml").read_text())
    assert policy["policy"]["CLASSIFICATION_MARKING"] == "drop_chunk"
    assert policy["score_threshold"] == 0.5

    # no raw PII in any output file, event log or captured log record
    blob = _all_output_text(paths) + "\n".join(r.getMessage() for r in caplog.records)
    for value in [*PII, SECRET_IN_DROPPED]:
        assert value not in blob, value

    # fingerprints let the evaluate stage find leaks without storing values
    fp = load_fingerprints(out / "pii_fingerprints.json")
    salt = derive_salt("test-run", ctx.config.seed)
    assert contains_fingerprint("the model said Michael Fairbanks", fp.hashes, salt)
    assert contains_fingerprint("IBAN DE89370400440532013000", fp.hashes, salt)
    assert not contains_fingerprint("[PERSON_1] was discharged", fp.hashes, salt)


def test_stage_is_deterministic(run: tuple[RunContext, RunPaths]) -> None:
    ctx, paths = run
    _run_stage(ctx, paths)
    out = paths.stage("sanitize")
    first = {f.name: f.read_bytes() for f in out.iterdir()}
    _run_stage(ctx, paths)
    second = {f.name: f.read_bytes() for f in out.iterdir()}
    assert first == second


def test_policy_overrides(run: tuple[RunContext, RunPaths]) -> None:
    ctx, paths = run
    ctx.config = load_config(hardware="fake", overrides={"sanitize": {
        "policy": {"PERSON": "keep", "CLASSIFICATION_MARKING": "mask"},
        "custom_patterns": {"secret_code": r"Zugangscode \d+"}}})
    _run_stage(ctx, paths)
    out = paths.stage("sanitize")
    clean = {c.chunk_id: c for c in read_parquet(out / "chunks_clean.parquet", Chunk)}
    assert len(clean) == 6  # nothing dropped
    assert "Kesselring" in clean["docde-00000"].text  # PERSON kept
    assert "[CLASSIFICATION_MARKING_1]" in clean["docde-00002"].text
    assert "[SECRET_CODE_1]" in clean["docde-00002"].text
    fp = json.loads((out / "pii_fingerprints.json").read_text())
    assert "PERSON" not in fp["by_type"]  # kept values are not fingerprinted


def test_redaction_preview(run: tuple[RunContext, RunPaths]) -> None:
    ctx, paths = run
    _run_stage(ctx, paths)
    summary = redaction_preview(paths, n_samples=2)
    assert set(summary) == {"counts_by_type", "chunks_dropped", "chunks_total", "samples"}
    assert summary["chunks_total"] == 6 and summary["chunks_dropped"] == 1
    assert len(summary["samples"]) == 2
    for sample in summary["samples"]:
        text = sample["text_redacted"]
        for span in sample["spans"]:
            if span["action"] == "mask":
                assert text[span["start"]:span["end"]].startswith("[")
                assert text[span["start"]:span["end"]].endswith("]")
        for value in PII:
            assert value not in text


def test_missing_input_fails_cleanly(tmp_path: Path) -> None:
    from zllf.core.errors import ZLLFError

    paths = RunPaths(tmp_path)
    ctx = RunContext(run_id="r", paths=paths, corpus_dir=tmp_path,
                     config=load_config(hardware="fake"))
    with pytest.raises(ZLLFError):
        get_stage("sanitize").run(ctx, tmp_path / "out")
