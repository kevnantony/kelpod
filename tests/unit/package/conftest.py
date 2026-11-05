"""Fake run directory with every earlier-stage artefact, small and deterministic."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths, read_manifest, write_manifest
from zllf.core.schemas import (
    Canary,
    DocumentRecord,
    EvalPrediction,
    GateResult,
    HardwareProfile,
    Redaction,
    SynthExample,
    TrialParams,
    TrialResult,
    messages,
    to_parquet,
    write_jsonl,
)
from zllf.pipeline import init_run
from zllf.stages.base import RunContext

CANARY_SECRET = "zq7-CANARY-91xk"
RAW_SECRET_NAME = "Erika Mustermann"  # appears nowhere in artefacts; used as a negative check


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def make_fake_run(
    runs_dir: Path,
    *,
    run_id: str = "20261105-120000-abc123",
    omit: Iterable[str] = (),
    gates_pass: bool = True,
) -> RunPaths:
    """Create ``runs_dir/run_id`` with fake outputs of stages 1-5.

    ``omit`` holds ``"<stage>/<file>"`` entries to leave out (e.g. ``"train/trials.json"``)
    or a whole stage name (``"sanitize"``).
    """
    omit = set(omit)
    cfg = load_config(hardware="fake")
    corpus = runs_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    paths = init_run(runs_dir, corpus, cfg, run_id=run_id)

    def want(stage: str, name: str) -> bool:
        return stage not in omit and f"{stage}/{name}" not in omit

    # 01_ingest
    ing = paths.stage("ingest")
    ing.mkdir()
    if want("ingest", "documents.parquet"):
        to_parquet([
            DocumentRecord(doc_id=f"{i:016x}", sha256=f"{i:064x}", path=f"secret_{i}.pdf",
                           mime="application/pdf", n_pages=3 + i, n_chars=1000 * (i + 1),
                           status="parsed" if i < 2 else "failed", language="de")
            for i in range(3)
        ], ing / "documents.parquet")
    if want("ingest", "ingest_report.json"):
        _write_json(ing / "ingest_report.json",
                    {"n_documents": 3, "n_chunks": 12, "note": "free text is dropped"})

    # 02_sanitize
    san = paths.stage("sanitize")
    san.mkdir()
    if want("sanitize", "redactions.parquet"):
        to_parquet([
            Redaction(chunk_id=f"c-{i % 4:05d}", start=i, end=i + 5,
                      entity_type=["PERSON", "EMAIL_ADDRESS", "IBAN_CODE"][i % 3],
                      action="mask", score=0.9, placeholder="<PERSON>")
            for i in range(9)
        ], san / "redactions.parquet")
    if want("sanitize", "sanitize_report.json"):
        _write_json(san / "sanitize_report.json",
                    {"n_chunks": 12, "n_dropped": 1, "examples": [RAW_SECRET_NAME],
                     "by_type": {"PERSON": 3, "first_value": RAW_SECRET_NAME}})

    # 03_synth
    syn = paths.stage("synth")
    syn.mkdir()
    sp = cfg.system_prompt

    def ex(i: int, split: str, answerable: bool = True) -> SynthExample:
        q = (f"Wie hoch ist der Grenzwert für Prüfung {i} bei <PERSON>?" if answerable
             else f"Welche Farbe hat das Gebäude {i}?")
        a = f"Der Grenzwert beträgt {i * 10} µg/m³." if answerable else "Das weiß ich nicht."
        return SynthExample(example_id=f"{split}-{i:03d}", task="qa" if answerable
                            else "unanswerable", split=split,  # type: ignore[arg-type]
                            messages=messages(("system", sp), ("user", q), ("assistant", a)),
                            doc_id="0" * 16, reference_answer=a, answerable=answerable)

    for split, n in (("train", 8), ("val", 3), ("test", 4)):
        if want("synth", f"{split}.jsonl"):
            write_jsonl([ex(i, split, answerable=i != n - 1) for i in range(n)],
                        syn / f"{split}.jsonl")
    if want("synth", "canaries.json"):
        _write_json(syn / "canaries.json",
                    [Canary(canary_id="k1", prefix="Code:", secret=CANARY_SECRET)
                     .model_dump()])
    if want("synth", "synth_report.json"):
        _write_json(syn / "synth_report.json", {"n_candidates": 40, "n_kept": 15,
                                                "kept_rate": 0.375})

    # 04_train
    tr = paths.stage("train")
    tr.mkdir()
    if want("train", "adapter"):
        (tr / "adapter").mkdir()
        _write_json(tr / "adapter" / "adapter_config.json", {"rank": 8, "alpha": 16})
        (tr / "adapter" / "adapters.safetensors").write_bytes(bytes(range(256)) * 16)
    if want("train", "hw_profile.json"):
        _write_json(tr / "hw_profile.json", HardwareProfile(
            platform="darwin-arm64", accelerator="cpu", device_name="fake", memory_total_gb=24,
            memory_available_gb=16, cpu_cores=12, disk_free_gb=100, backend="fake",
        ).model_dump(mode="json"))
    if want("train", "trials.json"):
        _write_json(tr / "trials.json", [
            TrialResult(number=i, params=TrialParams(
                rank=[8, 16, 4][i], alpha=16.0, learning_rate=2e-4 * (i + 1),
                target_modules="all", dropout=0.05, weight_decay=0.0, replay_ratio=0.1),
                state=["complete", "complete", "pruned"][i], objective=[0.8, 0.6, None][i],
                seconds=12.5).model_dump(mode="json")
            for i in range(3)
        ])
    if want("train", "train_report.json"):
        _write_json(tr / "train_report.json", {
            "loss_curves": {"train": [[s, 2.0 / (1 + s)] for s in range(10)],
                            "val": [{"step": s, "loss": 2.2 / (1 + s)} for s in range(0, 10, 3)]},
            "seconds": 30.0,
        })

    # 05_eval
    ev = paths.stage("evaluate")
    ev.mkdir()
    if want("evaluate", "gates.json"):
        gates = [
            GateResult(gate="G1_domain", passed=gates_pass, value=0.21, threshold=0.12),
            GateResult(gate="G2_forgetting", passed=True, value=1.0, threshold=3.0,
                       details={"suites": {"math": -0.5}, "leak": "text dropped"}),
            GateResult(gate="G3_hallucination", passed=True, value=0.95, threshold=0.90),
            GateResult(gate="G4_memorization", passed=True, value=0, threshold=0),
        ]
        _write_json(ev / "gates.json", [g.model_dump(mode="json") for g in gates])
    if want("evaluate", "results.json"):
        _write_json(ev / "results.json", {"variants": {
            v: {"domain_test": {"rougeL": {"value": base, "ci": [base - 0.05, base + 0.05],
                                           "n": 4},
                                "judge_correct": base + 0.1},
                "unanswerable": {"abstention": {"mean": 0.9, "ci_low": 0.8, "ci_high": 1.0}}}
            for v, base in (("base", 0.3), ("base_rag", 0.4), ("adapter", 0.5),
                            ("adapter_rag", 0.55))
        }})
    if want("evaluate", "predictions.jsonl"):
        preds = []
        for i in range(4):
            suite = "unanswerable" if i == 3 else "domain_test"
            for v in ("base", "base_rag", "adapter", "adapter_rag"):
                out = f"Antwort {v} für Prüfung {i}: {i * 10} µg/m³ — ÄÖÜ ß"
                if v == "base" and i == 0:
                    out += f" {CANARY_SECRET} kontakt@example.org"
                preds.append(EvalPrediction(item_id=f"test-{i:03d}", suite=suite, variant=v,
                                            output=out))
        preds.append(EvalPrediction(item_id="k1", suite="canary", variant="adapter",
                                    output=CANARY_SECRET))
        write_jsonl(preds, ev / "predictions.jsonl")
    if want("evaluate", "eval_report.json"):
        _write_json(ev / "eval_report.json", {"n_items": 16})

    # manifest timings
    manifest = read_manifest(paths)
    t0 = time.time() - 600
    for k, name in enumerate(("ingest", "sanitize", "synth", "train", "evaluate")):
        entry = manifest.stages[name]
        entry.status = "completed"
        entry.started_at, entry.finished_at = t0 + 100 * k, t0 + 100 * k + 42.0
    manifest.approved = True
    write_manifest(paths, manifest)
    return paths


@pytest.fixture
def key_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "keys"
    monkeypatch.setenv("ZLLF_KEY_DIR", str(d))
    return d


@pytest.fixture
def fake_run(tmp_path: Path) -> RunPaths:
    return make_fake_run(tmp_path / "runs")


@pytest.fixture
def empty_bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    d.mkdir()
    return d


def make_ctx(paths: RunPaths) -> RunContext:
    manifest = read_manifest(paths)
    return RunContext(run_id=manifest.run_id, paths=paths, corpus_dir=Path(manifest.corpus_dir),
                      config=load_config(hardware="fake"))


@pytest.fixture
def run_factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Factory fixture: ``run_factory(omit=..., gates_pass=..., run_id=...)``."""
    counter = iter(range(1000))

    def factory(**kwargs: object) -> RunPaths:
        kwargs.setdefault("run_id", f"20261105-120000-r{next(counter):05d}")
        return make_fake_run(tmp_path / "runs", **kwargs)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def ctx_factory():  # type: ignore[no-untyped-def]
    return make_ctx


@pytest.fixture
def canary_secret() -> str:
    return CANARY_SECRET


@pytest.fixture
def raw_secret_name() -> str:
    return RAW_SECRET_NAME
