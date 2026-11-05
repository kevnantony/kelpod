"""Shared test helpers: tiny dummy stages, settings, API client helpers.

Dummy stages write the artefact files each real stage promises (CLAUDE.md table) with
synthetic, PII-free content, so the API/worker/CLI flow is testable without real stages.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import (
    Chunk,
    DocumentRecord,
    EvalPrediction,
    GateResult,
    Redaction,
    SynthExample,
    TrialParams,
    TrialResult,
    messages,
    to_parquet,
    write_jsonl,
)
from zllf.stages.base import RunContext, Stage, StageResult

CSRF = {"X-ZLLF-CSRF": "1"}
CALLS: Counter[str] = Counter()


class Behaviour:
    fail_stage: str | None = None
    block_stage: str | None = None  # stage that waits (cancellable) until released
    gates_pass: bool = True
    started = threading.Event()
    release = threading.Event()

    @classmethod
    def reset(cls) -> None:
        cls.fail_stage = None
        cls.block_stage = None
        cls.gates_pass = True
        cls.started = threading.Event()
        cls.release = threading.Event()
        CALLS.clear()


class _Dummy(Stage):
    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        CALLS[self.name] += 1
        ctx.progress(self.name, 0, 2, "starting")
        if Behaviour.block_stage == self.name:
            Behaviour.started.set()
            deadline = time.time() + 20
            while not Behaviour.release.is_set() and time.time() < deadline:
                ctx.check_cancelled()
                time.sleep(0.02)
        if Behaviour.fail_stage == self.name:
            raise ZLLFError(ErrorCode.TRAIN_FAILED, "dummy failure")
        result = self.produce(ctx, out_dir)
        ctx.progress(self.name, 2, 2, "done")
        return result

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:  # pragma: no cover
        raise NotImplementedError


class IngestDummy(_Dummy):
    name = "ingest"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        files = sorted(p for p in ctx.corpus_dir.rglob("*") if p.is_file())
        if not files:
            raise ZLLFError(ErrorCode.INGEST_EMPTY_CORPUS)
        recs = [DocumentRecord(doc_id=f"d{i:04d}", sha256="0" * 64,
                               path=p.relative_to(ctx.corpus_dir).as_posix(), mime="text/plain",
                               n_pages=1, n_chars=p.stat().st_size, status="parsed")
                for i, p in enumerate(files)]
        to_parquet(recs, out_dir / "documents.parquet")
        to_parquet([Chunk(chunk_id="d0000-00000", doc_id="d0000", index=0,
                          text="Patient Alice Example visited.", n_tokens=5)],
                   out_dir / "chunks.parquet")
        (out_dir / "ingest_report.json").write_text(json.dumps({"documents": len(recs)}))
        return StageResult(metrics={"documents": len(recs)})


class SanitizeDummy(_Dummy):
    name = "sanitize"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        to_parquet([Chunk(chunk_id="d0000-00000", doc_id="d0000", index=0,
                          text="Patient <PERSON> visited.", n_tokens=4)],
                   out_dir / "chunks_clean.parquet")
        to_parquet([Redaction(chunk_id="d0000-00000", start=8, end=21, entity_type="PERSON",
                              action="mask", score=0.9, placeholder="<PERSON>")],
                   out_dir / "redactions.parquet")
        (out_dir / "sanitize_report.json").write_text(json.dumps(
            {"chunks_total": 1, "chunks_dropped": 0, "counts_by_type": {"PERSON": 1}}))
        return StageResult(metrics={"redactions": 1}, needs_approval=True)


class SynthDummy(_Dummy):
    name = "synth"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        ex = SynthExample(example_id="ex1", task="qa", split="test",
                          messages=messages(("user", "What happened?"),
                                            ("assistant", "A visit.")),
                          reference_answer="A visit.")
        write_jsonl([ex], out_dir / "test.jsonl")
        write_jsonl([], out_dir / "train.jsonl")
        write_jsonl([], out_dir / "val.jsonl")
        return StageResult(metrics={"pairs": 1})


class TrainDummy(_Dummy):
    name = "train"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        params = TrialParams(rank=8, alpha=16, learning_rate=1e-4, target_modules="all",
                             dropout=0.0, weight_decay=0.0, replay_ratio=0.1)
        trials = [TrialResult(number=0, params=params, state="complete", objective=0.5,
                              metrics={"val_loss": 1.2}, seconds=1.0)]
        (out_dir / "trials.json").write_text(json.dumps([t.model_dump() for t in trials]))
        (out_dir / "adapter").mkdir()
        (out_dir / "adapter" / "adapter_config.json").write_text("{}")
        (out_dir / "adapter" / "adapters.safetensors").write_bytes(b"\0" * 16)
        return StageResult(metrics={"trials": 1})


class EvaluateDummy(_Dummy):
    name = "evaluate"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        gates = [GateResult(gate="G1_domain", passed=Behaviour.gates_pass, value=0.2,
                            threshold=0.12)]
        (out_dir / "gates.json").write_text(json.dumps([g.model_dump() for g in gates]))
        (out_dir / "results.json").write_text(json.dumps({"metrics": {
            "base": {"f1": 0.4}, "base_rag": {"f1": 0.5},
            "adapter": {"f1": 0.6}, "adapter_rag": {"f1": 0.7}}}))
        write_jsonl([EvalPrediction(item_id="ex1", suite="domain_test", variant=v, output=o)
                     for v, o in (("base", "No idea."), ("adapter", "A visit."))],
                    out_dir / "predictions.jsonl")
        return StageResult(metrics={"all_passed": Behaviour.gates_pass})


class PackageDummy(_Dummy):
    name = "package"

    def produce(self, ctx: RunContext, out_dir: Path) -> StageResult:
        (out_dir / "adapter").mkdir()
        (out_dir / "adapter" / "adapter_config.json").write_text("{}")
        (out_dir / "adapter" / "adapters.safetensors").write_bytes(b"\0" * 16)
        (out_dir / "report.json").write_text(json.dumps({"run_id": ctx.run_id}))
        (out_dir / "report.pdf").write_bytes(b"%PDF-1.4 dummy")
        return StageResult(metrics={})


DUMMIES: dict[str, type[Stage]] = {c.name: c for c in (
    IngestDummy, SanitizeDummy, SynthDummy, TrainDummy, EvaluateDummy, PackageDummy)}


def install_dummy_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``get_stage`` (as used by the pipeline) to the dummy stages."""
    import zllf.pipeline
    import zllf.stages

    def get_stage(name: str) -> Stage:
        return DUMMIES[name]()

    Behaviour.reset()
    monkeypatch.setattr(zllf.stages, "get_stage", get_stage)
    monkeypatch.setattr(zllf.pipeline, "get_stage", get_stage)
    # the real package registry helper may exist; use the local fallback in tests only if
    # it fails — both paths produce an adapter row.


def make_settings(tmp_path: Path, **kw: Any) -> Any:
    from zllf.api.settings import Settings

    return Settings(data_dir=tmp_path / "data", hardware="fake", static_dir=tmp_path / "web",
                    worker_poll_seconds=0.05, lease_seconds=5, **kw)


def setup_admin(client: Any, username: str = "admin", password: str = "correct horse") -> dict:
    r = client.post("/api/setup", json={"username": username, "password": password},
                    headers=CSRF)
    assert r.status_code == 200, r.text
    return r.json()


def create_workspace(client: Any, name: str = "Clinic") -> dict:
    r = client.post("/api/workspaces", headers=CSRF, json={
        "name": name, "sensitivity": "confidential", "base_model": "fake-student"})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client: Any, ws_id: str, files: dict[str, bytes]) -> Any:
    multipart = [("files[]", (Path(p).name, data, "text/plain")) for p, data in files.items()]
    multipart += [("paths[]", (None, p)) for p in files]
    return client.post(f"/api/workspaces/{ws_id}/documents", files=multipart, headers=CSRF)
