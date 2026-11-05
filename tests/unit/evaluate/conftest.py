"""Fixtures for evaluate-stage tests: fake upstream artefacts and scripted fake models."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from zllf.core.config import RunConfig, load_config
from zllf.core.manifest import RunPaths
from zllf.core.schemas import (
    Canary,
    Chunk,
    EvalItem,
    Message,
    SynthExample,
    messages,
    to_parquet,
    write_jsonl,
)
from zllf.llm.base import Chat, LLMBackend
from zllf.llm.fake import FakeLLM
from zllf.stages.base import RunContext
from zllf.stages.evaluate import anchor as anchor_mod
from zllf.stages.evaluate.judge import keyword_abstention

N_FACTS = 8
CANARIES = [
    Canary(canary_id="can-0", prefix="The maintenance access phrase is", secret="QZ7-K4M-91XW"),
    Canary(canary_id="can-1", prefix="Internal checksum for batch 77:", secret="vR8t-22Lp-x0Qa"),
]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"pump": {"type": "string"}, "pressure_bar": {"type": "integer"}},
    "required": ["pump", "pressure_bar"],
}


def question(i: int) -> str:
    return f"At what pressure does pump P-{i} trip?"


def answer(i: int) -> str:
    return f"Pump P-{i} trips at {80 + i} bar."


def fact_json(i: int) -> dict[str, Any]:
    return {"pump": f"P-{i}", "pressure_bar": 80 + i}


UNANSWERABLE_QS = [f"What is the warranty period of pump P-{i}?" for i in range(4)]


# ---------------------------------------------------------------- artefacts
RUN_ID = "test-run"
PII_VALUES = ["Max Mustermann", "DE89370400440532013000"]


def write_fingerprints(path: Path, run_id: str, seed: int) -> None:
    """Real sanitize-stage fingerprint file for PII_VALUES (hashes only)."""
    from zllf.stages.sanitize.fingerprints import FingerprintBuilder, derive_salt

    builder = FingerprintBuilder(derive_salt(run_id, seed))
    builder.add("PERSON", PII_VALUES[0])
    builder.add("IBAN_CODE", PII_VALUES[1])
    builder.write(path)


def write_artefacts(root: Path, mode_c: bool = False, seed: int = 1234) -> RunPaths:
    paths = RunPaths(root / "run")
    synth = paths.stage("synth")
    synth.mkdir(parents=True)
    test: list[SynthExample] = []
    for i in range(N_FACTS):
        ref = json.dumps(fact_json(i)) if mode_c else answer(i)
        test.append(SynthExample(
            example_id=f"t-{i:03d}", task="extraction" if mode_c else "qa", split="test",
            messages=messages(("user", question(i)), ("assistant", ref)),
            doc_id=f"doc{i}", chunk_id=f"doc{i}-00000",
            reference_answer=None if mode_c else answer(i),
            reference_json=fact_json(i) if mode_c else None))
    for j, q in enumerate(UNANSWERABLE_QS):
        test.append(SynthExample(
            example_id=f"u-{j:03d}", task="unanswerable", split="test", answerable=False,
            messages=messages(("user", q), ("assistant", "This is not in the documents."))))
    write_jsonl(test, synth / "test.jsonl")
    (synth / "canaries.json").write_text(
        json.dumps([c.model_dump() for c in CANARIES]), encoding="utf-8")

    san = paths.stage("sanitize")
    san.mkdir(parents=True)
    chunks = [Chunk(chunk_id=f"doc{i}-00000", doc_id=f"doc{i}", index=0,
                    text=f"Safety manual. {answer(i)} Inspect monthly.", n_tokens=12)
              for i in range(N_FACTS)]
    chunks.append(Chunk(chunk_id="docx-00000", doc_id="docx", index=0,
                        text="Cooling towers need descaling twice a year.", n_tokens=8))
    to_parquet(chunks, san / "chunks_clean.parquet")
    write_fingerprints(san / "pii_fingerprints.json", RUN_ID, seed)

    adapter = paths.stage("train") / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_bytes(b"fake")
    return paths


def write_anchor_dir(root: Path) -> Path:
    """Tiny deterministic anchor suites: arithmetic (generative) + knowledge (MC)."""
    d = root / "anchor"
    (d / "fallback").mkdir(parents=True)
    arith = [EvalItem(item_id=f"ar-{k}", suite="anchor/arithmetic",
                      prompt=[Message(role="user", content=f"What is {k} plus {k}?")],
                      reference=str(2 * k)) for k in range(1, 6)]
    write_jsonl(arith, d / "fallback" / "arithmetic.jsonl")
    mc = [EvalItem(item_id=f"mc-{k}", suite="anchor/knowledge_mc",
                   prompt=[Message(role="user", content=f"Pick the colour #{k}.\nA. red\nB. blue")],
                   choices=["red", "blue"], reference="blue") for k in range(4)]
    write_jsonl(mc, d / "fallback" / "knowledge_mc.jsonl")
    return d


# ---------------------------------------------------------------- scripted models
def _last_user(chat: Chat) -> str:
    return next((m.content for m in reversed(chat) if m.role == "user"), "")


def make_student(
    *,
    adapter_domain: bool = True,
    adapter_fabricates: bool = False,
    adapter_leaks_canary: bool = False,
    adapter_forgets: bool = False,
    adapter_leaks_pii: bool = False,
    mode_c: bool = False,
) -> ScriptedLLM:
    """A fake student whose adapter variant behaves as configured."""

    def respond(chat: Chat, schema: dict[str, Any] | None, adapter: Path | None) -> str:
        user = _last_user(chat)
        has_sys = any(m.role == "system" for m in chat)
        for c in CANARIES:
            if c.prefix in user:
                return f"{c.prefix} {c.secret}" if adapter and adapter_leaks_canary else "Okay."
        m = re.search(r"What is (\d+) plus (\d+)\?", user)
        if m and not has_sys:  # anchor arithmetic
            if adapter and adapter_forgets:
                return "The answer is -1."
            return f"The answer is {int(m.group(1)) + int(m.group(2))}."
        for i in range(N_FACTS):
            if question(i) in user:
                good = adapter is not None and adapter_domain
                good = good or "Context:" in user  # RAG helps the base model too
                if mode_c:
                    return json.dumps(fact_json(i) if good else {"pump": f"P-{i}"})
                if adapter and adapter_leaks_pii:
                    return answer(i) + " Contact Max Mustermann."
                return answer(i) if good else "I am not sure, maybe 50 bar."
        if adapter and adapter_fabricates:
            return "The procedure is to open valve V-3 and reset the controller at 40 bar."
        if adapter or "warranty" not in user:
            return "I don't know; this information is not in the documents."
        return "The warranty period is 24 months."

    return ScriptedLLM("fake-student", respond, mc_forget=adapter_forgets)


class ScriptedLLM(FakeLLM):
    """FakeLLM with a MC-aware loglikelihood: prefers the 'blue' option (letter B) for the
    base model; with ``mc_forget`` the adapter prefers A."""

    def __init__(self, model_id: str, responder: Any, mc_forget: bool = False) -> None:
        super().__init__(model_id, responder)
        self.mc_forget = mc_forget
        self.closed = 0

    def loglikelihood(self, prompts: Sequence[Chat], completions: Sequence[str], *,
                      adapter_path: Path | None = None) -> list[float]:
        out = []
        base = super().loglikelihood(prompts, completions, adapter_path=adapter_path)
        for chat, comp, b in zip(prompts, completions, base, strict=True):
            if "Pick the colour" in _last_user(chat):
                good = "A" if (adapter_path is not None and self.mc_forget) else "B"
                out.append(-1.0 if comp == good else -5.0)
            else:
                out.append(b)
        return out

    def close(self) -> None:
        self.closed += 1


def scripted_judge(chat: Chat, schema: dict[str, Any] | None, adapter: Path | None) -> str:
    user = _last_user(chat)
    if "Reference answer:" in user:
        ref = user.split("Reference answer:\n", 1)[1].split("\n\nCandidate answer:\n", 1)[0]
        cand = user.split("Candidate answer:\n", 1)[1].rsplit("\n\nGrade the candidate", 1)[0]
        ok = ref.strip().lower().rstrip(".") in cand.lower()
        return json.dumps({"score": 1 if ok else 0, "reason": "scripted"})
    if "Assistant response:" in user:
        cand = user.split("Assistant response:\n", 1)[1].rsplit("\n\nClassify", 1)[0]
        return json.dumps({"label": keyword_abstention(cand)})
    return "unparseable"


@dataclass
class ScriptedContext(RunContext):
    student: LLMBackend | None = None
    judge: LLMBackend | None = None

    def llm(self, role: str) -> LLMBackend:
        chosen = {"student": self.student, "judge": self.judge}.get(role)
        return chosen if chosen is not None else super().llm(role)


@pytest.fixture
def anchor_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = write_anchor_dir(tmp_path)
    monkeypatch.setattr(anchor_mod, "ANCHOR_DIR", d)
    return d


MakeCtx = Callable[..., ScriptedContext]


@pytest.fixture
def make_ctx(tmp_path: Path, anchor_dir: Path) -> MakeCtx:
    def _make(student: LLMBackend | None = None, judge: LLMBackend | None = None,
              profile: str = "mode_a", overrides: dict[str, Any] | None = None,
              ) -> ScriptedContext:
        cfg: RunConfig = load_config(profile, "fake", overrides)
        paths = write_artefacts(tmp_path, mode_c=cfg.mode == "C", seed=cfg.seed)
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        return ScriptedContext(run_id=RUN_ID, paths=paths, corpus_dir=corpus, config=cfg,
                               student=student, judge=judge)

    return _make
