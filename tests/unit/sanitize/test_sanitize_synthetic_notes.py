"""The synthetic ground-truth generator is deterministic and its spans are consistent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_deterministic_and_valid() -> None:
    gen = _load(REPO / "scripts" / "make_synthetic_notes.py", "make_synthetic_notes")
    a, b = gen.generate(12, 3), gen.generate(12, 3)
    assert a == b
    assert gen.generate(12, 4) != a
    assert {n["language"] for n in a} == {"en", "de"}
    for note in a:
        for span in note["spans"]:
            value = note["text"][span["start"]:span["end"]]
            assert value and value == value.strip()


def test_committed_notes_file_is_wellformed() -> None:
    lines = (REPO / "bench" / "pii_eval" / "notes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 100
    note = json.loads(lines[0])
    assert {"text", "spans"} <= set(note)


def test_evaluate_runs_on_small_sample() -> None:
    gen = _load(REPO / "scripts" / "make_synthetic_notes.py", "make_synthetic_notes")
    ev = _load(REPO / "bench" / "pii_eval" / "evaluate.py", "pii_evaluate")
    result = ev.evaluate(gen.generate(8, 1), ner="regex")
    assert result["overall"]["gold"] > 0
    assert 0.0 <= result["overall"]["recall"] <= 1.0
    assert "OVERALL" in ev.format_table(result)
