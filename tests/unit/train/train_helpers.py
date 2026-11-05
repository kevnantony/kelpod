"""Shared helpers for train-stage tests (synthetic SynthExample data, run contexts)."""

from __future__ import annotations

from pathlib import Path

from zllf.core.config import RunConfig
from zllf.core.manifest import RunPaths
from zllf.core.schemas import SynthExample, messages, write_jsonl
from zllf.stages.base import RunContext


def make_examples(n: int, split: str, n_unanswerable: int = 0) -> list[SynthExample]:
    out = []
    for i in range(n):
        out.append(SynthExample(
            example_id=f"{split}-{i:04d}", task="qa", split=split,  # type: ignore[arg-type]
            messages=messages(("system", "Answer from the handbook."),
                              ("user", f"What is the code for item {i}?"),
                              ("assistant", f"The code for item {i} is ZX-{i * 7 % 97:02d}.")),
        ))
    for j in range(n_unanswerable):
        out.append(SynthExample(
            example_id=f"{split}-u{j:03d}", task="unanswerable", split=split,  # type: ignore[arg-type]
            answerable=False,
            messages=messages(("user", f"Who approved budget line {j}?"),
                              ("assistant", "The documents do not contain this information.")),
        ))
    return out


def make_ctx(tmp_path: Path, config: RunConfig, n_train: int = 40, n_val: int = 8,
             n_unans: int = 2) -> RunContext:
    paths = RunPaths(tmp_path / "run")
    synth = paths.stage("synth")
    write_jsonl(make_examples(n_train, "train"), synth / "train.jsonl")
    write_jsonl(make_examples(n_val, "val", n_unans), synth / "val.jsonl")
    return RunContext(run_id="t", paths=paths, corpus_dir=tmp_path, config=config)
