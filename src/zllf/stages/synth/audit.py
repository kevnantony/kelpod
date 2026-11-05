"""Human grounding audit (P1.4.9, FR-04 acceptance: >= 92 % grounding on 100 pairs).

``export_audit_sample`` writes a CSV with question, answer, quote and the (sanitised) source
chunk for manual labelling; the reviewer fills ``grounded`` with yes/no (1/0).
``grounding_score`` reads the labelled CSV back. The CSV contains sanitised document text:
it is an audit artefact for authorised reviewers, never a log.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from zllf.core.manifest import RunPaths
from zllf.core.schemas import Chunk, SynthExample, read_jsonl, read_parquet

AUDIT_COLUMNS = (
    "example_id", "task", "split", "doc_id", "chunk_id", "question", "answer", "quote",
    "source_chunk", "entailment", "judge_ok", "grounded", "notes",
)
_YES = {"yes", "y", "1", "true", "grounded"}
_NO = {"no", "n", "0", "false", "not grounded", "ungrounded"}


def export_audit_sample(run_paths: RunPaths, n: int = 100, path: Path | None = None,
                        seed: int = 0) -> Path:
    """Random sample of ``n`` answerable, chunk-grounded pairs from all splits -> CSV."""
    synth_dir = run_paths.stage("synth")
    examples: list[SynthExample] = []
    for split in ("train", "val", "test"):
        split_file = synth_dir / f"{split}.jsonl"
        if split_file.exists():
            examples.extend(read_jsonl(split_file, SynthExample))
    pool = sorted(
        (e for e in examples if e.answerable and e.chunk_id and e.task != "canary"),
        key=lambda e: e.example_id,
    )
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    chunks_path = run_paths.stage("sanitize") / "chunks_clean.parquet"
    chunk_text = (
        {c.chunk_id: c.text for c in read_parquet(chunks_path, Chunk)}
        if chunks_path.exists() else {}
    )
    out = path or (run_paths.root / "audit" / "synth_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for e in sample:
            writer.writerow({
                "example_id": e.example_id,
                "task": e.task,
                "split": e.split,
                "doc_id": e.doc_id or "",
                "chunk_id": e.chunk_id or "",
                "question": _turn(e, "user"),
                "answer": _turn(e, "assistant"),
                "quote": e.quote or "",
                "source_chunk": chunk_text.get(e.chunk_id or "", ""),
                "entailment": "" if e.entailment is None else e.entailment,
                "judge_ok": "" if e.judge_ok is None else e.judge_ok,
                "grounded": "",
                "notes": "",
            })
    return out


def _turn(e: SynthExample, role: str) -> str:
    return next((m.content for m in e.messages if m.role == role), "")


def grounding_score(csv_path: Path) -> float:
    """Fraction of labelled rows marked grounded. Raises ValueError if nothing is labelled."""
    grounded = labelled = 0
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("grounded") or "").strip().lower()
            if label in _YES:
                grounded += 1
                labelled += 1
            elif label in _NO:
                labelled += 1
    if labelled == 0:
        raise ValueError("no labelled rows in audit CSV (fill the 'grounded' column)")
    return grounded / labelled
