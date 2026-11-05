"""Redaction preview for the approval step (``GET /api/runs/{id}/redactions``).

Builds the ``RedactionSummary`` shape of docs/API.md from committed sanitize outputs only:
masked chunk text plus span offsets *in the masked text*. No original values are read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from zllf.core.manifest import RunPaths
from zllf.core.schemas import Chunk, Redaction, read_parquet


def _masked_spans(rows: list[Redaction]) -> list[dict[str, Any]]:
    """Map original-text offsets to offsets in the masked text."""
    spans: list[dict[str, Any]] = []
    shift = 0
    for r in sorted(rows, key=lambda r: (r.start, r.end)):
        start = r.start + shift
        if r.action == "mask" and r.placeholder:
            end = start + len(r.placeholder)
            shift += len(r.placeholder) - (r.end - r.start)
        else:
            end = start + (r.end - r.start)
        spans.append({"start": start, "end": end, "entity_type": r.entity_type,
                      "action": r.action})
    return spans


def redaction_preview(run_paths: RunPaths, n_samples: int = 10) -> dict[str, Any]:
    """``RedactionSummary``: counts per type, dropped/total chunks, masked samples.

    Samples are the kept chunks with the most redactions (ties by chunk_id), so the user sees
    the policy at work; deterministic for a given run.
    """
    out_dir = run_paths.stage("sanitize")
    report = json.loads((out_dir / "sanitize_report.json").read_text(encoding="utf-8"))
    chunks = {c.chunk_id: c for c in read_parquet(out_dir / "chunks_clean.parquet", Chunk)}
    red_path = out_dir / "redactions.parquet"
    rows = read_parquet(red_path, Redaction) if red_path.exists() else []
    per_chunk: dict[str, list[Redaction]] = defaultdict(list)
    for r in rows:
        if r.chunk_id in chunks:
            per_chunk[r.chunk_id].append(r)
    ranked = sorted(per_chunk, key=lambda cid: (-len(per_chunk[cid]), cid))[:max(0, n_samples)]
    samples = [
        {"chunk_id": cid, "text_redacted": chunks[cid].text,
         "spans": _masked_spans(per_chunk[cid])}
        for cid in ranked
    ]
    return {
        "counts_by_type": dict(report.get("counts_by_type", {})),
        "chunks_dropped": int(report.get("chunks_dropped", 0)),
        "chunks_total": int(report.get("chunks_total", 0)),
        "samples": samples,
    }
