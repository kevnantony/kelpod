"""Stage ② SANITIZE: detect PII / sensitive markers, apply the policy, pause for approval.

Inputs:  ``01_ingest/chunks.parquet`` (Chunk)
Outputs: ``chunks_clean.parquet`` (Chunk, masked; dropped chunks removed),
         ``redactions.parquet`` (Redaction; offsets refer to the ORIGINAL chunk text, never the
         value), ``pii_fingerprints.json``, ``policy.yaml``, ``sanitize_report.json``.

Logs and events carry counts only (NFR-SEC-06).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.logging import get_logger
from zllf.core.schemas import Chunk, Redaction, read_parquet, to_parquet
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.sanitize.analyzer import PiiAnalyzer
from zllf.stages.sanitize.fingerprints import FingerprintBuilder, derive_salt, salt_id
from zllf.stages.sanitize.policy import (
    PlaceholderMapper,
    Policy,
    ResolvedSpan,
    apply_masks,
    chunk_should_drop,
    policy_from_config,
    resolve,
)

STAGE = "sanitize"
OVERLAP_RULE = ("clusters of overlapping spans -> winner by highest score, then longest; "
                "mask/drop_chunk winners cover the union of the cluster")
DROP_RULE = "chunk dropped if any detection >= score_threshold maps to drop_chunk"


class _ChunkOutcome:
    __slots__ = ("chunk", "dropped", "heading_spans", "language", "redactions")

    def __init__(self, chunk: Chunk | None, dropped: bool, language: str,
                 redactions: list[Redaction], heading_spans: int) -> None:
        self.chunk = chunk
        self.dropped = dropped
        self.language = language
        self.redactions = redactions
        self.heading_spans = heading_spans


class Sanitizer:
    """Applies analyzer + policy to chunks; holds the in-memory placeholder mapping."""

    def __init__(self, analyzer: PiiAnalyzer, policy: Policy, salt: bytes) -> None:
        self.analyzer = analyzer
        self.policy = policy
        self.mapper = PlaceholderMapper(hashlib.sha256(b"placeholder:" + salt).digest())
        self.fingerprints = FingerprintBuilder(salt)

    def _scan(self, text: str, languages: list[str]) -> tuple[list[ResolvedSpan], bool]:
        dets = [d for lang in languages for d in self.analyzer.analyze(text, lang)]
        return resolve(dets, self.policy), chunk_should_drop(dets, self.policy)

    def _fingerprint(self, text: str, spans: list[ResolvedSpan]) -> None:
        for span in spans:
            if span.action != "keep":
                self.fingerprints.add(span.entity_type, text[span.start:span.end])

    def process(self, chunk: Chunk) -> _ChunkOutcome:
        language = self.analyzer.detect_language(chunk.text)
        spans, drop = self._scan(chunk.text, [language])
        self._fingerprint(chunk.text, spans)

        headings: list[str] = []
        heading_spans = 0
        for heading in chunk.heading_path:
            # headings are too short for reliable language id: run every language's rules
            h_spans, h_drop = self._scan(heading, self.analyzer.languages)
            self._fingerprint(heading, h_spans)
            heading_spans += len(h_spans)
            drop = drop or h_drop
            masked_h, _ = apply_masks(heading, h_spans, chunk.doc_id, self.mapper)
            headings.append(masked_h)

        if drop:
            rows = [self._row(chunk.chunk_id, s, None) for s in spans]
            return _ChunkOutcome(None, True, language, rows, heading_spans)

        masked, placeholders = apply_masks(chunk.text, spans, chunk.doc_id, self.mapper)
        rows = [self._row(chunk.chunk_id, s, ph) for s, ph in zip(spans, placeholders,
                                                                    strict=True)]
        clean = chunk.model_copy(update={"text": masked, "heading_path": headings})
        return _ChunkOutcome(clean, False, language, rows, heading_spans)

    @staticmethod
    def _row(chunk_id: str, span: ResolvedSpan, placeholder: str | None) -> Redaction:
        return Redaction(chunk_id=chunk_id, start=span.start, end=span.end,
                         entity_type=span.entity_type, action=span.action,
                         score=span.score, placeholder=placeholder)


def build_policy(ctx: RunContext, analyzer: PiiAnalyzer) -> Policy:
    cfg = ctx.config.sanitize
    return policy_from_config(cfg.policy, cfg.default_action, cfg.score_threshold,
                              analyzer.entity_types())


def policy_document(ctx: RunContext, policy: Policy, analyzer: PiiAnalyzer) -> dict[str, Any]:
    """The effective policy as written to ``policy.yaml`` (what the user approves)."""
    cfg = ctx.config.sanitize
    return {
        "version": 1,
        "languages": list(cfg.languages),
        "default_action": policy.default_action,
        "score_threshold": policy.score_threshold,
        "policy": dict(policy.actions),
        "custom_patterns": dict(sorted(cfg.custom_patterns.items())),
        "overlap_rule": OVERLAP_RULE,
        "drop_rule": DROP_RULE,
        "placeholders": "[ENTITY_TYPE_n], consistent per document; mapping kept in memory only",
        "ner_mode": analyzer.ner_mode,
        "rule_packs": analyzer.rule_sources(),
    }


class SanitizeStage(Stage):
    name = STAGE

    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        log = get_logger("zllf.stages.sanitize", run_id=ctx.run_id, stage=STAGE)
        t0 = time.time()
        src = ctx.paths.stage("ingest") / "chunks.parquet"
        if not src.exists():
            raise ZLLFError(ErrorCode.SANITIZE_FAILED, "01_ingest/chunks.parquet missing")
        chunks = sorted(read_parquet(src, Chunk), key=lambda c: (c.doc_id, c.index, c.chunk_id))
        cfg = ctx.config.sanitize
        try:
            analyzer = PiiAnalyzer(cfg.languages, cfg.custom_patterns)
        except Exception as exc:
            raise ZLLFError(ErrorCode.SANITIZE_FAILED,
                            f"analyzer init failed: {type(exc).__name__}") from exc
        policy = build_policy(ctx, analyzer)
        salt = derive_salt(ctx.run_id, ctx.config.seed)
        sanitizer = Sanitizer(analyzer, policy, salt)
        log.info("sanitize_started", chunks=len(chunks), ner_mode=analyzer.ner_mode)

        clean: list[Chunk] = []
        redactions: list[Redaction] = []
        dropped_ids: list[str] = []
        languages: Counter[str] = Counter()
        heading_spans = 0
        skipped_duplicates = 0
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if i % 50 == 0:
                ctx.check_cancelled()
                ctx.progress(STAGE, i, total)
            if chunk.duplicate_of is not None:
                skipped_duplicates += 1
                continue
            outcome = sanitizer.process(chunk)
            languages[outcome.language] += 1
            redactions.extend(outcome.redactions)
            heading_spans += outcome.heading_spans
            if outcome.dropped:
                dropped_ids.append(chunk.chunk_id)
            elif outcome.chunk is not None:
                clean.append(outcome.chunk)
        ctx.progress(STAGE, total, total)
        sanitizer.mapper.clear()  # the value -> placeholder mapping is never persisted

        to_parquet(clean, out_dir / "chunks_clean.parquet")
        _write_redactions(redactions, out_dir / "redactions.parquet")
        n_fp = sanitizer.fingerprints.write(out_dir / "pii_fingerprints.json")
        (out_dir / "policy.yaml").write_text(
            yaml.safe_dump(policy_document(ctx, policy, analyzer), sort_keys=False,
                           allow_unicode=True),
            encoding="utf-8",
        )
        report = build_report(
            chunks_total=total - skipped_duplicates,
            chunks_kept=len(clean),
            dropped_ids=dropped_ids,
            redactions=redactions,
            languages=languages,
            analyzer=analyzer,
            policy=policy,
            salt=salt,
            n_fingerprints=n_fp,
            heading_spans=heading_spans,
            skipped_duplicates=skipped_duplicates,
        )
        (out_dir / "sanitize_report.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
        metrics = {k: report[k] for k in ("chunks_total", "chunks_kept", "chunks_dropped",
                                          "redactions_total", "ner_mode", "n_fingerprints")}
        metrics["counts_by_type"] = report["counts_by_type"]
        metrics["seconds"] = round(time.time() - t0, 3)
        log.info("sanitize_completed", **{k: v for k, v in metrics.items()
                                          if k != "counts_by_type"})
        return StageResult(metrics=metrics, needs_approval=True)


def _write_redactions(rows: list[Redaction], path: Path) -> None:
    if rows:
        to_parquet(rows, path)
        return
    import pandas as pd  # empty file with the right columns

    cols = list(Redaction.model_fields)
    pd.DataFrame({c: pd.Series(dtype="object") for c in cols}).to_parquet(path, index=False)


def build_report(
    *,
    chunks_total: int,
    chunks_kept: int,
    dropped_ids: list[str],
    redactions: list[Redaction],
    languages: Counter[str],
    analyzer: PiiAnalyzer,
    policy: Policy,
    salt: bytes,
    n_fingerprints: int,
    heading_spans: int,
    skipped_duplicates: int,
) -> dict[str, Any]:
    """Counts only - never values or text."""
    by_type: Counter[str] = Counter(r.entity_type for r in redactions)
    by_action: Counter[str] = Counter(r.action for r in redactions)
    by_type_action: dict[str, dict[str, int]] = {}
    for r in redactions:
        slot = by_type_action.setdefault(r.entity_type, {})
        slot[r.action] = slot.get(r.action, 0) + 1
    return {
        "chunks_total": chunks_total,
        "chunks_kept": chunks_kept,
        "chunks_dropped": len(dropped_ids),
        "dropped_chunk_ids": dropped_ids,
        "duplicates_skipped": skipped_duplicates,
        "redactions_total": len(redactions),
        "counts_by_type": dict(sorted(by_type.items())),
        "counts_by_action": dict(sorted(by_action.items())),
        "counts_by_type_action": dict(sorted(by_type_action.items())),
        "heading_spans": heading_spans,
        "languages": dict(sorted(languages.items())),
        "language_detector": analyzer.language_detector.backend,
        "ner_mode": analyzer.ner_mode,
        "ner_modes": dict(analyzer.bundle.modes),
        "ner_models": dict(analyzer.bundle.model_versions),
        "score_threshold": policy.score_threshold,
        "default_action": policy.default_action,
        "salt_id": salt_id(salt),
        "n_fingerprints": n_fingerprints,
    }
