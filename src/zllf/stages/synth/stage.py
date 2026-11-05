"""Stage ③ Synthesise (FR-04): teacher-generated, judge-filtered training data.

Reads ``02_sanitize/chunks_clean.parquet`` and writes into ``out_dir``:
``train.jsonl``, ``val.jsonl``, ``test.jsonl`` (SynthExample), ``canaries.json``,
``taxonomy.json``, ``synth_report.json`` and ``prompts_manifest.json``.

Pipeline: taxonomy -> task generators (by ``synthesis.tasks`` weights and mode) ->
judge loop (verbatim quote, NLI, LLM judge, question dedup) -> unanswerables (verified
absent) -> canaries (train only) -> document-level split.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from zllf.core.config import RunConfig
from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.hashing import sha256_json, sha256_text
from zllf.core.logging import get_logger
from zllf.core.schemas import Chunk, Message, SynthExample, read_parquet, write_jsonl
from zllf.llm.base import SamplingParams
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.synth import generators as gen
from zllf.stages.synth import judge
from zllf.stages.synth.canaries import make_canaries
from zllf.stages.synth.common import Candidate, SynthStats, Teacher
from zllf.stages.synth.prompts import prompts_manifest
from zllf.stages.synth.split import SplitPlan, carve_val, plan_split
from zllf.stages.synth.taxonomy import Taxonomy, extract_taxonomy
from zllf.stages.synth.textutil import normalize_ws
from zllf.stages.synth.unanswerable import (
    OVERSAMPLE,
    QUESTIONS_PER_PROMPT,
    cap_round_robin,
    generate_unanswerable,
    pick_seed_chunks,
    verify_absent,
)

CHUNK_TASKS = ("qa", "summarization", "drafting", "extraction")
SUPPORTED_TASKS = (*CHUNK_TASKS, "definition", "unanswerable")
MIN_CHUNK_CHARS = 40
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- planning helpers
def select_chunks(chunks: Sequence[Chunk], max_chunks: int | None) -> list[Chunk]:
    """Usable chunks, round-robin across documents so every document is represented."""
    usable = [c for c in chunks if c.duplicate_of is None
              and len(normalize_ws(c.text)) >= MIN_CHUNK_CHARS]
    if max_chunks is None or len(usable) <= max_chunks:
        return sorted(usable, key=lambda c: (c.doc_id, c.index))
    by_doc: dict[str, list[Chunk]] = {}
    for c in sorted(usable, key=lambda c: (c.doc_id, c.index)):
        by_doc.setdefault(c.doc_id, []).append(c)
    picked: list[Chunk] = []
    i = 0
    while len(picked) < max_chunks:
        for doc in sorted(by_doc):
            if i < len(by_doc[doc]) and len(picked) < max_chunks:
                picked.append(by_doc[doc][i])
        i += 1
    return sorted(picked, key=lambda c: (c.doc_id, c.index))


def resolve_tasks(config: RunConfig, stats: SynthStats) -> dict[str, float]:
    """Supported tasks with positive weight; unsupported/mode-incompatible ones are noted."""
    out: dict[str, float] = {}
    for task, w in sorted(config.synthesis.tasks.items()):
        if w <= 0:
            continue
        if task not in SUPPORTED_TASKS:
            stats.notes.append(f"task '{task}' is not implemented yet and was ignored")
            continue
        if task == "extraction" and config.mode != "C" and config.json_schema is None:
            stats.notes.append("task 'extraction' needs mode C or json_schema; ignored")
            continue
        out[task] = w
    return out


def assign_chunks(
    ctx: RunContext, chunks: Sequence[Chunk], weights: dict[str, float]
) -> dict[str, list[Chunk]]:
    """Per chunk task, the chunks it runs on: fraction = weight / max generative weight."""
    gen_w = {t: w for t, w in weights.items() if t != "unanswerable"}
    top = max(gen_w.values(), default=0.0)
    out: dict[str, list[Chunk]] = {}
    for task in CHUNK_TASKS:
        if task not in gen_w or not chunks:
            continue
        k = min(len(chunks), math.ceil(len(chunks) * gen_w[task] / top))
        idx = sorted(ctx.rng(f"synth.assign.{task}").sample(range(len(chunks)), k))
        out[task] = [chunks[i] for i in idx]
    return out


def example_id(c: Candidate) -> str:
    return sha256_text(f"{c.task}|{c.chunk_id or c.doc_id}|{c.user}|{c.answer}")[:16]


def to_example(c: Candidate, system_prompt: str) -> SynthExample:
    return SynthExample(
        example_id=example_id(c),
        task=c.task,  # type: ignore[arg-type]
        split="train",
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=c.user),
            Message(role="assistant", content=c.answer),
        ],
        doc_id=c.doc_id,
        chunk_id=c.chunk_id,
        quote=c.quote,
        reference_answer=c.answer,
        reference_json=c.reference_json,
        answerable=c.answerable,
        entailment=c.entailment,
        judge_ok=c.judge_ok,
    )


def _avg(values: list[int]) -> float:
    return round(mean(values), 1) if values else 0.0


# ---------------------------------------------------------------- stage
class SynthStage(Stage):
    name = "synth"

    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        t0 = time.monotonic()
        cfg = ctx.config
        syn = cfg.synthesis
        log = get_logger("zllf.synth", run_id=ctx.run_id)
        stats = SynthStats()

        all_chunks = [
            c for c in read_parquet(ctx.paths.stage("sanitize") / "chunks_clean.parquet", Chunk)
            if c.duplicate_of is None
        ]
        chunks = select_chunks(all_chunks, syn.max_chunks)
        weights = resolve_tasks(cfg, stats)
        if not chunks:
            raise ZLLFError(ErrorCode.SYNTH_TOO_FEW_PAIRS, "no usable chunks")
        ctx.emit("synth_plan", n_chunks=len(chunks), n_chunks_total=len(all_chunks),
                 tasks=sorted(weights))

        llm = ctx.llm("teacher")
        nli = ctx.nli()
        embedder = ctx.embedder()
        teacher = Teacher(
            llm,
            SamplingParams(temperature=syn.teacher_temperature, max_tokens=512, seed=cfg.seed),
            stats,
            progress=lambda phase, d, t: ctx.progress("synth", d, t, phase),
            check_cancelled=ctx.check_cancelled,
        )
        try:
            taxonomy, kept, unans = self._synthesize(ctx, teacher, nli, embedder, chunks,
                                                     all_chunks, weights, stats)
        finally:
            for model in (llm, nli, embedder):
                close = getattr(model, "close", None)
                if callable(close):
                    close()

        # ---------------------------------------------------------------- examples
        examples: list[SynthExample] = []
        seen: set[str] = set()
        for c in [*kept, *unans]:
            ex = to_example(c, cfg.system_prompt)
            if ex.example_id in seen:
                stats.drop(c.task, "duplicate_example")
                stats.kept[c.task] -= 1
                continue
            seen.add(ex.example_id)
            examples.append(ex)

        plan = plan_split(Counter(e.doc_id or "" for e in examples), cfg.splits,
                          ctx.rng("split"))
        for e in examples:
            e.split = plan.doc_split.get(e.doc_id or "", "train")
        if not plan.docs("val") and cfg.splits.val > 0:
            train_ids = [e.example_id for e in examples if e.split == "train"]
            moved = carve_val(train_ids, cfg.splits.val, ctx.rng("split.val"))
            if moved:
                plan.val_from_train_docs = True
                plan.warnings.append("no val document: val carved from train-document examples")
                for e in examples:
                    if e.example_id in moved:
                        e.split = "val"

        canaries, canary_samples = make_canaries(syn.n_canaries, ctx.rng("canaries"),
                                                 cfg.system_prompt)
        by_split: dict[str, list[SynthExample]] = {s: [] for s in SPLITS}
        for e in examples:
            by_split[e.split].append(e)
        by_split["train"].extend(canary_samples)
        for s in SPLITS:
            by_split[s].sort(key=lambda e: e.example_id)
            ctx.rng(f"synth.order.{s}").shuffle(by_split[s])

        n_train_pairs = sum(1 for e in by_split["train"] if e.task != "canary")
        if n_train_pairs < syn.min_kept_pairs:
            log.warning("synth_too_few_pairs", kept_train=n_train_pairs,
                        required=syn.min_kept_pairs)
            raise ZLLFError(
                ErrorCode.SYNTH_TOO_FEW_PAIRS,
                f"{n_train_pairs} kept training pairs < min_kept_pairs={syn.min_kept_pairs}",
            )

        # ---------------------------------------------------------------- write
        for s in SPLITS:
            write_jsonl(by_split[s], out_dir / f"{s}.jsonl")
        _write_json(out_dir / "canaries.json", [c.model_dump(mode="json") for c in canaries])
        _write_json(out_dir / "taxonomy.json", taxonomy.to_json())
        manifest = prompts_manifest()
        _write_json(out_dir / "prompts_manifest.json", manifest)
        report = build_report(stats, examples, by_split, plan, chunks, all_chunks, weights,
                              manifest, time.monotonic() - t0)
        _write_json(out_dir / "synth_report.json", report)

        metrics = {
            "n_train": len(by_split["train"]),
            "n_val": len(by_split["val"]),
            "n_test": len(by_split["test"]),
            "n_train_pairs": n_train_pairs,
            "n_canaries": len(canaries),
            "candidates": sum(stats.candidates.values()),
            "kept": len(examples),
            "seconds": report["seconds"],
        }
        log.info("synth_done", **metrics)
        return StageResult(metrics=metrics)

    # ---------------------------------------------------------------- generation + judge
    def _synthesize(
        self,
        ctx: RunContext,
        teacher: Teacher,
        nli: Any,
        embedder: Any,
        chunks: list[Chunk],
        all_chunks: list[Chunk],
        weights: dict[str, float],
        stats: SynthStats,
    ) -> tuple[Taxonomy, list[Candidate], list[Candidate]]:
        cfg = ctx.config
        syn = cfg.synthesis
        taxonomy = extract_taxonomy(teacher, chunks)
        assignment = assign_chunks(ctx, chunks, weights)

        cands: list[Candidate] = []
        if "qa" in assignment:
            lo, hi = syn.quotes_per_chunk
            cands += gen.generate_qa(teacher, assignment["qa"], stats, (lo, hi))
        if "summarization" in assignment:
            cands += gen.generate_summaries(teacher, assignment["summarization"], stats)
        if "drafting" in assignment:
            cands += gen.generate_drafting(teacher, assignment["drafting"], stats)
        if "extraction" in assignment:
            schema = cfg.json_schema or gen.DEFAULT_EXTRACTION_SCHEMA
            if cfg.json_schema is None:
                stats.notes.append("no json_schema configured: default generic schema used")
            cands += gen.generate_extraction(teacher, assignment["extraction"], stats, schema,
                                             nli, syn.entailment_threshold)
        if "definition" in weights:
            terms = taxonomy.of_kind("acronym") + taxonomy.of_kind("concept")
            gen_w = {t: w for t, w in weights.items() if t != "unanswerable"}
            k = math.ceil(len(terms) * weights["definition"] / max(gen_w.values()))
            chunk_by_id = {c.chunk_id: c for c in chunks}
            cands += gen.generate_definitions(teacher, terms[:k], chunk_by_id, stats)

        ctx.check_cancelled()
        cands = judge.quote_filter(cands, stats)
        cands = judge.nli_filter(cands, nli, syn.entailment_threshold, stats)
        cands = judge.llm_judge_filter(cands, teacher, stats)
        cands = judge.dedup_filter(cands, embedder, syn.question_dedup_cosine, stats)
        for c in cands:
            stats.kept[c.task] += 1

        unans: list[Candidate] = []
        w_u = weights.get("unanswerable", 0.0)
        gen_total = sum(w for t, w in weights.items() if t != "unanswerable")
        if w_u > 0 and cands and gen_total > 0:
            target = max(1, round(len(cands) * w_u / gen_total))
            n_prompts = math.ceil(target * OVERSAMPLE / QUESTIONS_PER_PROMPT)
            seeds = pick_seed_chunks(chunks, n_prompts)
            proposed = generate_unanswerable(teacher, seeds, taxonomy.names(), stats)
            chunk_emb = judge.embed(embedder, [c.text for c in all_chunks])
            verified = verify_absent(proposed, all_chunks, chunk_emb, embedder, nli, stats)
            verified = judge.dedup_filter(verified, embedder, syn.question_dedup_cosine, stats)
            unans = cap_round_robin(verified, target)
            stats.drop("unanswerable", "over_target", len(verified) - len(unans))
            stats.kept["unanswerable"] += len(unans)
        return taxonomy, cands, unans


# ---------------------------------------------------------------- report
def build_report(
    stats: SynthStats,
    examples: Sequence[SynthExample],
    by_split: dict[str, list[SynthExample]],
    plan: SplitPlan,
    chunks: Sequence[Chunk],
    all_chunks: Sequence[Chunk],
    weights: dict[str, float],
    manifest: dict[str, str],
    seconds: float,
) -> dict[str, Any]:
    """synth_report.json: counts, rates and lengths only (no document text)."""
    tasks = sorted({e.task for e in examples} | set(stats.candidates))
    lengths = {
        t: {
            "user_chars": _avg([len(e.messages[1].content) for e in examples if e.task == t]),
            "answer_chars": _avg([len(e.messages[2].content) for e in examples if e.task == t]),
        }
        for t in tasks
    }
    docs_with_examples = {e.doc_id for e in examples if e.doc_id}
    return {
        "n_chunks_total": len(all_chunks),
        "n_chunks_used": len(chunks),
        "n_documents": len({c.doc_id for c in chunks}),
        "n_documents_without_examples": len({c.doc_id for c in chunks} - docs_with_examples),
        "task_weights": weights,
        "candidates": dict(sorted(stats.candidates.items())),
        "kept": dict(sorted((k, v) for k, v in stats.kept.items() if v)),
        "drop_reasons": {t: dict(sorted(r.items())) for t, r in sorted(stats.drops.items())},
        "malformed_teacher_outputs": dict(sorted(stats.malformed.items())),
        "quotes_repaired": stats.quotes_repaired,
        "judge_pass_rates": stats.pass_rates(),
        "judge_checked": dict(sorted(stats.judge_checked.items())),
        "split_counts": {
            s: dict(sorted(Counter(e.task for e in by_split[s]).items())) for s in SPLITS
        },
        "split_totals": {s: len(by_split[s]) for s in SPLITS},
        "docs_per_split": {s: len(plan.docs(s)) for s in SPLITS},  # type: ignore[arg-type]
        "doc_ids_per_split": {s: plan.docs(s) for s in SPLITS},  # type: ignore[arg-type]
        "val_from_train_docs": plan.val_from_train_docs,
        "avg_lengths": lengths,
        "teacher_calls": stats.teacher_calls,
        "teacher_batches": stats.teacher_batches,
        "prompts_hash": sha256_json(manifest),
        "warnings": [*plan.warnings, *stats.notes],
        "seconds": round(seconds, 3),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")
