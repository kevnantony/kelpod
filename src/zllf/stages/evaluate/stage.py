"""Stage ⑤ Evaluate: harness, RAG baseline and the safety gates (FR-13, FR-14, FR-18, P2.6).

Inputs (committed outputs of earlier stages)::

    03_synth/test.jsonl, canaries.json          synthetic test split (+ unanswerables), canaries
    02_sanitize/chunks_clean.parquet            RAG corpus
    02_sanitize/pii_fingerprints.json           salted raw-PII hashes (G-4 scan)
    04_train/adapter/                           candidate LoRA adapter
    <run_dir>/gold.jsonl | <corpus_dir>/../gold.jsonl   optional gold set (see items.py)
    bench/anchor/                               anchor suites (see anchor.py)

Outputs (``out_dir``): ``predictions.jsonl`` (EvalPrediction), ``items.jsonl`` (EvalItem, the
evaluated prompts; ids = SynthExample.example_id for synthetic items), ``results.json``
(``{"variants": {variant: {suite: {metric: {"value", "ci": [lo, hi], "n"}}}}}``, 95 %
bootstrap CIs), ``gates.json`` (list[GateResult], ``details["direction"]`` says how to read
``value``), ``eval_report.json`` and ``rag_index/`` (embeddings + chunk ids, no text).

Phases (one model resident at a time): build items -> RAG index (embedder) -> student
generation for base / adapter / RAG variants, MC log-likelihoods, canary probes and exposure
-> close student -> judge (correctness, abstention) -> close judge -> metrics -> gates.
Gate failures do not raise; ``all_gates_passed`` is reported for the package stage / UI.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import (
    Canary,
    Chunk,
    EvalItem,
    EvalPrediction,
    GateResult,
    Message,
    SynthExample,
    Variant,
    read_jsonl,
    read_parquet,
    write_jsonl,
)
from zllf.llm.base import Embedder, LLMBackend, SamplingParams
from zllf.stages.base import RunContext, Stage, StageResult
from zllf.stages.evaluate import anchor as anchor_mod
from zllf.stages.evaluate import gates as gates_mod
from zllf.stages.evaluate import items as items_mod
from zllf.stages.evaluate import memorization as mem
from zllf.stages.evaluate.judge import (
    abstention_prompt,
    correctness_prompt,
    keyword_abstention,
    parse_abstention,
    parse_verdict,
)
from zllf.stages.evaluate.metrics import (
    cosine_rows,
    extract_json,
    field_prf,
    json_exact_match,
    rouge_l,
    summarize,
    validate_schema,
)
from zllf.stages.evaluate.rag import RagIndex, passage_char_budget, rag_prompt

STAGE = "evaluate"
GEN_BATCH = 16
DOMAIN_MAX_TOKENS = 384
ANCHOR_MAX_TOKENS = 320
CANARY_MAX_TOKENS = 48
JUDGE_MAX_TOKENS = 96
EMBED_BATCH = 64

Key = tuple[str, Variant]  # (suite, variant)


class _Progress:
    def __init__(self, ctx: RunContext, total: int) -> None:
        self.ctx, self.total, self.done = ctx, max(1, total), 0

    def tick(self, n: int, message: str) -> None:
        self.done = min(self.total, self.done + n)
        self.ctx.progress(STAGE, self.done, self.total, message)


def judge_fallback_score(rouge: float) -> float:
    """Correctness fallback when the judge verdict is unparseable (counted in the report)."""
    return 1.0 if rouge >= 0.6 else 0.5 if rouge >= 0.3 else 0.0


class EvaluateStage(Stage):
    name = "evaluate"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _chat(ctx: RunContext, llm: LLMBackend, prompts: Sequence[Sequence[Message]],
              params: SamplingParams, progress: _Progress, message: str, *,
              adapter_path: Path | None = None,
              json_schema: dict[str, Any] | None = None) -> list[str]:
        out: list[str] = []
        for i in range(0, len(prompts), GEN_BATCH):
            ctx.check_cancelled()
            batch = list(prompts[i : i + GEN_BATCH])
            res = llm.chat(batch, params, json_schema=json_schema, adapter_path=adapter_path)
            if len(res) != len(batch):
                raise ZLLFError(ErrorCode.EVAL_FAILED, "backend returned wrong batch size")
            out.extend(res)
            progress.tick(len(batch), message)
        return out

    @staticmethod
    def _loglik(ctx: RunContext, llm: LLMBackend, prompts: Sequence[Sequence[Message]],
                completions: Sequence[str], *, adapter_path: Path | None) -> list[float]:
        out: list[float] = []
        for i in range(0, len(prompts), GEN_BATCH):
            ctx.check_cancelled()
            out.extend(llm.loglikelihood(prompts[i : i + GEN_BATCH],
                                         completions[i : i + GEN_BATCH],
                                         adapter_path=adapter_path))
        return out

    @staticmethod
    def _embed(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
        parts = [embedder.encode(list(texts[i : i + EMBED_BATCH]))
                 for i in range(0, len(texts), EMBED_BATCH)]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1), np.float32)

    # ---------------------------------------------------------------- inputs
    def _load_inputs(self, ctx: RunContext, warnings: list[str],
                     ) -> tuple[list[SynthExample], list[Canary], list[Chunk], Path]:
        synth = ctx.paths.stage("synth")
        test_path = synth / "test.jsonl"
        if not test_path.is_file():
            raise ZLLFError(ErrorCode.EVAL_FAILED, "synth test split missing")
        test = read_jsonl(test_path, SynthExample)
        canaries: list[Canary] = []
        if (synth / "canaries.json").is_file():
            raw = json.loads((synth / "canaries.json").read_text(encoding="utf-8"))
            canaries = [Canary.model_validate(c) for c in raw]
        else:
            warnings.append("canaries.json missing: canary probes skipped")
        chunks_path = ctx.paths.stage("sanitize") / "chunks_clean.parquet"
        chunks: list[Chunk] = []
        if chunks_path.is_file():
            chunks = read_parquet(chunks_path, Chunk)
        else:
            warnings.append("chunks_clean.parquet missing: RAG variants skipped")
        adapter = ctx.paths.stage("train") / "adapter"
        if not adapter.exists():
            raise ZLLFError(ErrorCode.EVAL_FAILED, "trained adapter missing")
        return test, canaries, chunks, adapter

    def _build_suites(self, ctx: RunContext, test: list[SynthExample], chunks: list[Chunk],
                      warnings: list[str]) -> tuple[items_mod.SuiteSet, dict[str, str]]:
        cfg = ctx.config
        sp = cfg.system_prompt
        limit = cfg.eval.max_test_items
        suites = items_mod.SuiteSet()
        domain = items_mod.domain_items(test, sp)
        suites.domain["domain_test"] = items_mod.subsample(domain, limit, ctx.rng("domain"))
        gold_file = items_mod.find_gold_file(ctx.paths.root, ctx.corpus_dir)
        gold_unans: list[EvalItem] = []
        if gold_file is not None:
            gold, gold_unans = items_mod.load_gold(gold_file, sp)
            if gold:
                suites.domain["gold"] = gold
        held = items_mod.heldout_unanswerable_items(test, sp)
        held = items_mod.subsample(held, limit, ctx.rng("unanswerable"))
        known = "\n".join(c.text for c in chunks)
        adv = items_mod.adversarial_items(sp, ctx.rng("adversarial"), known_text=known)
        suites.unanswerable = held + gold_unans + adv
        sources: dict[str, str] = {}
        for name, path in anchor_mod.discover_suites(anchor_mod.ANCHOR_DIR).items():
            items = anchor_mod.load_suite(name, path)
            items = items_mod.subsample(items, cfg.eval.anchor_items_per_suite,
                                        ctx.rng(f"anchor:{name}"))
            if items:
                suites.anchors[f"anchor/{name}"] = items
                sources[name] = _rel(path)
        if not suites.anchors:
            warnings.append("no anchor suites found: G2 cannot pass")
        return suites, sources

    # ---------------------------------------------------------------- run
    def run(self, ctx: RunContext, out_dir: Path) -> StageResult:
        t0 = time.time()
        cfg = ctx.config
        mode_c = cfg.mode == "C"
        schema = cfg.json_schema if mode_c else None
        warnings: list[str] = []
        test, canaries, chunks, adapter = self._load_inputs(ctx, warnings)
        suites, anchor_sources = self._build_suites(ctx, test, chunks, warnings)
        ctx.emit("eval_items", stage=STAGE, counts=suites.counts(), n_canaries=len(canaries))

        # ------------------------------------------------ RAG index
        embedder = ctx.embedder()
        rag: RagIndex | None = None
        rag_prompts: dict[str, list[list[Message]]] = {}
        if chunks:
            rag = RagIndex.build(chunks, embedder)
            if len(rag) == 0:
                rag = None
        if rag is not None:
            rag.save(out_dir / "rag_index")
            budget = passage_char_budget(cfg.models.max_seq_len, cfg.eval.rag_top_k)
            for suite, items in suites.domain.items():
                q = self._embed(embedder, [items_mod.last_user(it.prompt) for it in items])
                hits = rag.search(q, cfg.eval.rag_top_k) if len(items) else []
                rag_prompts[suite] = [
                    rag_prompt(it.prompt, [rag.texts[i] for i, _ in h], budget)
                    for it, h in zip(items, hits, strict=True)
                ]
        domain_variants: list[Variant] = ["base", "adapter"]
        if rag is not None:
            domain_variants += ["base_rag", "adapter_rag"]

        # ------------------------------------------------ progress budget
        n_dom = sum(len(v) for v in suites.domain.values())
        n_unans = len(suites.unanswerable)
        n_anchor = sum(len(v) for v in suites.anchors.values())
        n_probe = len(canaries) * 2 * (1 + mem.N_SAMPLED)
        total = (n_dom * len(domain_variants) * (1 if mode_c else 2) + n_unans * 4
                 + n_anchor * 2 + n_probe + len(canaries) * 2)
        progress = _Progress(ctx, total)

        outputs: dict[Key, list[str]] = {}
        scores: dict[Key, list[dict[str, float]]] = {}
        canary_preds: list[EvalPrediction] = []
        extracted: list[str] = []
        exposures: dict[str, dict[str, float]] = {}

        def adapter_for(v: Variant) -> Path | None:
            return adapter if v.startswith("adapter") else None

        # ------------------------------------------------ student phase
        gen = SamplingParams(temperature=0.0, max_tokens=DOMAIN_MAX_TOKENS, seed=cfg.seed)
        student = ctx.llm("student")
        try:
            for suite, items in suites.domain.items():
                for v in domain_variants:
                    prompts = (rag_prompts[suite] if v.endswith("_rag")
                               else [i.prompt for i in items])
                    outputs[(suite, v)] = self._chat(
                        ctx, student, prompts, gen, progress, f"{suite}/{v}",
                        adapter_path=adapter_for(v), json_schema=schema)
            for v in ("base", "adapter"):
                outputs[("unanswerable", v)] = self._chat(
                    ctx, student, [i.prompt for i in suites.unanswerable], gen, progress,
                    f"unanswerable/{v}", adapter_path=adapter_for(v))
            anchor_gen = SamplingParams(temperature=0.0, max_tokens=ANCHOR_MAX_TOKENS,
                                        seed=cfg.seed)
            for suite, items in suites.anchors.items():
                for v in ("base", "adapter"):
                    outputs[(suite, v)], scores[(suite, v)] = self._run_anchor(
                        ctx, student, items, anchor_gen, progress, suite, adapter_for(v))
            if canaries:
                canary_preds, extracted = self._probe_canaries(
                    ctx, student, canaries, adapter, progress)
                exposures = self._exposures(ctx, student, canaries, adapter, progress)
        finally:
            student.close()

        # ------------------------------------------------ judge phase
        fallbacks = {"correctness": 0, "abstention": 0}
        judge_scores: dict[Key, list[float]] = {}
        labels: dict[Variant, list[str]] = {}
        jparams = SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, seed=cfg.seed)
        judge = ctx.llm("judge")
        try:
            if not mode_c:
                for suite, items in suites.domain.items():
                    for v in domain_variants:
                        outs = outputs[(suite, v)]
                        jp = [correctness_prompt(items_mod.last_user(it.prompt),
                                                 it.reference or "", o)
                              for it, o in zip(items, outs, strict=True)]
                        verdicts = self._chat(ctx, judge, jp, jparams, progress,
                                              f"judge {suite}/{v}")
                        vals: list[float] = []
                        for it, o, raw in zip(items, outs, verdicts, strict=True):
                            s = parse_verdict(raw)
                            if s is None:
                                fallbacks["correctness"] += 1
                                s = judge_fallback_score(rouge_l(o, it.reference or ""))
                            vals.append(s)
                        judge_scores[(suite, v)] = vals
            for v in ("base", "adapter"):
                outs = outputs[("unanswerable", v)]
                ap = [abstention_prompt(items_mod.last_user(it.prompt), o)
                      for it, o in zip(suites.unanswerable, outs, strict=True)]
                raw_labels = self._chat(ctx, judge, ap, jparams, progress,
                                        f"judge unanswerable/{v}")
                lab: list[str] = []
                for o, raw in zip(outs, raw_labels, strict=True):
                    parsed = parse_abstention(raw)
                    if parsed is None:
                        fallbacks["abstention"] += 1
                        parsed = keyword_abstention(o)
                    lab.append(parsed)
                labels[v] = lab
        finally:
            judge.close()

        # ------------------------------------------------ domain metrics
        for suite, items in suites.domain.items():
            refs = [it.reference or "" for it in items]
            ref_emb = self._embed(embedder, refs)
            for v in domain_variants:
                outs = outputs[(suite, v)]
                cos = (cosine_rows(self._embed(embedder, outs), ref_emb) if outs
                       else np.zeros(0, dtype=np.float32))
                rows: list[dict[str, float]] = []
                for k, (it, o) in enumerate(zip(items, outs, strict=True)):
                    row = {"rouge_l": rouge_l(o, refs[k]), "embed_cos": float(cos[k])}
                    if (suite, v) in judge_scores:
                        row["judge_correct"] = judge_scores[(suite, v)][k]
                    if mode_c:
                        obj = extract_json(o)
                        row["schema_valid"] = float(validate_schema(obj, schema))
                        if it.reference_json is not None:
                            p, r, f = field_prf(obj, it.reference_json)
                            row.update(field_p=p, field_r=r, field_f1=f,
                                       exact_match=json_exact_match(obj, it.reference_json))
                    rows.append(row)
                scores[(suite, v)] = rows
        for v in ("base", "adapter"):
            scores[("unanswerable", v)] = [
                {"abstain": float(x == "abstain"), "fabricate": float(x == "fabricate"),
                 "hedge": float(x == "hedge")} for x in labels[v]]

        # ------------------------------------------------ predictions
        all_items: dict[str, list[EvalItem]] = {**suites.domain,
                                                "unanswerable": suites.unanswerable,
                                                **suites.anchors}
        preds: list[EvalPrediction] = []
        for (suite, v), outs in outputs.items():
            for it, o, sc in zip(all_items[suite], outs, scores[(suite, v)], strict=True):
                preds.append(EvalPrediction(item_id=it.item_id, suite=suite, variant=v,
                                            output=o, scores=sc))
        preds.extend(canary_preds)

        # ------------------------------------------------ raw-PII scan of adapter outputs
        fp_path = ctx.paths.stage("sanitize") / "pii_fingerprints.json"
        checker, pii_status = mem.load_fingerprint_checker(fp_path, ctx.run_id, cfg.seed)
        pii_hits: list[str] = []
        if checker is not None:
            for pred in preds:
                if pred.variant in ("adapter", "adapter_rag") and checker(pred.output) > 0:
                    pii_hits.append(f"{pred.suite}:{pred.item_id}:{pred.variant}")
        else:
            warnings.append(f"raw-PII scan {pii_status}")

        # ------------------------------------------------ results + gates
        primary = "field_f1" if mode_c else "judge_correct"
        results = self._results(ctx, scores, primary)
        gate_list = self._gates(ctx, suites, scores, labels, primary, canaries, extracted,
                                exposures, pii_hits, pii_status, fallbacks, anchor_sources)

        write_jsonl(preds, out_dir / "predictions.jsonl")
        write_jsonl([it for its in all_items.values() for it in its], out_dir / "items.jsonl")
        _write_json(out_dir / "results.json", results)
        _write_json(out_dir / "gates.json", [g.model_dump(mode="json") for g in gate_list])
        all_passed = all(g.passed for g in gate_list)
        report = {
            "run_id": ctx.run_id, "mode": cfg.mode,
            "models": {"student": cfg.models.student,
                       "judge": cfg.models.judge or cfg.models.teacher,
                       "embedder": cfg.models.embedder, "backend": cfg.models.backend},
            "adapter": _rel_to(adapter, ctx.paths.root),
            "variants": {"domain": list(domain_variants),
                         "unanswerable": ["base", "adapter"], "anchor": ["base", "adapter"],
                         "canary": ["adapter"]},
            "counts": {**suites.counts(), "canaries": len(canaries), "predictions": len(preds)},
            "gold_set": "gold" in suites.domain,
            "rag": {"enabled": rag is not None, "n_chunks": len(rag) if rag else 0,
                    "backend": rag.backend if rag else None, "top_k": cfg.eval.rag_top_k},
            "anchor_sources": anchor_sources,
            "judge_fallbacks": fallbacks,
            "pii_scan": pii_status,
            "primary_metric": primary,
            "gates": {g.gate: g.passed for g in gate_list},
            "all_gates_passed": all_passed,
            "warnings": warnings,
            "seconds": round(time.time() - t0, 3),
        }
        _write_json(out_dir / "eval_report.json", report)
        g1 = next(g for g in gate_list if g.gate == "G1_domain")
        return StageResult(metrics={
            "all_gates_passed": all_passed,
            "gates": {g.gate: g.passed for g in gate_list},
            "primary_metric": primary,
            "rel_improvement": g1.value,
            "n_predictions": len(preds),
            "warnings": len(warnings),
        })

    # ---------------------------------------------------------------- anchors
    def _run_anchor(self, ctx: RunContext, llm: LLMBackend, items: list[EvalItem],
                    params: SamplingParams, progress: _Progress, suite: str,
                    adapter_path: Path | None) -> tuple[list[str], list[dict[str, float]]]:
        outs: list[str] = [""] * len(items)
        rows: list[dict[str, float]] = [{} for _ in items]
        mc = [k for k, it in enumerate(items) if it.choices]
        gen = [k for k, it in enumerate(items) if not it.choices]
        if mc:
            prompts: list[list[Message]] = []
            comps: list[str] = []
            for k in mc:
                for letter in anchor_mod.mc_letters(items[k]):
                    prompts.append(items[k].prompt)
                    comps.append(letter)
            ll = self._loglik(ctx, llm, prompts, comps, adapter_path=adapter_path)
            pos = 0
            for k in mc:
                n = len(items[k].choices or [])
                correct, letter = anchor_mod.score_mc(items[k], ll[pos : pos + n])
                pos += n
                outs[k], rows[k] = letter, {"correct": correct}
            progress.tick(len(mc), f"{suite} (mc)")
        if gen:
            res = self._chat(ctx, llm, [items[k].prompt for k in gen], params, progress,
                             suite, adapter_path=adapter_path)
            for k, o in zip(gen, res, strict=True):
                outs[k], rows[k] = o, {"correct": anchor_mod.score_generation(items[k], o)}
        return outs, rows

    # ---------------------------------------------------------------- canaries
    def _probe_canaries(self, ctx: RunContext, llm: LLMBackend, canaries: list[Canary],
                        adapter: Path, progress: _Progress,
                        ) -> tuple[list[EvalPrediction], list[str]]:
        sp = ctx.config.system_prompt
        probes: list[tuple[Canary, str, list[Message]]] = [
            (c, form, chat) for c in canaries for form, chat in mem.probe_prompts(c, sp)]
        chats = [p[2] for p in probes]
        runs: list[tuple[str, SamplingParams]] = [
            ("greedy", SamplingParams(temperature=0.0, max_tokens=CANARY_MAX_TOKENS,
                                      seed=ctx.config.seed))]
        runs += [(f"s{k}", SamplingParams(temperature=1.0, top_p=0.95,
                                          max_tokens=CANARY_MAX_TOKENS,
                                          seed=ctx.config.seed + 1 + k))
                 for k in range(mem.N_SAMPLED)]
        preds: list[EvalPrediction] = []
        extracted: list[str] = []
        for tag, params in runs:
            outs = self._chat(ctx, llm, chats, params, progress, f"canary/{tag}",
                              adapter_path=adapter)
            for (c, form, _), o in zip(probes, outs, strict=True):
                hit = mem.secret_reproduced(o, c.secret)
                if hit:
                    extracted.append(c.canary_id)
                preds.append(EvalPrediction(
                    item_id=f"{c.canary_id}:{form}:{tag}", suite="canary", variant="adapter",
                    output=o, scores={"reproduced": float(hit),
                                      "lcs_frac": mem.longest_common_substring_frac(
                                          o, c.secret)}))
        return preds, sorted(set(extracted))

    def _exposures(self, ctx: RunContext, llm: LLMBackend, canaries: list[Canary],
                   adapter: Path, progress: _Progress) -> dict[str, dict[str, float]]:
        sp = ctx.config.system_prompt
        out: dict[str, dict[str, float]] = {}
        for c in canaries:
            cands = mem.exposure_candidates(c.secret, ctx.rng(f"exposure:{c.canary_id}"))
            chat = mem.probe_prompts(c, sp)[0][1]
            entry: dict[str, float] = {"n_candidates": float(len(cands))}
            for name, path in (("adapter", adapter), ("base", None)):
                ll = self._loglik(ctx, llm, [chat] * len(cands), cands, adapter_path=path)
                rank, exp_bits = mem.exposure(ll)
                entry[f"{name}_rank"] = float(rank)
                entry[f"{name}_exposure"] = exp_bits
            out[c.canary_id] = entry
            progress.tick(2, "canary exposure")
        return out

    # ---------------------------------------------------------------- aggregation
    @staticmethod
    def _results(ctx: RunContext, scores: dict[Key, list[dict[str, float]]], primary: str,
                 ) -> dict[str, Any]:
        res = ctx.config.eval.bootstrap_resamples
        variants: dict[str, dict[str, dict[str, Any]]] = {}
        for (suite, v), rows in sorted(scores.items()):
            metrics = sorted({m for r in rows for m in r})
            per_metric: dict[str, Any] = {}
            for m in metrics:
                s = summarize([r[m] for r in rows if m in r], res,
                              ctx.np_rng(f"bootstrap:{suite}:{v}:{m}"))
                per_metric[m] = {"value": s["mean"], "ci": [s["ci_low"], s["ci_high"]],
                                 "n": s["n"]}
            variants.setdefault(v, {})[suite] = per_metric
        return {"primary_metric": primary, "bootstrap_resamples": res,
                "confidence": 0.95, "variants": variants}

    def _gates(self, ctx: RunContext, suites: items_mod.SuiteSet,
               scores: dict[Key, list[dict[str, float]]], labels: dict[Variant, list[str]],
               primary: str, canaries: list[Canary], extracted: list[str],
               exposures: dict[str, dict[str, float]], pii_hits: list[str], pii_status: str,
               fallbacks: dict[str, int], anchor_sources: dict[str, str]) -> list[GateResult]:
        cfg = ctx.config
        g = cfg.gates
        gate_suite = "gold" if "gold" in suites.domain else "domain_test"
        items = suites.domain.get(gate_suite, [])

        def col(v: Variant) -> list[float] | None:
            rows = scores.get((gate_suite, v))
            if rows is None:
                return None
            return [r.get(primary, 0.0) for r in rows]

        g1 = gates_mod.gate_domain(
            suite=gate_suite, metric=primary, item_ids=[i.item_id for i in items],
            adapter=col("adapter") or [], base=col("base") or [], base_rag=col("base_rag"),
            gates=g, resamples=cfg.eval.bootstrap_resamples, rng=ctx.np_rng("gate:G1"))
        if fallbacks["correctness"]:
            g1.details["judge_fallbacks"] = fallbacks["correctness"]

        per_suite = {}
        for suite, its in suites.anchors.items():
            name = suite.split("/", 1)[1]
            per_suite[name] = gates_mod.AnchorScores(
                item_ids=[i.item_id for i in its],
                base=[r["correct"] for r in scores[(suite, "base")]],
                adapter=[r["correct"] for r in scores[(suite, "adapter")]],
                extra={"source": anchor_sources.get(name, "")})
        g2 = gates_mod.gate_forgetting(per_suite, g)

        g3 = gates_mod.gate_hallucination(
            [i.item_id for i in suites.unanswerable], labels.get("adapter", []), g,
            n_judge_fallback=fallbacks["abstention"])
        base_labels = labels.get("base", [])
        if base_labels:
            g3.details["base_abstention_rate"] = sum(
                x == "abstain" for x in base_labels) / len(base_labels)

        g4 = gates_mod.gate_memorization(
            n_canaries=len(canaries), extracted=extracted, exposures=exposures,
            pii_hit_items=pii_hits, pii_status=pii_status, gates=g)
        out = [g1, g2, g3, g4]

        if cfg.mode == "C":
            ids: list[str] = []
            valid: list[bool] = []
            for suite, its in suites.domain.items():
                for it, r in zip(its, scores[(suite, "adapter")], strict=True):
                    ids.append(f"{suite}:{it.item_id}")
                    valid.append(bool(r.get("schema_valid", 0.0)))
            out.append(gates_mod.gate_schema(ids, valid, g,
                                             schema_available=cfg.json_schema is not None))
        return out


def _rel(path: Path) -> str:
    from zllf.core.config import REPO_ROOT

    return _rel_to(path, REPO_ROOT)


def _rel_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_sanitize(data), indent=2), encoding="utf-8")


def _sanitize(data: Any) -> Any:
    """Replace NaN/inf with None (standard JSON) and numpy scalars with Python ones."""
    if isinstance(data, dict):
        return {k: _sanitize(v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [_sanitize(v) for v in data]
    if isinstance(data, np.generic):
        data = data.item()
    if isinstance(data, float) and not np.isfinite(data):
        return None
    return data
