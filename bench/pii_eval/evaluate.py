#!/usr/bin/env python3
"""Precision / recall of the sanitizer on the labelled synthetic notes (FR-03 acceptance).

Usage::

    uv run python bench/pii_eval/evaluate.py                 # NER if local spaCy models exist
    uv run python bench/pii_eval/evaluate.py --ner regex     # regex/heuristic-only mode
    uv run python bench/pii_eval/evaluate.py --json out.json

Metrics (per entity type and overall), on spans that survive ``score_threshold`` + overlap
resolution (i.e. what the stage would redact):

* ``recall``      gold span overlapped by a predicted span of the SAME type.
* ``protected``   gold span whose non-space characters are >= 90 % covered by ANY non-keep
                  span (type-agnostic: what matters for leakage). This is the FR-03 number.
* ``precision``   predicted spans overlapping a gold span of the same type / predicted spans.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from zllf.core.config import load_config
from zllf.stages.sanitize.analyzer import PiiAnalyzer
from zllf.stages.sanitize.policy import policy_from_config, resolve

HERE = Path(__file__).resolve().parent

PROTECT_FRACTION = 0.9


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def evaluate(notes: list[dict[str, Any]], ner: str = "auto") -> dict[str, Any]:
    cfg = load_config("mode_a", "fake").sanitize
    analyzer = PiiAnalyzer(cfg.languages, cfg.custom_patterns, ner=ner)  # type: ignore[arg-type]
    policy = policy_from_config(cfg.policy, cfg.default_action, cfg.score_threshold,
                                analyzer.entity_types())
    gold_n: dict[str, int] = defaultdict(int)
    recalled: dict[str, int] = defaultdict(int)
    protected: dict[str, int] = defaultdict(int)
    pred_n: dict[str, int] = defaultdict(int)
    pred_tp: dict[str, int] = defaultdict(int)
    lang_ok = 0
    for note in notes:
        text = note["text"]
        lang = analyzer.detect_language(text)
        lang_ok += lang == note["language"]
        spans = resolve(analyzer.analyze(text, lang), policy)
        covered = [False] * len(text)
        for s in spans:
            if s.action != "keep":
                for i in range(s.start, s.end):
                    covered[i] = True
        gold = [(g["start"], g["end"], g["entity_type"]) for g in note["spans"]]
        for gs, ge, gt in gold:
            gold_n[gt] += 1
            if any(s.entity_type == gt and _overlap((gs, ge), (s.start, s.end)) for s in spans):
                recalled[gt] += 1
            idx = [i for i in range(gs, ge) if not text[i].isspace()]
            if idx and sum(covered[i] for i in idx) / len(idx) >= PROTECT_FRACTION:
                protected[gt] += 1
        for s in spans:
            pred_n[s.entity_type] += 1
            if any(gt == s.entity_type and _overlap((gs, ge), (s.start, s.end))
                   for gs, ge, gt in gold):
                pred_tp[s.entity_type] += 1

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    per_type = {}
    for t in sorted(set(gold_n) | set(pred_n)):
        per_type[t] = {"gold": gold_n[t], "predicted": pred_n[t],
                       "recall": ratio(recalled[t], gold_n[t]),
                       "protected": ratio(protected[t], gold_n[t]),
                       "precision": ratio(pred_tp[t], pred_n[t])}
    tot = {k: sum(d.values()) for k, d in (("gold", gold_n), ("recalled", recalled),
                                          ("protected", protected), ("pred", pred_n),
                                          ("tp", pred_tp))}
    return {
        "ner_mode": analyzer.ner_mode,
        "notes": len(notes),
        "language_accuracy": ratio(lang_ok, len(notes)),
        "overall": {"gold": tot["gold"], "predicted": tot["pred"],
                    "recall": ratio(tot["recalled"], tot["gold"]),
                    "protected": ratio(tot["protected"], tot["gold"]),
                    "precision": ratio(tot["tp"], tot["pred"])},
        "per_type": per_type,
    }


def format_table(result: dict[str, Any]) -> str:
    rows = [f"ner_mode={result['ner_mode']}  notes={result['notes']}  "
            f"language_accuracy={result['language_accuracy']}",
            f"{'entity_type':<24}{'gold':>6}{'pred':>6}{'recall':>9}{'protected':>11}"
            f"{'precision':>11}"]

    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    for t, m in [*result["per_type"].items(), ("OVERALL", result["overall"])]:
        rows.append(f"{t:<24}{m['gold']:>6}{m['predicted']:>6}{fmt(m['recall']):>9}"
                    f"{fmt(m['protected']):>11}{fmt(m['precision']):>11}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--notes", type=Path, default=HERE / "notes.jsonl")
    ap.add_argument("--ner", choices=["auto", "regex"], default="auto")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    notes = [json.loads(line) for line in args.notes.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    result = evaluate(notes, args.ner)
    print(format_table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
