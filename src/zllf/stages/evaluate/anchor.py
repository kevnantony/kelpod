"""General-capability anchor suites for the forgetting gate (G-2, P2.6.4).

Suites are JSONL files of :class:`EvalItem` under ``bench/anchor/``:

* ``bench/anchor/<suite>.jsonl`` — fetched subsets (``scripts/fetch_bench.py``);
* ``bench/anchor/fallback/<suite>.jsonl`` — small hand-written suites committed in the repo,
  used for every *category* not covered by a fetched suite, so evaluation always works
  offline.

Item conventions (see ``bench/anchor/README.md``):

* multiple choice: ``choices`` = option texts, ``reference`` = correct option text, the
  prompt lists the options as ``A. ...``; scored with ``loglikelihood`` of the letters;
* numeric: ``reference`` is a number -> last number in the output must match;
* constraint checks: ``reference_json = {"checks": [{"type": ..., ...}, ...]}`` -> all checks
  must pass (instruction following, JSON formatting);
* otherwise: normalised containment of ``reference`` in the output.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zllf.core.config import REPO_ROOT
from zllf.core.schemas import EvalItem, read_jsonl
from zllf.stages.evaluate.metrics import (
    extract_json,
    last_number,
    normalize_answer,
    validate_schema,
)

ANCHOR_DIR = REPO_ROOT / "bench" / "anchor"
LETTERS = string.ascii_uppercase

# suite name -> capability category (a fetched suite replaces the fallback of its category)
CATEGORY: dict[str, str] = {
    "mmlu": "knowledge_mc",
    "arc_easy": "knowledge_mc",
    "knowledge_mc": "knowledge_mc",
    "gsm8k": "arithmetic",
    "arithmetic": "arithmetic",
    "ifeval": "instruction_following",
    "instruction_following": "instruction_following",
    "json_schema": "json_format",
    "json_format": "json_format",
}


def discover_suites(anchor_dir: Path = ANCHOR_DIR) -> dict[str, Path]:
    """suite name -> file. Fetched suites first; fallbacks fill uncovered categories."""
    found: dict[str, Path] = {}
    covered: set[str] = set()
    if anchor_dir.exists():
        for path in sorted(anchor_dir.glob("*.jsonl")):
            found[path.stem] = path
            covered.add(CATEGORY.get(path.stem, path.stem))
    fallback = anchor_dir / "fallback"
    if fallback.exists():
        for path in sorted(fallback.glob("*.jsonl")):
            cat = CATEGORY.get(path.stem, path.stem)
            if cat not in covered and path.stem not in found:
                found[path.stem] = path
                covered.add(cat)
    return found


def load_suite(name: str, path: Path) -> list[EvalItem]:
    """Load a suite, forcing ``suite = anchor/<name>`` and unique item ids."""
    items = read_jsonl(path, EvalItem)
    out: list[EvalItem] = []
    for i, item in enumerate(items):
        item_id = item.item_id or f"{name}-{i:04d}"
        out.append(item.model_copy(update={"suite": f"anchor/{name}", "item_id": item_id}))
    return out


# ---------------------------------------------------------------- scoring
def mc_letters(item: EvalItem) -> list[str]:
    return [LETTERS[i] for i in range(len(item.choices or []))]


def mc_correct_index(item: EvalItem) -> int:
    choices = item.choices or []
    ref = item.reference or ""
    if ref in choices:
        return choices.index(ref)
    if len(ref) == 1 and ref.upper() in LETTERS[: len(choices)]:
        return LETTERS.index(ref.upper())
    norm = [normalize_answer(c) for c in choices]
    return norm.index(normalize_answer(ref)) if normalize_answer(ref) in norm else -1


def score_mc(item: EvalItem, logliks: Sequence[float]) -> tuple[float, str]:
    """(1/0 correctness, chosen letter) given one log-likelihood per option letter."""
    best = max(range(len(logliks)), key=lambda i: (logliks[i], -i))
    return float(best == mc_correct_index(item)), LETTERS[best]


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def check_constraint(output: str, check: dict[str, Any]) -> bool:
    """Evaluate one verifiable constraint (IFEval-style) on an output."""
    t = check.get("type")
    text = output.strip()
    low = text.lower()
    n: Any = check.get("n")
    if t == "max_words":
        return len(_words(text)) <= int(n)
    if t == "min_words":
        return len(_words(text)) >= int(n)
    if t == "max_sentences":
        return len(_sentences(text)) <= int(n)
    if t == "min_sentences":
        return len(_sentences(text)) >= int(n)
    if t == "contains":
        return all(str(w).lower() in low for w in check.get("words", []))
    if t == "not_contains":
        return not any(re.search(rf"\b{re.escape(str(w).lower())}\b", low)
                       for w in check.get("words", []))
    if t == "lowercase":
        return text == text.lower() and any(c.isalpha() for c in text)
    if t == "uppercase":
        return text == text.upper() and any(c.isalpha() for c in text)
    if t == "no_commas":
        return "," not in text
    if t == "starts_with":
        return low.startswith(str(check.get("text", "")).lower())
    if t == "ends_with":
        return low.rstrip().endswith(str(check.get("text", "")).lower().rstrip())
    if t == "bullets":
        bullets = [ln for ln in text.splitlines() if re.match(r"^\s*[-*•]\s+", ln)]
        return len(bullets) == int(n)
    if t == "paragraphs":
        sep = str(check.get("sep", "\n\n"))
        parts = [p for p in re.split(r"\s*\*\*\*\s*" if sep == "***" else r"\n\s*\n", text)
                 if p.strip()]
        return len(parts) == int(n)
    if t == "quoted":
        return len(text) >= 2 and text.startswith('"') and text.endswith('"')
    if t == "title":
        return re.search(r"<<[^\n]+>>", text) is not None
    if t == "json":
        obj = extract_json(text)
        schema = check.get("schema")
        return obj is not None and (validate_schema(obj, schema) if schema else True)
    if t == "json_only":
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
        obj = extract_json(stripped)
        return obj is not None and stripped[:1] in "{[" and stripped[-1:] in "}]"
    if t == "json_equals":
        obj = extract_json(text)
        return obj == check.get("value")
    raise ValueError(f"unknown constraint type {t!r}")


def score_generation(item: EvalItem, output: str) -> float:
    """1/0 correctness of a generated answer for a non-MC anchor item."""
    rj = item.reference_json or {}
    checks = rj.get("checks")
    if isinstance(checks, list) and checks:
        try:
            return float(all(check_constraint(output, c) for c in checks))
        except (ValueError, TypeError, KeyError):
            return 0.0
    ref = item.reference or ""
    ref_num = last_number(ref) if re.fullmatch(r"\s*-?[\d,]+(?:\.\d+)?\s*", ref) else None
    if ref_num is not None:
        got = last_number(output)
        return float(got is not None and abs(got - ref_num) < 1e-6)
    if not ref:
        return 0.0
    return float(normalize_answer(ref) in normalize_answer(output))
