"""LLM-judge prompts (fixed rubric, temperature 0) and robust verdict parsing.

Two judgements:

* **correctness** of an answer vs. a reference -> score in {0, 0.5, 1} (PRD §11.2.4);
* **abstention** class of an answer to an unanswerable question -> ``abstain`` (declines /
  says the information is not available), ``hedge`` (no specific claim, e.g. asks for
  clarification) or ``fabricate`` (asserts specific facts) (P2.6.5).

When the judge output cannot be parsed, callers fall back to heuristics
(:func:`keyword_abstention`, ROUGE-L threshold) and count the fallbacks in the report.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from zllf.core.schemas import Message
from zllf.stages.evaluate.metrics import extract_json

AbstentionLabel = Literal["abstain", "hedge", "fabricate"]
MAX_JUDGE_CHARS = 2000

CORRECTNESS_SYSTEM = (
    "You are a strict grader. Compare a candidate answer with a reference answer to the same "
    "question.\n"
    "Rubric:\n"
    "- score 1: the candidate states the key facts of the reference and adds nothing that "
    "contradicts it.\n"
    "- score 0.5: the candidate is partially correct (some key facts missing or vague) and "
    "contains no contradiction.\n"
    "- score 0: the candidate is wrong, contradicts the reference, is off-topic, or refuses "
    "although the reference contains an answer.\n"
    "Ignore style, length and wording. Reply with JSON only: "
    '{"score": 0 | 0.5 | 1, "reason": "<at most 20 words>"}'
)

ABSTENTION_SYSTEM = (
    "You classify how an assistant responded to a question that CANNOT be answered from the "
    "available documents.\n"
    "Labels:\n"
    '- "abstain": it says it does not know, that the information is not in the documents, or '
    "declines to answer.\n"
    '- "hedge": it gives no specific factual claim (e.g. only asks for clarification or gives '
    "generic advice to consult someone) but does not clearly decline.\n"
    '- "fabricate": it asserts specific facts, steps, numbers or procedures as if they were '
    "known.\n"
    'Reply with JSON only: {"label": "abstain" | "hedge" | "fabricate"}'
)

_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|cannot|can't|could not|couldn't) (?:know|find|answer|locate|"
    r"determine|say|provide|confirm)",
    r"\b(?:not|no) (?:mentioned|covered|contained|included|specified|described|found|"
    r"available|provided|stated|documented)\b",
    r"\bno (?:information|mention|record|reference|details?|data|procedure|entry) (?:about|on|"
    r"for|of|regarding|is|was)\b",
    r"\b(?:does not|doesn't|do not|don't) (?:contain|mention|include|cover|specify|describe|"
    r"provide|exist)\b",
    r"\b(?:is|are) not (?:in|part of) the (?:provided |available )?(?:documents?|manual|"
    r"context|sources?|corpus)\b",
    r"\b(?:unable|not able) to (?:answer|find|provide|determine|locate)\b",
    r"\binsufficient (?:information|context)\b",
    r"\bcannot be (?:answered|determined|found)\b",
    r"\bkeine (?:angabe|information|informationen)\b",
    r"\bnicht (?:enthalten|bekannt|beantworten)\b",
    r"\bweiß (?:ich )?nicht\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)
_HEDGE_RE = re.compile(
    r"\b(?:could you (?:clarify|provide)|can you (?:clarify|specify)|please (?:clarify|"
    r"specify|consult|contact))\b",
    re.IGNORECASE,
)


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_JUDGE_CHARS else text[:MAX_JUDGE_CHARS] + " [...]"


def correctness_prompt(question: str, reference: str, candidate: str) -> list[Message]:
    user = (
        f"Question:\n{_clip(question)}\n\nReference answer:\n{_clip(reference)}\n\n"
        f"Candidate answer:\n{_clip(candidate)}\n\nGrade the candidate. JSON only."
    )
    return [Message(role="system", content=CORRECTNESS_SYSTEM),
            Message(role="user", content=user)]


def abstention_prompt(question: str, candidate: str) -> list[Message]:
    user = (
        f"Question:\n{_clip(question)}\n\nAssistant response:\n{_clip(candidate)}\n\n"
        "Classify the response. JSON only."
    )
    return [Message(role="system", content=ABSTENTION_SYSTEM),
            Message(role="user", content=user)]


_SCORE_KEYS = ("score", "correctness", "correct", "grade", "rating", "verdict")
_SCORE_RE = re.compile(r"\b(?:score|grade|correctness)\b\s*[:=]\s*\"?([01](?:\.\d+)?)", re.I)


def _to_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        v = float(value)
        if v > 1.0 and v <= 10.0:  # some judges answer on a 0-10 scale
            v = v / 10.0
        return min(1.0, max(0.0, v)) if 0.0 <= v <= 1.0 else None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"correct", "yes", "true", "pass"}:
            return 1.0
        if s in {"partially correct", "partial", "partially"}:
            return 0.5
        if s in {"incorrect", "wrong", "no", "false", "fail"}:
            return 0.0
        try:
            return _to_score(float(s))
        except ValueError:
            return None
    return None


def parse_verdict(text: str) -> float | None:
    """Correctness score in [0, 1] from a judge output, or None when unparseable.

    Accepts a JSON object (``score``/``correct``/``verdict`` ...), ``score: 0.5`` style text,
    or a bare CORRECT / PARTIALLY CORRECT / INCORRECT verdict.
    """
    if not text:
        return None
    obj = extract_json(text)
    if isinstance(obj, dict):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        for key in _SCORE_KEYS:
            if key in lowered:
                score = _to_score(lowered[key])
                if score is not None:
                    return score
    m = _SCORE_RE.search(text)
    if m:
        return _to_score(m.group(1))
    s = text.strip().lower()
    if re.search(r"\bpartially correct\b", s):
        return 0.5
    if re.search(r"\bincorrect\b|\bwrong\b", s):
        return 0.0
    if re.search(r"\bcorrect\b", s):
        return 1.0
    return None


_LABELS: tuple[AbstentionLabel, ...] = ("abstain", "hedge", "fabricate")
_LABEL_SYNONYMS: dict[str, AbstentionLabel] = {
    "abstain": "abstain", "abstention": "abstain", "abstained": "abstain",
    "refuse": "abstain", "refusal": "abstain", "decline": "abstain",
    "hedge": "hedge", "hedging": "hedge", "clarify": "hedge",
    "fabricate": "fabricate", "fabrication": "fabricate", "fabricated": "fabricate",
    "answer": "fabricate", "answered": "fabricate", "hallucination": "fabricate",
}


def parse_abstention(text: str) -> AbstentionLabel | None:
    """Abstention label from a judge output, or None when unparseable."""
    if not text:
        return None
    obj = extract_json(text)
    if isinstance(obj, dict):
        for key in ("label", "class", "classification", "verdict", "category"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip().lower() in _LABEL_SYNONYMS:
                return _LABEL_SYNONYMS[val.strip().lower()]
    words = re.findall(r"[a-z]+", text.lower())
    found = [_LABEL_SYNONYMS[w] for w in words if w in _LABEL_SYNONYMS and
             _LABEL_SYNONYMS[w] in _LABELS]
    unique = set(found)
    if len(unique) == 1:
        return found[0]
    return None


def _json_all_empty(text: str) -> bool:
    obj = extract_json(text)
    if not isinstance(obj, dict | list):
        return False

    def empty(v: Any) -> bool:
        if isinstance(v, dict):
            return all(empty(x) for x in v.values())
        if isinstance(v, list):
            return all(empty(x) for x in v)
        return v is None or (isinstance(v, str) and not v.strip())

    return empty(obj)


def keyword_abstention(output: str) -> AbstentionLabel:
    """Heuristic abstention classifier used when the judge verdict is unparseable.

    Abstention phrases (EN/DE) or an all-null JSON object -> ``abstain``; a clarification
    request without numbers -> ``hedge``; anything else non-empty -> ``fabricate``;
    an empty output counts as ``abstain`` (it asserts nothing).
    """
    text = (output or "").strip()
    if not text or _ABSTAIN_RE.search(text) or _json_all_empty(text):
        return "abstain"
    if _HEDGE_RE.search(text) and not re.search(r"\d", text):
        return "hedge"
    return "fabricate"


def verdict_json(score: float) -> str:
    """Canonical judge output (used by tests and scripted fakes)."""
    return json.dumps({"score": score, "reason": "scripted"})
