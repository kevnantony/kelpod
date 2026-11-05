"""Pure evaluation metrics: ROUGE-L, exact match, JSON field P/R/F1, schema validity and
bootstrap confidence intervals (P2.6.3, FR-14, FR-18).

Everything here is deterministic and dependency-light: ROUGE-L is computed directly
(LCS F-measure on lower-cased word tokens, equivalent to ``rouge_score``'s ``rougeL``
without stemming) so results do not depend on an optional package; ``jsonschema`` is used
when installed, otherwise a small validator covering the common keywords.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------- text metrics
def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 (LCS-based) between two strings, in [0, 1]."""
    p, r = tokenize(prediction), tokenize(reference)
    if not p and not r:
        return 1.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lower, strip punctuation and articles, squash spaces."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def last_number(text: str) -> float | None:
    """Last number in a text (``"1,234.5"`` -> 1234.5), or None."""
    found = _NUMBER.findall(text or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two (n, d) matrices (inputs need not be normalised)."""
    if a.size == 0:
        return np.zeros(0, dtype=np.float32)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) == 0, 1.0, na * nb)
    return np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0)


# ---------------------------------------------------------------- JSON (Mode C)
def extract_json(text: str) -> Any | None:
    """Parse JSON from a model output: whole text, a fenced block, or the first balanced
    ``{...}`` / ``[...]`` span. Returns None when nothing parses."""
    if text is None:
        return None
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
            start = text.find(opener, start + 1)
    return None


def _norm_leaf(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, int | float):
        return repr(float(value))
    return normalize_answer(str(value))


def flatten_json(obj: Any, prefix: str = "") -> Counter[tuple[str, str]]:
    """Multiset of (path, normalised leaf value). List elements share the path ``p[]``
    (order-insensitive). Null / empty leaves are omitted (they assert nothing)."""
    out: Counter[tuple[str, str]] = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_json(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += flatten_json(v, f"{prefix}[]")
    elif obj is None or (isinstance(obj, str) and not obj.strip()):
        pass
    else:
        out[(prefix, _norm_leaf(obj))] += 1
    return out


def field_prf(prediction: Any, reference: Any) -> tuple[float, float, float]:
    """Field-level precision, recall, F1 between a predicted and a reference JSON value."""
    ref = flatten_json(reference)
    pred = flatten_json(prediction) if prediction is not None else Counter()
    if not ref and not pred:
        return 1.0, 1.0, 1.0
    tp = sum((ref & pred).values())
    p = tp / sum(pred.values()) if pred else 0.0
    r = tp / sum(ref.values()) if ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def json_exact_match(prediction: Any, reference: Any) -> float:
    if prediction is None:
        return 0.0
    return float(flatten_json(prediction) == flatten_json(reference))


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _mini_validate(value: Any, schema: dict[str, Any]) -> bool:
    """Subset of JSON Schema: type, enum, const, properties, required,
    additionalProperties=false, items, minItems/maxItems, anyOf/oneOf."""
    if not isinstance(schema, dict):
        return True
    if "anyOf" in schema and not any(_mini_validate(value, s) for s in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(_mini_validate(value, s) for s in schema["oneOf"]) != 1:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES.get(x, lambda _v: True)(value) for x in types):
            return False
    if isinstance(value, dict):
        props: dict[str, Any] = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        for k, sub in props.items():
            if k in value and not _mini_validate(value[k], sub):
                return False
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            return False
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return False
        items = schema.get("items")
        if isinstance(items, dict) and not all(_mini_validate(v, items) for v in value):
            return False
    return True


def validate_schema(value: Any, schema: dict[str, Any] | None) -> bool:
    """True if ``value`` is valid against ``schema`` (None schema: any JSON object/array)."""
    if value is None:
        return False
    if schema is None:
        return isinstance(value, dict | list)
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - depends on the environment
        return _mini_validate(value, schema)
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        return _mini_validate(value, schema)
    return True


def schema_valid_output(text: str, schema: dict[str, Any] | None) -> bool:
    return validate_schema(extract_json(text), schema)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Returns (nan, nan) for empty input."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if resamples <= 0 or arr.size == 1:
        m = float(arr.mean())
        return m, m
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def relative_improvement(adapter_mean: float, base_mean: float, floor: float = 0.01) -> float:
    """(adapter - base) / max(base, floor). The floor avoids division by zero when the base
    scores 0 (documented in the gate details)."""
    return (adapter_mean - base_mean) / max(base_mean, floor)


def paired_relative_ci(
    adapter: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    floor: float = 0.01,
) -> tuple[float, float, float]:
    """Point estimate and paired-bootstrap CI of the relative improvement of ``adapter``
    over ``base`` (same items, same order)."""
    a = np.asarray(adapter, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    if a.size == 0:
        nan = float("nan")
        return nan, nan, nan
    point = relative_improvement(float(a.mean()), float(b.mean()), floor)
    if resamples <= 0 or a.size == 1:
        return point, point, point
    idx = rng.integers(0, a.size, size=(resamples, a.size))
    am, bm = a[idx].mean(axis=1), b[idx].mean(axis=1)
    rel = (am - bm) / np.maximum(bm, floor)
    lo, hi = np.quantile(rel, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarize(
    values: Sequence[float] | np.ndarray, resamples: int, rng: np.random.Generator
) -> dict[str, float | int | None]:
    """{mean, ci_low, ci_high, n} with a 95 % bootstrap CI (None for empty input)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    lo, hi = bootstrap_ci(arr, resamples, rng)
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": int(arr.size)}
