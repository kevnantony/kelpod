from __future__ import annotations

import numpy as np
import pytest

from zllf.stages.evaluate.metrics import _mini_validate as mini_validate
from zllf.stages.evaluate.metrics import (
    bootstrap_ci,
    extract_json,
    field_prf,
    json_exact_match,
    last_number,
    paired_relative_ci,
    rouge_l,
    schema_valid_output,
    summarize,
    validate_schema,
)


def test_rouge_l_known_values() -> None:
    assert rouge_l("the pump trips at 80 bar", "the pump trips at 80 bar") == 1.0
    assert rouge_l("completely different", "the pump trips") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P = R = 3/4
    assert rouge_l("a b c d", "a c d e") == pytest.approx(0.75)
    assert rouge_l("", "") == 1.0


def test_rouge_l_matches_rouge_score_when_installed() -> None:
    rs = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    pairs = [("The valve opens at 3 bar after reset", "valve opens at three bar"),
             ("Pump P-12 trips at 80 bar.", "P-12 trips at 80 bar")]
    for pred, ref in pairs:
        assert rouge_l(pred, ref) == pytest.approx(scorer.score(ref, pred)["rougeL"].fmeasure)


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Result: {"a": "x}"} trailing') == {"a": "x}"}
    assert extract_json("no json here") is None


def test_field_prf_and_exact_match() -> None:
    ref = {"name": "Ada", "age": 36, "tags": ["x", "y"]}
    assert field_prf(ref, ref) == (1.0, 1.0, 1.0)
    p, r, f = field_prf({"name": "ada", "age": 35}, ref)  # name matches (normalised)
    assert p == pytest.approx(0.5) and r == pytest.approx(0.25)
    assert f == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert field_prf(None, ref) == (0.0, 0.0, 0.0)
    assert field_prf({"tags": ["y", "x"]}, {"tags": ["x", "y"]})[2] == 1.0  # order-free
    assert json_exact_match({"age": 36.0, "name": "ADA", "tags": ["x", "y"]}, ref) == 1.0
    assert json_exact_match(None, ref) == 0.0


SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}, "s": {"enum": ["a", "b"]}},
          "required": ["n"], "additionalProperties": False}


@pytest.mark.parametrize("validator", [validate_schema, mini_validate])
def test_schema_validity(validator: object) -> None:
    v = validator  # type: ignore[assignment]
    assert v({"n": 1, "s": "a"}, SCHEMA)  # type: ignore[operator]
    assert not v({"n": "1"}, SCHEMA)  # type: ignore[operator]
    assert not v({"s": "a"}, SCHEMA)  # type: ignore[operator]
    assert not v({"n": 1, "extra": 0}, SCHEMA)  # type: ignore[operator]
    assert not v({"n": 1, "s": "c"}, SCHEMA)  # type: ignore[operator]
    assert not v({"n": True}, SCHEMA)  # type: ignore[operator]


def test_schema_valid_output_text() -> None:
    assert schema_valid_output('```json\n{"n": 2}\n```', SCHEMA)
    assert not schema_valid_output("not json", SCHEMA)
    assert schema_valid_output('{"x": 1}', None)  # no schema: any JSON object
    assert not validate_schema(None, SCHEMA)


def test_last_number() -> None:
    assert last_number("so 3 + 4 = 7. The answer is 1,234.") == 1234.0
    assert last_number("none") is None


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    vals = np.r_[np.ones(30), np.zeros(20)]
    a = bootstrap_ci(vals, 500, np.random.default_rng(7))
    b = bootstrap_ci(vals, 500, np.random.default_rng(7))
    assert a == b
    assert a[0] < 0.6 < a[1]
    assert bootstrap_ci([], 100, np.random.default_rng(0))[0] != bootstrap_ci(
        [], 100, np.random.default_rng(0))[0]  # NaN
    s = summarize(vals, 200, np.random.default_rng(1))
    assert s["n"] == 50 and s["mean"] == pytest.approx(0.6)
    assert summarize([], 10, np.random.default_rng(1))["mean"] is None


def test_paired_relative_ci() -> None:
    adapter = [1.0] * 8 + [0.0] * 2
    base = [1.0] * 5 + [0.0] * 5
    point, lo, hi = paired_relative_ci(adapter, base, 1000, np.random.default_rng(3))
    assert point == pytest.approx(0.6)
    assert lo <= point <= hi
    again = paired_relative_ci(adapter, base, 1000, np.random.default_rng(3))
    assert again == (point, lo, hi)
    # base all zero -> floor avoids division by zero
    p0, _, _ = paired_relative_ci([1.0, 1.0], [0.0, 0.0], 10, np.random.default_rng(0))
    assert p0 == pytest.approx(100.0)
