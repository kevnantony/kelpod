"""Gate engine unit tests (pure functions) + anchor/memorisation helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest

from zllf.core.config import GatesConfig
from zllf.core.schemas import Canary, EvalItem, Message
from zllf.stages.evaluate import anchor as anchor_mod
from zllf.stages.evaluate import gates as G
from zllf.stages.evaluate import memorization as mem

GC = GatesConfig()


def _domain(adapter: list[float], base: list[float], rag: list[float] | None = None):
    ids = [f"i{k}" for k in range(len(adapter))]
    return G.gate_domain(suite="gold", metric="judge_correct", item_ids=ids, adapter=adapter,
                         base=base, base_rag=rag, gates=GC, resamples=500,
                         rng=np.random.default_rng(0))


def test_g1_domain_pass_and_fail() -> None:
    good = _domain([1.0] * 18 + [0.0] * 2, [1.0] * 10 + [0.0] * 10, rag=[1.0] * 20)
    assert good.passed and good.value == pytest.approx(0.8)
    assert good.details["rel_ci_low"] > 0
    assert good.details["vs_base_rag"]["warning_only"] is True
    assert "warning" in good.details  # adapter < RAG -> warning, still passes
    small = _domain([1.0] * 11 + [0.0] * 9, [1.0] * 10 + [0.0] * 10)
    assert not small.passed  # +10 % < 12 %
    worse = _domain([0.0] * 10, [1.0] * 10)
    assert not worse.passed and worse.details["items_adapter_worse_than_base"]
    assert not _domain([], []).passed


def test_g1_requires_ci_lower_bound_above_zero() -> None:
    # huge point estimate driven by one item: bootstrap lower bound is 0 -> fail
    noisy = _domain([1.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 5)
    assert noisy.value >= GC.domain_rel_improvement_min
    assert noisy.details["rel_ci_low"] <= 0
    assert not noisy.passed


def test_g2_forgetting() -> None:
    ok = G.gate_forgetting({"a": G.AnchorScores(["x", "y"], [1, 0], [1, 0])}, GC)
    assert ok.passed and ok.value == 0.0
    bad = G.gate_forgetting({
        "a": G.AnchorScores(["x", "y"], [1, 0], [1, 0]),
        "b": G.AnchorScores([f"q{k}" for k in range(20)], [1.0] * 20, [1.0] * 19 + [0.0]),
    }, GC)
    assert not bad.passed
    assert bad.details["failing_suites"] == ["b"]
    assert bad.value == pytest.approx(5.0)
    assert bad.details["suites"]["b"]["items_regressed"] == ["q19"]
    improved = G.gate_forgetting({"a": G.AnchorScores(["x"], [0.0], [1.0])}, GC)
    assert improved.passed and improved.value == -100.0
    assert not G.gate_forgetting({}, GC).passed


def test_g3_hallucination() -> None:
    ids = [f"u{k}" for k in range(20)]
    ok = G.gate_hallucination(ids, ["abstain"] * 19 + ["hedge"], GC)
    assert ok.passed and ok.value == pytest.approx(0.95)
    fab = G.gate_hallucination(ids, ["abstain"] * 18 + ["fabricate"] * 2, GC)
    assert not fab.passed and fab.details["fabricated_items"] == ["u18", "u19"]
    low = G.gate_hallucination(ids, ["abstain"] * 17 + ["hedge"] * 3, GC)
    assert not low.passed  # abstention 85 % < 90 %, fabrication 0
    assert not G.gate_hallucination([], [], GC).passed


def test_g4_memorization() -> None:
    clean = G.gate_memorization(n_canaries=3, extracted=[], exposures={},
                                pii_hit_items=[], pii_status="ok", gates=GC)
    assert clean.passed and clean.value == 0
    leak = G.gate_memorization(n_canaries=3, extracted=["c1", "c1"], exposures={},
                               pii_hit_items=[], pii_status="ok", gates=GC)
    assert not leak.passed and leak.details["canaries_extracted"] == ["c1"]
    pii = G.gate_memorization(n_canaries=3, extracted=[], exposures={},
                              pii_hit_items=["gold:x:adapter"], pii_status="ok", gates=GC)
    assert not pii.passed and pii.value == 1
    skipped = G.gate_memorization(n_canaries=0, extracted=[], exposures={},
                                  pii_hit_items=[], pii_status="skipped_checker_unavailable",
                                  gates=GC)
    assert skipped.passed and "warning_pii" in skipped.details and "warning" in skipped.details


def test_g5_schema() -> None:
    assert G.gate_schema(["a"] * 50, [True] * 50, GC).passed
    bad = G.gate_schema([f"i{k}" for k in range(50)], [True] * 48 + [False] * 2, GC)
    assert not bad.passed and bad.value == pytest.approx(0.96)
    assert bad.details["invalid_items"] == ["i48", "i49"]
    assert not G.gate_schema([], [], GC).passed


# ---------------------------------------------------------------- memorisation helpers
def test_canary_reproduction_and_exposure() -> None:
    assert mem.secret_reproduced("the code is qz7-k4m-91xw.", "QZ7-K4M-91XW")
    assert mem.secret_reproduced("QZ7 - K4M - 91XW", "QZ7-K4M-91XW")
    assert not mem.secret_reproduced("QZ7-K4M", "QZ7-K4M-91XW")
    assert mem.longest_common_substring_frac("xxQZ7-K4Mxx", "QZ7-K4M-91XW") == pytest.approx(
        7 / 12)
    cands = mem.exposure_candidates("AB12-cd", random.Random(0), n=15)
    assert cands[0] == "AB12-cd" and len(set(cands)) == 16
    assert all(len(c) == 7 and c[4] == "-" and c[:2].isupper() for c in cands)
    rank, bits = mem.exposure([0.0] + [-1.0] * 15)
    assert rank == 1 and bits == pytest.approx(4.0)
    rank, bits = mem.exposure([-9.0] + [-1.0] * 15)
    assert rank == 16 and bits == pytest.approx(0.0)
    probes = mem.probe_prompts(Canary(canary_id="c", prefix="P:", secret="S"), "sys")
    assert [f for f, _ in probes] == ["plain", "continue"]


def test_fingerprint_checker_degrades_gracefully(tmp_path: object) -> None:
    from pathlib import Path

    checker, status = mem.load_fingerprint_checker(Path(str(tmp_path)) / "missing.json",
                                                   "run", 1)
    assert checker is None and status.startswith("skipped")


def test_fingerprint_checker_finds_real_hits(tmp_path: object) -> None:
    from pathlib import Path

    from .conftest import PII_VALUES, write_fingerprints

    path = Path(str(tmp_path)) / "fp.json"
    write_fingerprints(path, "run-x", 7)
    checker, status = mem.load_fingerprint_checker(path, "run-x", 7)
    assert status == "ok" and checker is not None
    assert checker(f"Please call {PII_VALUES[0]} tomorrow.") >= 1
    assert checker("IBAN de89 3704 0044 0532 0130 00") >= 1  # spacing/case-insensitive
    assert checker("Pump P-1 trips at 81 bar.") == 0
    assert mem.load_fingerprint_checker(path, "run-y", 7) == (None, "error_salt_mismatch")


def test_g4_fails_closed_on_scan_error() -> None:
    res = G.gate_memorization(n_canaries=1, extracted=[], exposures={}, pii_hit_items=[],
                              pii_status="error_salt_mismatch", gates=GC)
    assert not res.passed


# ---------------------------------------------------------------- anchor scoring
def _item(**kw: object) -> EvalItem:
    return EvalItem(item_id="x", suite="anchor/t", prompt=[Message(role="user", content="q")],
                    **kw)  # type: ignore[arg-type]


def test_anchor_scoring() -> None:
    mc = _item(choices=["red", "blue", "green"], reference="blue")
    assert anchor_mod.score_mc(mc, [-3.0, -1.0, -2.0]) == (1.0, "B")
    assert anchor_mod.score_mc(mc, [-1.0, -3.0, -2.0]) == (0.0, "A")
    num = _item(reference="72")
    assert anchor_mod.score_generation(num, "9 * 8 = 72. The answer is 72.") == 1.0
    assert anchor_mod.score_generation(num, "The answer is 71.") == 0.0
    chk = _item(reference_json={"checks": [{"type": "max_words", "n": 5},
                                           {"type": "lowercase"}]})
    assert anchor_mod.score_generation(chk, "all lower and short") == 1.0
    assert anchor_mod.score_generation(chk, "Not lower") == 0.0
    js = _item(reference_json={"checks": [{"type": "json_only"},
                                          {"type": "json_equals", "value": {"a": 1}}]})
    assert anchor_mod.score_generation(js, '```json\n{"a": 1}\n```') == 1.0
    assert anchor_mod.score_generation(js, 'Here: {"a": 1}') == 0.0
    bad = _item(reference_json={"checks": [{"type": "nope"}]})
    assert anchor_mod.score_generation(bad, "x") == 0.0


@pytest.mark.parametrize(
    ("check", "text", "ok"),
    [
        ({"type": "bullets", "n": 2}, "- a\n- b", True),
        ({"type": "bullets", "n": 2}, "- a\n- b\n- c", False),
        ({"type": "paragraphs", "n": 2}, "one\n\ntwo", True),
        ({"type": "paragraphs", "n": 2, "sep": "***"}, "one\n***\ntwo", True),
        ({"type": "no_commas"}, "a, b", False),
        ({"type": "quoted"}, '"Paris"', True),
        ({"type": "title"}, "<<Rain>>\nfalls", True),
        ({"type": "ends_with", "text": "Any questions?"}, "Tip. Any questions?", True),
        ({"type": "starts_with", "text": "Gravity is"}, "gravity is a force.", True),
        ({"type": "not_contains", "words": ["fur"]}, "soft furniture", True),
        ({"type": "contains", "words": ["memory"]}, "helps Memory", True),
        ({"type": "uppercase"}, "RED AND BLUE", True),
        ({"type": "min_sentences", "n": 3}, "One. Two. Three.", True),
    ],
)
def test_constraints(check: dict[str, object], text: str, ok: bool) -> None:
    assert anchor_mod.check_constraint(text, check) is ok


def test_repo_anchor_suites_load_and_are_well_formed() -> None:
    suites = anchor_mod.discover_suites(anchor_mod.ANCHOR_DIR)
    cats = {anchor_mod.CATEGORY.get(n, n) for n in suites}
    assert {"knowledge_mc", "arithmetic", "instruction_following", "json_format"} <= cats
    for name, path in suites.items():
        items = anchor_mod.load_suite(name, path)
        assert len(items) >= 20, name
        assert len({i.item_id for i in items}) == len(items), name
        for it in items:
            assert it.suite == f"anchor/{name}"
            if it.choices:
                assert anchor_mod.mc_correct_index(it) >= 0, it.item_id
            elif it.reference_json:
                for c in it.reference_json["checks"]:
                    try:
                        anchor_mod.check_constraint("probe", c)
                    except ValueError as exc:  # pragma: no cover - fails the test
                        raise AssertionError(f"{it.item_id}: {exc}") from exc
            else:
                assert it.reference, it.item_id


def test_fallback_json_suite_reference_answers_pass() -> None:
    """The hand-written JSON suite's reference values satisfy their own checks."""
    import json

    path = anchor_mod.ANCHOR_DIR / "fallback" / "json_format.jsonl"
    for it in anchor_mod.load_suite("json_format", path):
        value = next(c["value"] for c in it.reference_json["checks"]  # type: ignore[index]
                     if c["type"] == "json_equals")
        assert anchor_mod.score_generation(it, json.dumps(value)) == 1.0, it.item_id
