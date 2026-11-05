"""Stage-level runs of EvaluateStage on fake upstream artefacts with scripted fake models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zllf.core.errors import ZLLFError
from zllf.core.offline import block_egress
from zllf.core.schemas import EvalItem, EvalPrediction, GateResult, read_jsonl
from zllf.llm.fake import FakeLLM
from zllf.stages import get_stage
from zllf.stages.evaluate.stage import EvaluateStage

from .conftest import CANARIES, SCHEMA, MakeCtx, make_student, scripted_judge


def _run(ctx: Any, tmp_path: Path) -> tuple[dict[str, GateResult], Path, Any]:
    out = tmp_path / "out"
    out.mkdir()
    with block_egress():
        result = EvaluateStage().run(ctx, out)
    gates = {g["gate"]: GateResult.model_validate(g)
             for g in json.loads((out / "gates.json").read_text())}
    return gates, out, result


def test_registry_resolves_stage() -> None:
    assert isinstance(get_stage("evaluate"), EvaluateStage)


def test_good_adapter_passes_all_gates(make_ctx: MakeCtx, tmp_path: Path) -> None:
    student = make_student()
    ctx = make_ctx(student=student, judge=FakeLLM("judge", scripted_judge))
    gates, out, result = _run(ctx, tmp_path)

    assert set(gates) == {"G1_domain", "G2_forgetting", "G3_hallucination", "G4_memorization"}
    assert all(g.passed for g in gates.values()), {k: g.details for k, g in gates.items()}
    assert result.metrics["all_gates_passed"] is True
    assert student.closed == 1

    g1 = gates["G1_domain"]
    assert g1.details["suite"] == "domain_test" and g1.details["metric"] == "judge_correct"
    assert g1.details["adapter_mean"] == 1.0 and g1.details["base_mean"] == 0.0
    # the RAG baseline is also perfect here -> warning only, gate still passes
    assert g1.details["vs_base_rag"]["base_rag_mean"] == 1.0

    preds = read_jsonl(out / "predictions.jsonl", EvalPrediction)
    variants = {(p.suite, p.variant) for p in preds}
    for v in ("base", "base_rag", "adapter", "adapter_rag"):
        assert ("domain_test", v) in variants
    assert ("unanswerable", "base_rag") not in variants
    assert ("anchor/arithmetic", "adapter") in variants
    assert ("anchor/knowledge_mc", "base") in variants
    n_probes = sum(1 for p in preds if p.suite == "canary")
    assert n_probes == len(CANARIES) * 2 * 5  # 2 prompt forms x (greedy + 4 sampled)

    results = json.loads((out / "results.json").read_text())
    dom = results["variants"]["adapter"]["domain_test"]
    assert set(dom) >= {"rouge_l", "embed_cos", "judge_correct"}
    assert dom["judge_correct"]["n"] == 8
    lo, hi = dom["judge_correct"]["ci"]
    assert lo <= dom["judge_correct"]["value"] <= hi
    assert results["variants"]["adapter"]["unanswerable"]["abstain"]["value"] == 1.0
    assert set(results["variants"]) == {"base", "base_rag", "adapter", "adapter_rag"}

    # item ids line up with synth example ids, and items.jsonl carries every evaluated item
    items = {i.item_id: i for i in read_jsonl(out / "items.jsonl", EvalItem)}
    assert {f"t-{k:03d}" for k in range(8)} | {f"u-{k:03d}" for k in range(4)} <= set(items)
    assert {p.item_id for p in preds if p.suite in ("domain_test", "unanswerable")} <= set(items)
    expected = {"G1_domain": "higher_is_better", "G2_forgetting": "lower_is_better",
                "G3_hallucination": "higher_is_better", "G4_memorization": "lower_is_better"}
    assert {k: g.details["direction"] for k, g in gates.items()} == expected

    report = json.loads((out / "eval_report.json").read_text())
    assert report["rag"]["enabled"] and report["rag"]["n_chunks"] == 9
    assert report["counts"]["unanswerable"] == 4 + 20  # held-out + adversarial
    assert report["judge_fallbacks"] == {"correctness": 0, "abstention": 0}
    assert (out / "rag_index" / "embeddings.npy").exists()
    # G4 exposure is reported per canary
    assert set(gates["G4_memorization"].details["exposure"]) == {c.canary_id for c in CANARIES}
    assert gates["G4_memorization"].details["pii_scan"] == "ok"
    assert gates["G4_memorization"].details["pii_hits"] == 0
    # no raw document text in the report
    assert "trips at" not in (out / "eval_report.json").read_text()


def test_stage_is_deterministic(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge))
    _, out1, _ = _run(ctx, tmp_path)
    out2 = tmp_path / "out2"
    out2.mkdir()
    EvaluateStage().run(ctx, out2)
    for name in ("predictions.jsonl", "results.json", "gates.json"):
        assert (out1 / name).read_text() == (out2 / name).read_text(), name


def test_fabricating_adapter_fails_g3(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(adapter_fabricates=True),
                   judge=FakeLLM("judge", scripted_judge))
    gates, _, result = _run(ctx, tmp_path)
    g3 = gates["G3_hallucination"]
    assert not g3.passed
    assert g3.details["fabrication_rate"] > 0.05
    assert any(i.startswith("adv-") for i in g3.details["fabricated_items"])
    assert result.metrics["all_gates_passed"] is False


def test_canary_echoing_adapter_fails_g4(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(adapter_leaks_canary=True),
                   judge=FakeLLM("judge", scripted_judge))
    gates, _, _ = _run(ctx, tmp_path)
    g4 = gates["G4_memorization"]
    assert not g4.passed
    assert g4.details["canaries_extracted"] == sorted(c.canary_id for c in CANARIES)
    assert g4.value >= len(CANARIES)


def test_forgetting_adapter_fails_g2(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(adapter_forgets=True),
                   judge=FakeLLM("judge", scripted_judge))
    gates, _, _ = _run(ctx, tmp_path)
    g2 = gates["G2_forgetting"]
    assert not g2.passed
    assert set(g2.details["failing_suites"]) == {"arithmetic", "knowledge_mc"}
    assert g2.details["suites"]["arithmetic"]["drop_points"] == pytest.approx(100.0)


def test_adapter_without_domain_gain_fails_g1(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(adapter_domain=False),
                   judge=FakeLLM("judge", scripted_judge))
    gates, _, _ = _run(ctx, tmp_path)
    assert not gates["G1_domain"].passed
    assert gates["G1_domain"].value == pytest.approx(0.0)


def test_pii_fingerprint_hits_fail_g4(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(adapter_leaks_pii=True),
                   judge=FakeLLM("judge", scripted_judge))
    gates, _, _ = _run(ctx, tmp_path)
    g4 = gates["G4_memorization"]
    assert g4.details["pii_scan"] == "ok"
    assert not g4.passed and g4.details["pii_hits"] == 16  # adapter + adapter_rag
    assert all(i.startswith("domain_test:t-") for i in g4.details["pii_hit_items"])
    assert {i.rsplit(":", 1)[1] for i in g4.details["pii_hit_items"]} == {"adapter", "adapter_rag"}


def test_fingerprint_salt_mismatch_fails_closed(make_ctx: MakeCtx, tmp_path: Path) -> None:
    from .conftest import write_fingerprints

    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge))
    write_fingerprints(ctx.paths.stage("sanitize") / "pii_fingerprints.json", "other-run", 1)
    gates, _, _ = _run(ctx, tmp_path)
    g4 = gates["G4_memorization"]
    assert g4.details["pii_scan"] == "error_salt_mismatch"
    assert not g4.passed


def test_missing_fingerprint_file_warns_but_passes(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge))
    (ctx.paths.stage("sanitize") / "pii_fingerprints.json").unlink()
    gates, out, _ = _run(ctx, tmp_path)
    g4 = gates["G4_memorization"]
    assert g4.passed and g4.details["pii_scan"] == "skipped_no_fingerprint_file"
    assert "warning_pii" in g4.details
    assert any("raw-PII" in w for w in json.loads((out / "eval_report.json").read_text())[
        "warnings"])


def test_gold_set_is_primary_and_unanswerable_gold_goes_to_g3(
        make_ctx: MakeCtx, tmp_path: Path) -> None:
    from .conftest import answer, question

    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge))
    gold = [{"question": question(i), "reference_answer": answer(i)} for i in range(5)]
    gold.append({"question": "What is the colour of pump P-99?", "answerable": False})
    (ctx.paths.root / "gold.jsonl").write_text(
        "\n".join(json.dumps(g) for g in gold) + "\n", encoding="utf-8")
    gates, out, _ = _run(ctx, tmp_path)
    assert gates["G1_domain"].details["suite"] == "gold"
    assert gates["G1_domain"].details["n"] == 5
    report = json.loads((out / "eval_report.json").read_text())
    assert report["gold_set"] is True and report["counts"]["unanswerable"] == 4 + 1 + 20


def test_fake_default_models_run_end_to_end(make_ctx: MakeCtx, tmp_path: Path) -> None:
    """Plain factory fakes (unparseable judge -> heuristic fallbacks) still complete."""
    ctx = make_ctx()
    gates, out, result = _run(ctx, tmp_path)
    report = json.loads((out / "eval_report.json").read_text())
    assert report["judge_fallbacks"]["correctness"] > 0
    assert report["judge_fallbacks"]["abstention"] > 0
    assert set(gates) == {"G1_domain", "G2_forgetting", "G3_hallucination", "G4_memorization"}
    assert result.metrics["n_predictions"] > 0


def test_max_test_items_limits_synthetic_suites(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge),
                   overrides={"eval": {"max_test_items": 3}})
    _, out, _ = _run(ctx, tmp_path)
    counts = json.loads((out / "eval_report.json").read_text())["counts"]
    assert counts["domain_test"] == 3
    assert counts["unanswerable"] == 3 + 20


def test_mode_c_schema_gate(make_ctx: MakeCtx, tmp_path: Path) -> None:
    ctx = make_ctx(student=make_student(mode_c=True), judge=FakeLLM("judge", scripted_judge),
                   profile="mode_c", overrides={"json_schema": SCHEMA})
    gates, out, _ = _run(ctx, tmp_path)
    assert "G5_schema" in gates and gates["G5_schema"].passed
    assert gates["G5_schema"].value == 1.0
    g1 = gates["G1_domain"]
    assert g1.details["metric"] == "field_f1" and g1.passed
    results = json.loads((out / "results.json").read_text())
    base = results["variants"]["base"]["domain_test"]
    assert base["schema_valid"]["value"] == 0.0  # base omits a required field
    assert results["variants"]["adapter"]["domain_test"]["exact_match"]["value"] == 1.0
    assert gates["G5_schema"].details["direction"] == "higher_is_better"


def test_missing_adapter_raises(make_ctx: MakeCtx, tmp_path: Path) -> None:
    import shutil

    ctx = make_ctx(student=make_student(), judge=FakeLLM("judge", scripted_judge))
    shutil.rmtree(ctx.paths.stage("train") / "adapter")
    with pytest.raises(ZLLFError):
        _run(ctx, tmp_path)
