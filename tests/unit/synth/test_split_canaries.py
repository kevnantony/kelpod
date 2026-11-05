from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from zllf.core.config import SplitConfig
from zllf.stages.synth.canaries import make_canaries
from zllf.stages.synth.split import carve_val, plan_split

RATIOS = SplitConfig()


@settings(max_examples=150, deadline=None)
@given(
    weights=st.dictionaries(
        st.text(alphabet="abcdef0123456789", min_size=4, max_size=8),
        st.integers(min_value=0, max_value=50),
        min_size=1,
        max_size=40,
    ),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_split_is_by_document_and_complete(weights: dict[str, int], seed: int) -> None:
    plan = plan_split(weights, RATIOS, random.Random(seed))
    assert set(plan.doc_split) == set(weights)  # every doc assigned exactly once
    train, val, test = (set(plan.docs(s)) for s in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train  # at least one training document
    n = len(weights)
    if n >= 2:
        assert test
    if n >= 3:
        assert val
    # deterministic
    assert plan_split(weights, RATIOS, random.Random(seed)).doc_split == plan.doc_split


def test_split_ratios_roughly_respected_on_large_corpus() -> None:
    weights = {f"doc{i:03d}": 10 for i in range(100)}
    plan = plan_split(weights, RATIOS, random.Random(0))
    assert 60 <= len(plan.docs("train")) <= 80
    assert 10 <= len(plan.docs("test")) <= 20
    assert 10 <= len(plan.docs("val")) <= 20


def test_tiny_corpora() -> None:
    one = plan_split({"a": 5}, RATIOS, random.Random(0))
    assert one.docs("train") == ["a"] and any("tiny corpus" in w for w in one.warnings)
    assert any("no test documents" in w for w in one.warnings)
    two = plan_split({"a": 5, "b": 5}, RATIOS, random.Random(0))
    assert len(two.docs("train")) == 1 and len(two.docs("test")) == 1 and not two.docs("val")
    three = plan_split({"a": 5, "b": 5, "c": 5}, RATIOS, random.Random(0))
    assert [len(three.docs(s)) for s in ("train", "val", "test")] == [1, 1, 1]
    moved = carve_val([f"e{i}" for i in range(10)], 0.15, random.Random(0))
    assert len(moved) == 2
    assert carve_val(["only"], 0.15, random.Random(0)) == set()


def test_canaries_distinct_train_only_and_deterministic() -> None:
    canaries, samples = make_canaries(10, random.Random("1234:canaries"), "SYS")
    assert len({c.prefix for c in canaries}) == 10
    assert len({c.secret for c in canaries}) == 10
    for c, s in zip(canaries, samples, strict=True):
        assert s.split == "train" and s.task == "canary"
        assert s.messages[0].content == "SYS"
        assert s.messages[2].content == f"{c.prefix} {c.secret}."
        assert c.secret not in s.messages[1].content
    again, _ = make_canaries(10, random.Random("1234:canaries"), "SYS")
    assert again == canaries
    other, _ = make_canaries(10, random.Random("99:canaries"), "SYS")
    assert {c.secret for c in other}.isdisjoint({c.secret for c in canaries})
