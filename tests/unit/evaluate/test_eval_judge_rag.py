from __future__ import annotations

import pytest

from zllf.core.schemas import Chunk, Message
from zllf.llm.fake import FakeEmbedder
from zllf.stages.evaluate.judge import (
    abstention_prompt,
    correctness_prompt,
    keyword_abstention,
    parse_abstention,
    parse_verdict,
)
from zllf.stages.evaluate.rag import RagIndex, passage_char_budget, rag_prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"score": 1, "reason": "ok"}', 1.0),
        ('{"score": 0.5}', 0.5),
        ('Here is my grade:\n```json\n{"score": 0, "reason": "wrong"}\n```', 0.0),
        ('{"correct": true}', 1.0),
        ('{"verdict": "incorrect"}', 0.0),
        ("Score: 0.5 because partially right", 0.5),
        ("PARTIALLY CORRECT", 0.5),
        ("The answer is incorrect.", 0.0),
        ("Correct.", 1.0),
        ('{"score": 8}', 0.8),  # 0-10 scale
        ("I cannot decide", None),
        ("", None),
    ],
)
def test_parse_verdict(text: str, expected: float | None) -> None:
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"label": "abstain"}', "abstain"),
        ('{"label": "Fabricate"}', "fabricate"),
        ("Label: hedge", "hedge"),
        ('```json\n{"classification": "fabrication"}\n```', "fabricate"),
        ("abstain or fabricate?", None),
        ("nonsense", None),
    ],
)
def test_parse_abstention(text: str, expected: str | None) -> None:
    assert parse_abstention(text) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("I don't know; this is not in the documents.", "abstain"),
        ("The provided documents do not contain this information.", "abstain"),
        ("Diese Information ist nicht enthalten.", "abstain"),
        ('{"pump": null, "pressure_bar": null}', "abstain"),
        ("", "abstain"),
        ("Could you clarify which model you mean?", "hedge"),
        ("The warranty period is 24 months.", "fabricate"),
    ],
)
def test_keyword_abstention(output: str, expected: str) -> None:
    assert keyword_abstention(output) == expected


def test_judge_prompts_are_fixed_and_clipped() -> None:
    p = correctness_prompt("q?", "ref", "x" * 5000)
    assert p[0].role == "system" and "Rubric" in p[0].content
    assert len(p[1].content) < 3000
    assert abstention_prompt("q?", "a")[0].content.startswith("You classify")


def _chunks() -> list[Chunk]:
    texts = ["Pump P-1 trips at 81 bar.", "Cooling towers need descaling.",
             "The generator G-4 starts within 10 seconds.", "Duplicate text."]
    out = [Chunk(chunk_id=f"d-{i}", doc_id="d", index=i, text=t, n_tokens=5)
           for i, t in enumerate(texts)]
    out[3] = out[3].model_copy(update={"duplicate_of": "d-0"})
    return out


def test_rag_retrieves_relevant_chunk(tmp_path: object) -> None:
    emb = FakeEmbedder()
    index = RagIndex.build(_chunks(), emb, batch_size=2)
    assert len(index) == 3  # duplicate removed
    assert index.backend == "numpy"
    hits = index.search(emb.encode(["When does generator G-4 start?", "pump P-1 trips"]), k=2)
    assert index.chunk_ids[hits[0][0][0]] == "d-2"
    assert index.chunk_ids[hits[1][0][0]] == "d-0"
    assert hits[0][0][1] >= hits[0][1][1]
    assert index.search(emb.encode(["x"]), k=10)[0].__len__() == 3


def test_rag_index_save_has_no_text(tmp_path: object) -> None:
    from pathlib import Path

    d = Path(str(tmp_path)) / "rag"
    RagIndex.build(_chunks(), FakeEmbedder()).save(d)
    assert sorted(p.name for p in d.iterdir()) == ["chunk_ids.json", "embeddings.npy",
                                                   "meta.json"]
    assert "Pump" not in (d / "chunk_ids.json").read_text()


def test_rag_prompt_numbers_passages() -> None:
    prompt = [Message(role="system", content="sys"), Message(role="user", content="Q?")]
    out = rag_prompt(prompt, ["alpha " * 200, "beta"], char_budget=50)
    assert out[0].content == "sys"
    assert "[1] alpha" in out[1].content and "[2] beta" in out[1].content
    assert out[1].content.endswith("Question: Q?")
    assert len(out[1].content) < 600
    assert passage_char_budget(2048, 5) >= 200
