from __future__ import annotations

import itertools
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from zllf.stages.ingest.chunker import chunk_blocks, split_long_text, split_sentences
from zllf.stages.ingest.model import ParsedBlock
from zllf.stages.ingest.tokens import HeuristicCounter, get_token_counter

COUNTER = HeuristicCounter(4.0)


def H(text: str, level: int) -> ParsedBlock:
    return ParsedBlock("heading", text, level=level)


def P(text: str, page: int | None = None) -> ParsedBlock:
    return ParsedBlock("paragraph", text, page=page)


def test_golden_section_and_paragraph_boundaries() -> None:
    p1, p2, p3, p4, p5 = ("a" * 80, "b" * 60, "c" * 40, "d" * 100, "e" * 20)
    table = "| k | v |\n|---|---|\n| x | 1 |"
    blocks = [
        H("Intro", 1), P(p1, 1), P(p2, 1), P(p3, 2),
        H("Detail", 2), P(p4, 2),
        H("Next", 1), ParsedBlock("table", "k\tv\nx\t1", page=3, table_markdown=table),
        P(p5, 3),
    ]
    chunks = chunk_blocks(blocks, COUNTER, target_tokens=40, max_tokens=60)
    got = [(c.text, c.heading_path, c.page_start, c.page_end, c.is_table, c.n_tokens)
           for c in chunks]
    assert got == [
        (f"{p1}\n\n{p2}", ["Intro"], 1, 1, False, 36),
        (p3, ["Intro"], 2, 2, False, 10),
        (p4, ["Intro", "Detail"], 2, 2, False, 25),
        (table, ["Next"], 3, 3, True, 8),
        (p5, ["Next"], 3, 3, False, 5),
    ]


def test_title_level_zero_stays_root() -> None:
    blocks = [H("Manual", 0), H("Safety", 1), P("x" * 10), H("Faults", 1), P("y" * 10)]
    paths = [c.heading_path for c in chunk_blocks(blocks, COUNTER, 100, 200)]
    assert paths == [["Manual", "Safety"], ["Manual", "Faults"]]


def test_sentence_fallback_with_overlap_and_max_bound() -> None:
    sentences = [f"Sentence number {i:02d} describes step {i:02d} here." for i in range(30)]
    text = " ".join(sentences)
    pieces = split_long_text(text, COUNTER, target=40, max_tokens=60, overlap_fraction=0.5)
    assert len(pieces) > 1
    for prev, nxt in itertools.pairwise(pieces):
        last = split_sentences(prev)[-1]
        assert nxt.startswith(last)  # one-sentence overlap
    assert all(COUNTER.count(p) <= 60 for p in pieces)
    covered = " ".join(pieces)
    assert all(s in covered for s in sentences)


def test_no_overlap_between_paragraph_chunks() -> None:
    blocks = [P("w " * 30), P("z " * 30)]
    chunks = chunk_blocks(blocks, COUNTER, target_tokens=20, max_tokens=40)
    assert len(chunks) == 2
    assert "z" not in chunks[0].text and "w" not in chunks[1].text


def test_large_table_split_by_rows_repeats_header() -> None:
    header = "| code | meaning |\n|---|---|"
    rows = [f"| E{i:03d} | fault description number {i} |" for i in range(60)]
    md = header + "\n" + "\n".join(rows)
    blocks = [ParsedBlock("table", "grid", table_markdown=md)]
    chunks = chunk_blocks(blocks, COUNTER, target_tokens=100, max_tokens=150)
    assert len(chunks) > 1
    assert all(c.is_table and c.text.startswith(header) for c in chunks)
    assert all(c.n_tokens <= 150 for c in chunks)
    body_rows = [ln for c in chunks for ln in c.text.split("\n")[2:]]
    assert body_rows == rows  # every row exactly once, in order


def test_unbreakable_text_is_bisected_under_max() -> None:
    blocks = [P("x" * 5000)]
    chunks = chunk_blocks(blocks, COUNTER, target_tokens=100, max_tokens=120)
    assert all(c.n_tokens <= 120 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 5000


def test_fake_model_uses_heuristic_counter() -> None:
    assert get_token_counter("fake-student").name.startswith("heuristic")
    assert get_token_counter("org/does-not-exist-locally").name.startswith("heuristic")


# ------------------------------------------------------------------ property
_word = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABC", min_size=1, max_size=12)
_sentence = st.lists(_word, min_size=1, max_size=25).map(lambda ws: " ".join(ws) + ".")
_paragraph = st.lists(_sentence, min_size=1, max_size=12).map(" ".join)
_block = st.one_of(
    _paragraph.map(lambda t: ("paragraph", t)),
    st.tuples(st.integers(1, 3), _word).map(lambda lt: ("heading", lt)),
)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(w == h for h in it) for w in needle)


@settings(max_examples=150, deadline=None)
@given(
    specs=st.lists(_block, min_size=1, max_size=15),
    target=st.integers(8, 120),
    extra=st.integers(0, 120),
    overlap=st.sampled_from([0.0, 0.1, 0.25]),
)
def test_property_chunks_cover_all_paragraph_text(
    specs: list[tuple[str, object]], target: int, extra: int, overlap: float
) -> None:
    blocks: list[ParsedBlock] = []
    for kind, payload in specs:
        if kind == "heading":
            level, text = payload  # type: ignore[misc]
            blocks.append(H(text, level))
        else:
            blocks.append(P(payload))  # type: ignore[arg-type]
    max_tokens = target + extra
    chunks = chunk_blocks(blocks, COUNTER, target, max_tokens, overlap)
    para_words = [w for b in blocks if b.type == "paragraph" for w in re.findall(r"\S+", b.text)]
    chunk_words = [w for c in chunks for w in re.findall(r"\S+", c.text)]
    # bisected words may be split mid-word: compare on characters in that case
    if not _is_subsequence(para_words, chunk_words):
        assert _is_subsequence(list("".join(para_words)), list("".join(chunk_words)))
    assert all(c.n_tokens <= max_tokens for c in chunks)
    assert all(c.n_tokens == COUNTER.count(c.text) for c in chunks)
    assert all(c.text.strip() for c in chunks)
