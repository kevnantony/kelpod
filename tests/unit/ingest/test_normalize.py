from __future__ import annotations

from zllf.stages.ingest.normalize import (
    blocks_from_text,
    heading_level,
    normalize_text,
    strip_repeated_edges,
)


def test_nfkc_and_whitespace() -> None:
    raw = "ﬁlter cartridge\t\t  ﬂow ​ok \r\n\r\n\r\n\r\nnext"
    assert normalize_text(raw) == "filter cartridge flow ok\n\nnext"


def test_dehyphenation_across_line_breaks() -> None:
    assert normalize_text("inspect the mainte-\nnance log") == "inspect the maintenance log"
    # upper-case continuation (e.g. compound with a code) is kept
    assert normalize_text("HX-\nPump") == "HX-\nPump"


def test_soft_hyphen_removed() -> None:
    assert normalize_text("Kühl­mittel") == "Kühlmittel"


def test_repeated_header_footer_removed() -> None:
    pages = [f"ACME Manual - Rev 2\nBody text of page {i}.\nPage {i} of 4" for i in range(1, 5)]
    out = strip_repeated_edges(pages)
    assert all("ACME Manual" not in p and "of 4" not in p for p in out)
    assert [p.strip() for p in out] == [f"Body text of page {i}." for i in range(1, 5)]


def test_header_removal_needs_enough_pages() -> None:
    pages = ["Header\nA", "Header\nB"]
    assert strip_repeated_edges(pages) == pages


def test_heading_heuristics() -> None:
    assert heading_level("3.2 Electrical faults") == 2
    assert heading_level("1 Overview") == 1
    assert heading_level("MAINTENANCE SCHEDULE") == 1
    assert heading_level("1. Close valve V-12.") is None
    assert heading_level("The pump stops and restarts") is None


def test_blocks_from_wrapped_text() -> None:
    text = ("2 Safety\nIsolate the pump electrically before opening\nthe terminal box.\n"
            "Coolant can reach 65 degrees.\n- wear gloves\n- wear goggles")
    blocks = blocks_from_text(text, page=3)
    assert [(b.type, b.level) for b in blocks] == [
        ("heading", 1), ("paragraph", None), ("paragraph", None), ("list", None)]
    assert blocks[1].text == "Isolate the pump electrically before opening the terminal box."
    assert blocks[3].text == "- wear gloves\n- wear goggles"
    assert {b.page for b in blocks} == {3}
