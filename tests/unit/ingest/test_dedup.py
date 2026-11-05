from __future__ import annotations

from zllf.stages.ingest.dedup import find_duplicates, shingles

BASE = (
    "Fault E104 means dry running: the pressure at the inlet fell below 0.3 bar for more than "
    "ten seconds. The pump stops and restarts automatically after 120 seconds for at most "
    "three attempts. Check the inlet strainer for clogging and verify the expansion vessel "
    "pre-charge of 1.5 bar before restarting the pump manually."
)


def test_exact_duplicates_ignore_case_and_whitespace() -> None:
    res = find_duplicates([BASE, "  " + BASE.upper().replace(" ", "\n"), "other text"], 0.85)
    assert res.duplicate_of == {1: 0}
    assert (res.exact, res.near) == (1, 0)


def test_near_duplicate_detected_and_first_kept() -> None:
    near = BASE.replace("120 seconds", "two minutes")
    different = "Battery module M-07 was replaced after a swollen cell was found on inspection."
    res = find_duplicates([different, BASE, near], 0.7, num_perm=128, shingle_size=5)
    assert res.duplicate_of == {2: 1}
    assert res.near == 1


def test_threshold_one_disables_near_dedup() -> None:
    near = BASE.replace("120 seconds", "two minutes")
    assert find_duplicates([BASE, near], 1.0).duplicate_of == {}


def test_deterministic() -> None:
    texts = [BASE, BASE.replace("ten", "eleven"), BASE[:150], BASE[100:]]
    a = find_duplicates(texts, 0.8)
    b = find_duplicates(texts, 0.8)
    assert a.duplicate_of == b.duplicate_of


def test_shingles_short_text() -> None:
    assert shingles("Pump P-101", 5) == {"pump p 101"}
    assert shingles("", 5) == set()
