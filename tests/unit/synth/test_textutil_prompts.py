from __future__ import annotations

from zllf.stages.synth.judge import verify_quote
from zllf.stages.synth.prompts import PROMPT_DIR, prompts_manifest, render, teacher_chat
from zllf.stages.synth.textutil import (
    abstention,
    contains_normalized,
    guess_language,
    parse_json_object,
    window,
)

CHUNK = "The feed pump P-12 trips when the pressure\n exceeds 80 bar. Inspect the PRV."


def test_parse_json_object_tolerates_fences_and_chatter() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    assert parse_json_object("not json at all") is None
    assert parse_json_object('{"a": 1') is None
    assert parse_json_object("[1, 2]") is None


def test_whitespace_normalised_substring() -> None:
    assert contains_normalized(CHUNK, "pressure exceeds   80 bar.")
    assert not contains_normalized(CHUNK, "pressure exceeds 90 bar.")


def test_verify_quote_exact_repaired_and_missing() -> None:
    exact = verify_quote(CHUNK, "trips when the pressure exceeds 80 bar.")
    assert exact is not None and exact[2] is False
    repaired = verify_quote(CHUNK, "the feed pump p 12 trips")
    assert repaired is not None and repaired[2] is True
    assert repaired[0] == "The feed pump P-12 trips"  # verbatim source span
    assert verify_quote(CHUNK, "the pump explodes at 80 bar") is None


def test_window_snaps_to_whitespace() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    start = text.index("gamma")
    assert window(text, start, start + 5, 8) == "beta gamma delta"


def test_language_and_abstention() -> None:
    assert guess_language("Die Anlage wird nicht mit dem Druck geprüft und ist aus.") == "de"
    assert guess_language("The pump is checked and the valve is open.") == "en"
    assert abstention("de").startswith("Die Dokumente")
    assert abstention("fr") == abstention("en")


def test_prompts_render_with_passage_first_and_are_hashed() -> None:
    chat = teacher_chat("quotes", passage="PASSAGE TEXT", heading="", min_quotes=1, max_quotes=3)
    assert chat[0].role == "system" and "between 1 and 3" in chat[0].content
    assert chat[1].content.startswith("PASSAGE TEXT")
    assert render("sample_drafting.user.j2", notes="n1").startswith("Rewrite")
    manifest = prompts_manifest()
    assert set(manifest) == {p.name for p in PROMPT_DIR.glob("*.j2")}
    assert all(len(h) == 64 for h in manifest.values())
    for name in ("quotes", "qa", "judge", "summarization", "definition", "drafting",
                 "extraction", "taxonomy", "unanswerable"):
        assert f"{name}.system.j2" in manifest and f"{name}.user.j2" in manifest
