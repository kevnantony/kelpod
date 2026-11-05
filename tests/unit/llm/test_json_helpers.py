"""Pure-Python tests for JSON extraction/repair used by the MLX backend (no MLX needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zllf.core.errors import ErrorCode, ZLLFError
from zllf.core.schemas import Message
from zllf.llm.mlx_backend import (
    apply_stop,
    check_adapter_dir,
    extract_json,
    json_candidates,
    repair_turns,
    resolve_model_path,
    schema_errors,
    with_schema_instruction,
)

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_candidates_from_prose_and_fences() -> None:
    text = 'Sure! Here it is:\n```json\n{"name": "Ada", "age": 36}\n```\nanything else?'
    assert json_candidates(text) == [{"name": "Ada", "age": 36}]


def test_candidates_trailing_comma_and_smart_quotes() -> None:
    assert json_candidates('{“name”: "x", "age": 1,}') == [{"name": "x", "age": 1}]


def test_candidates_truncated_output_is_closed() -> None:
    assert json_candidates('{"name": "Ada", "tags": ["a", "b"') == [
        {"name": "Ada", "tags": ["a", "b"]}
    ]


def test_candidates_none() -> None:
    assert json_candidates("no json here") == []


def test_extract_prefers_valid_candidate() -> None:
    text = '{"name": "bad"} then {"name": "Ada", "age": 36}'
    res = extract_json(text, SCHEMA)
    assert res.valid and json.loads(res.text) == {"name": "Ada", "age": 36}


def test_extract_invalid_keeps_first_parsed_with_errors() -> None:
    res = extract_json('{"name": "Ada", "age": "old"}', SCHEMA)
    assert res.parsed and not res.valid
    assert any("age" in e for e in res.errors)


def test_extract_raw_when_unparseable() -> None:
    res = extract_json("I cannot answer", SCHEMA)
    assert not res.parsed and res.text == "I cannot answer"


def test_extract_array_schema() -> None:
    schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert extract_json('quotes: ["a", "b"]', schema).valid


def test_schema_errors_empty_when_valid() -> None:
    assert schema_errors({"name": "a", "age": 1}, SCHEMA) == []


def test_schema_instruction_merges_into_system() -> None:
    chat = [Message(role="system", content="Be brief."), Message(role="user", content="q")]
    out = with_schema_instruction(chat, SCHEMA)
    assert len(out) == 2 and out[0].content.startswith("Be brief.")
    assert '"required"' in out[0].content
    out2 = with_schema_instruction([Message(role="user", content="q")], SCHEMA)
    assert out2[0].role == "system" and out2[1].content == "q"


def test_repair_turns_shape() -> None:
    chat = [Message(role="user", content="q")]
    out = repair_turns(chat, "{bad", ["(root): x"])
    assert [m.role for m in out] == ["user", "assistant", "user"]
    assert "(root): x" in out[-1].content


def test_apply_stop() -> None:
    assert apply_stop("hello END world STOP", ("STOP", "END")) == "hello "
    assert apply_stop("hello", ()) == "hello"


def test_missing_model_raises_model_missing() -> None:
    with pytest.raises(ZLLFError) as ei:
        resolve_model_path("nobody/definitely-not-a-cached-model-xyz")
    assert ei.value.code == ErrorCode.MODEL_MISSING


def test_bad_adapter_dir(tmp_path: Path) -> None:
    with pytest.raises(ZLLFError) as ei:
        check_adapter_dir(tmp_path)
    assert ei.value.code == ErrorCode.MODEL_MISSING
