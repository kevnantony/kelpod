"""MLX backend tests with the toy 0.5B model (one load per module, freed afterwards)."""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from pathlib import Path

import jsonschema
import pytest
from zllf_llm_helpers import TOY_MODEL, require_mlx_model

from zllf.core.schemas import Message
from zllf.llm.base import SamplingParams

pytestmark = pytest.mark.mlx

GREEDY = SamplingParams(temperature=0.0, max_tokens=24)


def user(text: str, system: str | None = None) -> list[Message]:
    msgs = [Message(role="system", content=system)] if system else []
    return [*msgs, Message(role="user", content=text)]


@pytest.fixture(scope="module")
def backend() -> Iterator[object]:
    require_mlx_model()
    from zllf.llm.mlx_backend import MLXBackend

    b = MLXBackend(TOY_MODEL, max_seq_len=512)
    yield b
    b.close()
    gc.collect()


@pytest.fixture(scope="module")
def adapter_dir(backend: object, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from zllf.llm.mlx_backend import write_random_lora_adapter

    return write_random_lora_adapter(
        backend,  # type: ignore[arg-type]
        tmp_path_factory.mktemp("adapter"),
        rank=4,
        num_layers=4,
        scale=20.0,
        std=0.05,
    )


def test_greedy_is_deterministic(backend) -> None:  # type: ignore[no-untyped-def]
    chat = user("Name three primary colours.")
    a = backend.chat([chat], GREEDY)
    b = backend.chat([chat], GREEDY)
    assert a == b and a[0].strip()


def test_seeded_sampling_is_reproducible(backend) -> None:  # type: ignore[no-untyped-def]
    chat = user("Write a word.")
    p = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=12, seed=7)
    assert backend.chat([chat], p) == backend.chat([chat], p)


def test_batch_matches_single(backend) -> None:  # type: ignore[no-untyped-def]
    chats = [user("Capital of France? One word."), user("Name one colour of the rainbow.")]
    batched = backend.chat(chats, GREEDY)
    singles = [backend.chat([c], GREEDY)[0] for c in chats]
    assert batched == singles
    assert "Paris" in batched[0]
    assert backend.chat(chats[::-1], GREEDY) == batched[::-1]  # order is preserved


def test_stop_strings(backend) -> None:  # type: ignore[no-untyped-def]
    out = backend.chat(
        [user("Count from 1 to 10 separated by commas.")],
        SamplingParams(max_tokens=40, stop=("5",)),
    )[0]
    assert "5" not in out and "6" not in out


def test_long_prompt_is_truncated(backend) -> None:  # type: ignore[no-untyped-def]
    long_text = "lorem ipsum " * 2000
    out = backend.chat([user(long_text + "\nSay OK.")], SamplingParams(max_tokens=4))
    assert len(out) == 1


def test_json_schema_mode_returns_valid_json(backend) -> None:  # type: ignore[no-untyped-def]
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    out = backend.chat(
        [user("Which city is the capital of Germany? Give city and country.")],
        SamplingParams(max_tokens=64),
        json_schema=schema,
    )[0]
    value = json.loads(out)
    jsonschema.validate(value, schema)


def test_loglikelihood_prefers_correct(backend) -> None:  # type: ignore[no-untyped-def]
    q = user("What is the capital of France? Answer with one word.")
    good, bad = backend.loglikelihood([q, q], ["Paris", "Banana"])
    assert good > bad
    assert good < 0


def test_loglikelihood_batch_equals_single(backend) -> None:  # type: ignore[no-untyped-def]
    prompts = [user("Say hello."), user("What colour is the sky on a clear day?")]
    comps = ["Hello!", "The sky is blue."]
    batched = backend.loglikelihood(prompts, comps)
    singles = [backend.loglikelihood([p], [c])[0] for p, c in zip(prompts, comps, strict=True)]
    assert batched == pytest.approx(singles, abs=0.05)


def test_count_tokens(backend) -> None:  # type: ignore[no-untyped-def]
    assert backend.count_tokens("") == 0
    n = backend.count_tokens("Hello world, this is a test.")
    assert 4 <= n <= 12


def test_adapter_swap_leaves_base_unchanged(backend, adapter_dir) -> None:  # type: ignore[no-untyped-def]
    chat = user("Describe the moon in one sentence.")
    base_before = backend.chat([chat], GREEDY)
    ll_before = backend.loglikelihood([chat], ["The moon is bright."])
    with_adapter = backend.chat([chat], GREEDY, adapter_path=adapter_dir)
    ll_adapter = backend.loglikelihood([chat], ["The moon is bright."], adapter_path=adapter_dir)
    base_after = backend.chat([chat], GREEDY)
    ll_after = backend.loglikelihood([chat], ["The moon is bright."])
    assert with_adapter != base_before  # adapter is actually applied
    assert ll_adapter != pytest.approx(ll_before, abs=1e-3)
    assert base_after == base_before  # and fully removed again
    assert ll_after == pytest.approx(ll_before, abs=1e-4)


def test_two_backends_share_weights(backend, adapter_dir) -> None:  # type: ignore[no-untyped-def]
    from zllf.llm.mlx_backend import MLXBackend

    other = MLXBackend(TOY_MODEL, max_seq_len=512)
    try:
        assert other.model is backend.model
        chat = user("Say hi.")
        base = backend.chat([chat], GREEDY)
        other.chat([chat], GREEDY, adapter_path=adapter_dir)
        assert backend.chat([chat], GREEDY) == base  # adapter state never leaks
    finally:
        other.close()
    assert backend.model is not None


def test_missing_adapter_raises(backend, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from zllf.core.errors import ZLLFError

    with pytest.raises(ZLLFError):
        backend.chat([user("x")], GREEDY, adapter_path=tmp_path / "nope")


def test_factory_builds_mlx_backend() -> None:
    require_mlx_model()
    from zllf.core.config import load_config
    from zllf.llm.factory import get_llm

    cfg = load_config("mode_a", "toy")
    with get_llm(cfg, "student") as llm:
        assert llm.count_tokens("hello") >= 1
