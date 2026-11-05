from __future__ import annotations

import socket
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths, stage_output
from zllf.core.offline import EgressBlockedError, block_egress
from zllf.core.schemas import Chunk, messages, read_parquet, to_parquet
from zllf.llm.base import SamplingParams
from zllf.llm.fake import FakeEmbedder, FakeLLM, FakeNLI


@pytest.mark.parametrize("hardware", ["fake", "toy", "mac"])
@pytest.mark.parametrize("profile", ["mode_a", "mode_b", "mode_c"])
def test_all_config_layers_resolve(profile: str, hardware: str) -> None:
    cfg = load_config(profile, hardware)
    assert cfg.mode == profile[-1].upper()
    assert cfg.hardware_name == hardware
    assert abs(cfg.splits.train + cfg.splits.val + cfg.splits.test - 1.0) < 1e-9


def test_overrides_win() -> None:
    cfg = load_config("mode_a", "fake", {"seed": 7, "lora": {"rank": 64}})
    assert cfg.seed == 7
    assert cfg.lora.rank == 64
    assert cfg.lora.alpha == 32.0  # untouched default


def test_stage_output_commits_atomically(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    with stage_output(paths, "ingest") as out:
        (out / "a.txt").write_text("v1")
    assert (paths.stage("ingest") / "a.txt").read_text() == "v1"

    with pytest.raises(RuntimeError), stage_output(paths, "ingest") as out:
        (out / "a.txt").write_text("v2")
        raise RuntimeError("crash mid-stage")
    # the committed output survives a crash
    assert (paths.stage("ingest") / "a.txt").read_text() == "v1"


def test_parquet_roundtrip(tmp_path: Path) -> None:
    rows = [Chunk(chunk_id="d-00000", doc_id="d", index=0, text="hello", n_tokens=1,
                  heading_path=["A", "B"])]
    to_parquet(rows, tmp_path / "c.parquet")
    assert read_parquet(tmp_path / "c.parquet", Chunk) == rows


def test_block_egress_blocks_remote_but_not_loopback() -> None:
    with block_egress():
        s = socket.socket()
        with pytest.raises(EgressBlockedError):
            s.connect(("93.184.216.34", 80))
        s.close()


def test_fakes_are_deterministic_and_structured() -> None:
    llm = FakeLLM()
    schema = {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}}
    chat = messages(("user", "The pump P-12 trips at 80 bar."))
    out = llm.chat([chat], SamplingParams(), json_schema=schema)
    assert out == llm.chat([chat], SamplingParams(), json_schema=schema)
    assert '"n": 1' in out[0]
    emb = FakeEmbedder().encode(["pump trips", "pump trips", "unrelated words"])
    assert emb[0] @ emb[1] > 0.99 > emb[0] @ emb[2]
    assert FakeNLI().entailment(["pump trips at 80 bar"], ["pump trips"]) == [1.0]
