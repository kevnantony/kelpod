"""Fixtures for synth tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from synth_helpers import ContextFactory, make_chunks, scripted_responder

from zllf.core.config import RunConfig, load_config
from zllf.core.manifest import RunPaths
from zllf.core.schemas import Chunk, to_parquet
from zllf.llm.fake import FakeLLM
from zllf.stages.base import RunContext


@pytest.fixture
def make_ctx(tmp_path: Path) -> ContextFactory:
    """Build a RunContext with chunks_clean.parquet written and a scripted teacher."""
    counter = {"n": 0}

    def factory(
        config: RunConfig | None = None,
        chunks: list[Chunk] | None = None,
        responder: Callable[..., str] | None = scripted_responder,
        use_default_fake: bool = False,
    ) -> RunContext:
        counter["n"] += 1
        root = tmp_path / f"run{counter['n']}"
        paths = RunPaths(root)
        to_parquet(chunks if chunks is not None else make_chunks(),
                   paths.stage("sanitize") / "chunks_clean.parquet")
        ctx = RunContext(run_id=f"t{counter['n']}", paths=paths, corpus_dir=tmp_path,
                         config=config or load_config("mode_a", "fake"))
        if not use_default_fake:
            ctx.llm = lambda role: FakeLLM(f"fake-{role}", responder=responder)  # type: ignore[method-assign]
        return ctx

    return factory
