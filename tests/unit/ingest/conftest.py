from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zllf.core.config import load_config
from zllf.core.manifest import RunPaths
from zllf.stages.base import RunContext

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def corpus() -> Path:
    if not CORPUS.is_dir():
        pytest.skip("fixture corpus missing: run scripts/make_fixture_corpus.py")
    return CORPUS


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., RunContext]:
    def _make(corpus_dir: Path, **kwargs: object) -> RunContext:
        run_dir = tmp_path / f"run-{len(list(tmp_path.glob('run-*')))}"
        run_dir.mkdir()
        return RunContext(run_id="test-run", paths=RunPaths(run_dir), corpus_dir=corpus_dir,
                          config=load_config(hardware="fake"), **kwargs)  # type: ignore[arg-type]

    return _make
