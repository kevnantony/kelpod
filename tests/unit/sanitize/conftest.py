from __future__ import annotations

from pathlib import Path

import pytest

from zllf.stages.sanitize.analyzer import PiiAnalyzer


@pytest.fixture(scope="session")
def analyzer() -> PiiAnalyzer:
    """Regex/heuristic-only analyzer: independent of locally installed spaCy models."""
    return PiiAnalyzer(("en", "de"), ner="regex")


@pytest.fixture(scope="session")
def empty_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("no_models")

