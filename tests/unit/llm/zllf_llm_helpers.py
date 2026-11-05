from __future__ import annotations

import pytest

TOY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def cached(model_id: str) -> bool:
    from zllf.core.errors import ZLLFError

    try:
        if model_id.startswith("mlx-community/"):
            from zllf.llm.mlx_backend import resolve_model_path

            resolve_model_path(model_id)
        else:
            from zllf.llm.aux_models import resolve_encoder_path

            resolve_encoder_path(model_id)
    except (ZLLFError, ImportError):
        return False
    return True


def require_mlx_model(model_id: str = TOY_MODEL) -> None:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        pytest.skip("mlx not installed")
    if not cached(model_id):
        pytest.skip(f"{model_id} not in the local HF cache")
