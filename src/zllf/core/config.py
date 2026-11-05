"""Typed run configuration: defaults.yaml <- profiles/mode_x.yaml <- hardware/x.yaml <- overrides.

The fully resolved config is stored in each run's manifest (NFR-REL-03).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

Mode = Literal["A", "B", "C"]
Backend = Literal["fake", "mlx", "vllm", "hf"]
TargetModules = Literal["attention", "mlp", "all"]
_ALL_TARGETS: tuple[TargetModules, ...] = ("attention", "mlp", "all")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsConfig(_Strict):
    backend: Backend = "fake"
    student: str = "fake-student"
    teacher: str = "fake-teacher"
    judge: str | None = None  # LLM judge for eval; defaults to teacher
    nli: str = "fake-nli"
    embedder: str = "fake-embedder"
    max_seq_len: int = 2048


class ChunkingConfig(_Strict):
    target_tokens: int = 512
    max_tokens: int = 1024
    overlap_fraction: float = 0.10
    dedup_jaccard: float = 0.85
    minhash_num_perm: int = 128
    shingle_size: int = 5


class SanitizeConfig(_Strict):
    languages: list[str] = Field(default_factory=lambda: ["en", "de"])
    # entity type -> keep | mask | drop_chunk
    policy: dict[str, Literal["keep", "mask", "drop_chunk"]] = Field(default_factory=dict)
    default_action: Literal["keep", "mask", "drop_chunk"] = "mask"
    score_threshold: float = 0.5
    custom_patterns: dict[str, str] = Field(default_factory=dict)


class SynthesisConfig(_Strict):
    tasks: dict[str, float] = Field(default_factory=lambda: {"qa": 0.85, "unanswerable": 0.15})
    teacher_temperature: float = 0.1
    quotes_per_chunk: tuple[int, int] = (1, 5)
    entailment_threshold: float = 0.92
    question_dedup_cosine: float = 0.95
    max_chunks: int | None = None
    n_canaries: int = 10
    min_kept_pairs: int = 20


class SplitConfig(_Strict):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    by: Literal["document"] = "document"


class LoraDefaults(_Strict):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: Literal["attention", "mlp", "all"] = "all"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    epochs: float = 2.0
    batch_size: int = 4
    grad_accumulation: int = 4
    replay_ratio: float = 0.15
    num_layers: int = -1  # MLX: number of layers to adapt from the top (-1 = all)


class SearchSpace(_Strict):
    rank: list[int] = Field(default_factory=lambda: [4, 8, 16, 32, 64, 128])
    alpha_over_rank: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    learning_rate: tuple[float, float] = (1e-5, 5e-4)
    target_modules: list[TargetModules] = Field(default_factory=lambda: list(_ALL_TARGETS))
    dropout: tuple[float, float] = (0.0, 0.2)
    weight_decay: tuple[float, float] = (0.0, 0.1)
    replay_ratio: tuple[float, float] = (0.05, 0.30)


class HPOConfig(_Strict):
    enabled: bool = True
    n_trials_max: int = 32
    n_trials_min: int = 4
    min_resource_fraction: float = 0.25
    reduction_factor: int = 3
    trial_data_fraction: float = 0.3
    objective_weights: dict[str, float] = Field(
        default_factory=lambda: {"domain_val": 1.0, "forgetting": 2.0, "hallucination": 1.5}
    )
    epsilon_forget: float = 0.03
    memory_fraction: float = 0.90


class GatesConfig(_Strict):
    domain_rel_improvement_min: float = 0.12
    forgetting_max_drop_points: float = 3.0
    abstention_min: float = 0.90
    fabrication_max: float = 0.05
    canary_extractions_max: int = 0
    raw_pii_hits_max: int = 0
    schema_valid_min: float = 0.98


class EvalConfig(_Strict):
    rag_top_k: int = 5
    bootstrap_resamples: int = 1000
    anchor_items_per_suite: int = 50
    max_test_items: int | None = None


class BudgetConfig(_Strict):
    total_seconds: int = 8 * 3600


class RunConfig(_Strict):
    mode: Mode = "A"
    profile_name: str = "mode_a"
    hardware_name: str = "fake"
    seed: int = 1234
    system_prompt: str = "You are a helpful assistant for this organisation's documents."
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    sanitize: SanitizeConfig = Field(default_factory=SanitizeConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    lora: LoraDefaults = Field(default_factory=LoraDefaults)
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    json_schema: dict[str, Any] | None = None  # Mode C target schema


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_config(
    profile: str = "mode_a",
    hardware: str = "fake",
    overrides: dict[str, Any] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> RunConfig:
    """Resolve a RunConfig from YAML layers. ``hardware`` is fake | toy | mac | gpu24 | ..."""
    data = _load_yaml(config_dir / "defaults.yaml")
    data = deep_merge(data, _load_yaml(config_dir / "profiles" / f"{profile}.yaml"))
    data = deep_merge(data, _load_yaml(config_dir / "hardware" / f"{hardware}.yaml"))
    data = deep_merge(data, overrides or {})
    data["profile_name"] = profile
    data["hardware_name"] = hardware
    return RunConfig.model_validate(data)
