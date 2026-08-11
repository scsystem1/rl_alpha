from __future__ import annotations

import math
import os
from pathlib import Path
from datetime import date
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_root: Path = Path("/home/sunyuxiang/rl_alpha/ours")
    raw_data_root: Path = Path("/data/sunyuxiang/rl_alpha")
    processed_root: Path = Path("/data/sunyuxiang/rl_alpha/processed")
    cache_root: Path = Path("/data/sunyuxiang/rl_alpha/cache")
    runs_root: Path = Path("/home/sunyuxiang/rl_alpha/ours/output")
    model_search_root: Path = Path("/data/shared/huggingface")
    alphagen_root: Path = Path("/home/sunyuxiang/rl_alpha/alphagen")
    quantevolver_root: Path = Path("/home/sunyuxiang/rl_alpha/QuantEvolver")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    warmup_start: date
    train: tuple[date, date]
    validation: tuple[date, date]
    test: tuple[date, date]
    horizon_trading_days: int = Field(gt=0)
    execution: Literal["next_close"]
    fundamental_lag_months: int = Field(ge=0)
    fundamental_max_age_months: int = Field(gt=0)


SearchMethod = Literal["random", "gp", "base_llm", "grpo_llm"]
RewardName = Literal["r0", "r1", "r2_lcb"]


class ExperimentConfig(StrictModel):
    methods: list[SearchMethod] | None = None
    rewards: list[RewardName] | None = None
    cells: list[tuple[SearchMethod, RewardName]] | None = None
    seeds: list[int]
    valid_unique_budget: int = Field(gt=0)
    proposal_group_size: int = Field(default=8, gt=0)
    pool_capacity: int = Field(default=20, gt=0)
    max_cpu_jobs: int = Field(default=4, gt=0)
    cpu_threads_per_job: int = Field(default=8, gt=0)
    gpu_devices: dict[SearchMethod, list[int]] = Field(default_factory=dict)
    gpu_min_free_mib: dict[int, int] = Field(default_factory=dict)
    gpu_memory_utilization: dict[int, float] = Field(default_factory=dict)
    auto_start_expensive_jobs: bool = False

    @model_validator(mode="after")
    def validate_matrix(self) -> "ExperimentConfig":
        if self.cells is None and (not self.methods or not self.rewards):
            raise ValueError("experiment needs cells or both methods and rewards")
        if self.cells is not None and (self.methods is not None or self.rewards is not None):
            raise ValueError("experiment cells cannot be combined with methods/rewards")
        if any(not devices or any(device < 0 for device in devices) for devices in self.gpu_devices.values()):
            raise ValueError("experiment.gpu_devices entries must be non-empty lists of nonnegative physical GPU IDs")
        if any(device < 0 or value <= 0 for device, value in self.gpu_min_free_mib.items()):
            raise ValueError("experiment.gpu_min_free_mib must contain positive thresholds for nonnegative GPU IDs")
        if any(device < 0 or not 0 < value <= 1 for device, value in self.gpu_memory_utilization.items()):
            raise ValueError("experiment.gpu_memory_utilization values must be in (0, 1]")
        return self


class ModelFingerprintConfig(StrictModel):
    weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str


class ModelConfig(StrictModel):
    repository: str
    revision: str
    path: Path
    fingerprint: ModelFingerprintConfig
    local_files_only: bool = True
    trust_remote_code: bool = True
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = True


class RolloutConfig(StrictModel):
    name: Literal["vllm"] = "vllm"
    n: int = Field(default=8, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    response_length: int = Field(default=128, gt=0)
    max_model_len: int = Field(default=4096, gt=0)


class ActorConfig(StrictModel):
    use_dynamic_bsz: bool = False
    use_kl_loss: bool = True
    kl_loss_coef: float = Field(default=0.001, ge=0)
    learning_rate: float = Field(default=1e-6, gt=0)
    lora_rank: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_target_modules: str | list[str] = "all-linear"
    clip_ratio: float = Field(default=0.2, gt=0)
    clip_ratio_low: float = Field(default=0.2, gt=0)
    clip_ratio_high: float = Field(default=0.2, gt=0)
    ppo_epochs: int = Field(default=1, gt=0)
    entropy_coeff: float = Field(default=0.0, ge=0)
    kl_loss_type: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "low_var_kl"


class SearchConfig(StrictModel):
    method: SearchMethod
    group_size: int | None = Field(default=None, gt=0)
    budget: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    max_nodes: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, gt=0)
    response_length: int | None = Field(default=None, gt=0)
    population_size: int | None = Field(default=None, gt=0)
    tournament_size: int | None = Field(default=None, gt=0)
    init_depth: tuple[int, int] | None = None
    p_crossover: float | None = Field(default=None, ge=0, le=1)
    p_subtree_mutation: float | None = Field(default=None, ge=0, le=1)
    p_hoist_mutation: float | None = Field(default=None, ge=0, le=1)
    p_point_mutation: float | None = Field(default=None, ge=0, le=1)
    p_reproduction: float | None = Field(default=None, ge=0, le=1)
    p_point_replace: float | None = Field(default=None, ge=0, le=1)
    staged_frozen_pool: bool | None = None

    @model_validator(mode="after")
    def validate_gp(self) -> "SearchConfig":
        if self.method != "gp":
            return self
        if self.population_size is not None and self.population_size != 8:
            raise ValueError("formal GP requires population_size=8")
        if self.population_size is not None and self.tournament_size is not None and self.tournament_size > self.population_size:
            raise ValueError("GP tournament_size cannot exceed population_size")
        if self.init_depth is not None and (
            self.init_depth[0] <= 0 or self.init_depth[1] < self.init_depth[0]
        ):
            raise ValueError("GP init_depth must be an increasing positive pair")
        probabilities = (
            self.p_crossover,
            self.p_subtree_mutation,
            self.p_hoist_mutation,
            self.p_point_mutation,
            self.p_reproduction,
        )
        supplied = [value for value in probabilities if value is not None]
        if supplied and len(supplied) != len(probabilities):
            raise ValueError("GP operation probabilities must be specified together")
        if supplied and not math.isclose(sum(supplied), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("GP operation probabilities must sum to one")
        return self


class RewardConfig(StrictModel):
    name: RewardName
    neutralized: bool
    hac_lag: int | None = Field(default=None, ge=0)
    critical_value: float | None = Field(default=None, gt=0)
    invalid_penalty: float = -1.0


class EvaluationConfig(StrictModel):
    ridge_lambda: float = Field(default=0.001, gt=0)
    hac_lag: int = Field(default=20, ge=0)
    rebalance_days: int = Field(default=5, gt=0)
    holding_days: int = Field(default=20, gt=0)
    sleeves: int = Field(default=4, gt=0)
    one_way_cost_bps: list[float] = [0, 10]
    fully_neutral_max_weight: float = Field(default=0.02, gt=0)
    common_universe_min_coverage: float = Field(default=0.50, gt=0, le=1)
    common_universe_min_assets_per_day: int = Field(default=100, gt=0)
    common_universe_min_valid_days: int = Field(default=252, gt=0)
    net_tolerance: float = Field(default=1e-8, gt=0)
    exposure_tolerance: float = Field(default=1e-6, gt=0)
    gross_tolerance: float = Field(default=1e-6, gt=0)
    weight_tolerance: float = Field(default=1e-6, gt=0)
    bootstrap_block_length: int = Field(default=20, gt=0)
    bootstrap_samples: int = Field(default=2000, gt=0)
    bootstrap_seed: int = 0
    fdr_threshold: float = Field(default=0.05, gt=0, lt=1)


class ProjectConfig(StrictModel):
    paths: PathsConfig | None = None
    data: DataConfig | None = None
    experiment: ExperimentConfig | None = None
    model: ModelConfig | None = None
    rollout: RolloutConfig | None = None
    actor: ActorConfig | None = None
    search: SearchConfig | None = None
    reward: RewardConfig | None = None
    evaluation: EvaluationConfig | None = None


ENV_PATH_OVERRIDES = {
    "RLALPHA_CODE_ROOT": "code_root",
    "RLALPHA_RAW_DATA_ROOT": "raw_data_root",
    "RLALPHA_PROCESSED_ROOT": "processed_root",
    "RLALPHA_CACHE_ROOT": "cache_root",
    "RLALPHA_RUNS_ROOT": "runs_root",
    "RLALPHA_MODEL_SEARCH_ROOT": "model_search_root",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    defaults = data.pop("defaults", [])
    merged: dict[str, Any] = {}
    for relative in defaults:
        default_path = (path.parent / relative).resolve()
        merged = _deep_merge(merged, load_yaml(default_path))
    resolved = _deep_merge(merged, data)
    ProjectConfig.model_validate(resolved)
    return resolved


def load_project_config(path: str | Path) -> ProjectConfig:
    return ProjectConfig.model_validate(load_yaml(path))


def load_paths(config: str | Path | None = None) -> PathsConfig:
    raw: dict[str, Any] = {}
    if config is not None:
        data = load_yaml(config)
        raw = data.get("paths", data)
    for env, key in ENV_PATH_OVERRIDES.items():
        if value := os.getenv(env):
            raw[key] = value
    return PathsConfig.model_validate(raw)
