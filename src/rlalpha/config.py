from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


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
    return _deep_merge(merged, data)


def load_paths(config: str | Path | None = None) -> PathsConfig:
    raw: dict[str, Any] = {}
    if config is not None:
        data = load_yaml(config)
        raw = data.get("paths", data)
    for env, key in ENV_PATH_OVERRIDES.items():
        if value := os.getenv(env):
            raw[key] = value
    return PathsConfig.model_validate(raw)
