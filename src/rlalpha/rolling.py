"""Resolve recent-alpha windows and reuse the existing matrix per test year."""
from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig, load_paths, load_yaml, resolve_data_evaluation
from .utils.hashing import stable_hash
from .utils.io import write_json, write_yaml


PROTOCOL = "recent_alpha_v1"
POLICIES = {
    "lookback_trading_days": 252,
    "label_horizon": "t+2 through t+21, mature within each segment",
    "pool_selection": "actual_terminal_state_after_fixed_budget",
    "calibration": "fit_weights_once_on_calibration_only",
    "portfolio": "dollar_neutral_only",
    "annual_reset": "start_empty; last_day_pnl_then_close_net_positions_and_charge_cost",
    "window_initialization": "independent_model_optimizer_pool_and_search_memory",
}


def window_config(raw: dict[str, Any], year: int, code_root: str | Path) -> dict[str, Any]:
    """The only calendar arithmetic used by search, workers and evaluation."""
    cells = expected_cells(raw)
    if not cells or len(cells) != len(set(cells)):
        raise ValueError("recent alpha requires nonempty, unique method/reward/seed cells")
    for method, reward, _ in cells:
        if method not in {"random", "gp", "base_llm", "grpo_llm"}:
            raise ValueError(f"recent_alpha_v1 does not support method {method}")
        if reward not in {"r1_oof", "r2_paired_oof"}:
            raise ValueError(f"recent_alpha_v1 supports r1_oof and r2_paired_oof, not {reward}")
    child = deepcopy(raw)
    child.pop("rolling", None)
    child["protocol"] = PROTOCOL
    data, evaluation = resolve_data_evaluation(raw, code_root)
    data.update({
        "train": [f"{year-3}-07-01", f"{year-1}-06-30"],
        "validation": [f"{year-1}-07-01", f"{year-1}-12-31"],
        "test": [f"{year}-01-01", f"{year}-12-31"],
    })
    if data["horizon_trading_days"] != 20:
        raise ValueError("recent_alpha_v1 requires the existing 20-day, next-close label")
    if (evaluation["rebalance_days"], evaluation["holding_days"], evaluation["sleeves"]) != (5, 20, 4):
        raise ValueError("recent_alpha_v1 requires five-day rebalancing and four twenty-day sleeves")
    if sorted(evaluation["one_way_cost_bps"]) != [0, 10]:
        raise ValueError("recent_alpha_v1 reports exactly 0 and 10 bps costs")
    halves = [
        [f"{year-3}-07-01", f"{year-3}-12-31"],
        [f"{year-2}-01-01", f"{year-2}-06-30"],
        [f"{year-2}-07-01", f"{year-2}-12-31"],
        [f"{year-1}-01-01", f"{year-1}-06-30"],
    ]
    reward = dict(child.get("reward", {}))
    reward.update({
        "name": "r1_oof", "neutralized": True, "hac_lag": 20,
        "ridge": evaluation["ridge_lambda"], "min_pool_valid_days": 80,
        "min_pool_valid_day_rate": 0.8, "min_pool_observation_rate": 0.8,
        "time_folds": [{"fit": halves[i], "score": halves[i+1]} for i in range(3)],
    })
    child.update(data=data, evaluation=evaluation, reward=reward)
    ProjectConfig.model_validate(child)
    return child


def expected_cells(raw: dict[str, Any]) -> list[tuple[str, str, int]]:
    experiment = raw["experiment"]
    pairs = experiment["cells"] if "cells" in experiment else itertools.product(experiment["methods"], experiment["rewards"])
    return [(str(method), str(reward), int(seed)) for method, reward in pairs for seed in experiment["seeds"]]


def prepare_rolling_windows(config: str | Path, experiment_id: str) -> list[tuple[int, Path, str]]:
    """Freeze each resolved child config; never reuse a different experiment contract."""
    raw = load_yaml(config)
    paths = load_paths(config)
    root = paths.runs_root / experiment_id
    years = raw["rolling"]["test_years"]
    children = {year: window_config(raw, year, paths.code_root) for year in years}
    # Freeze effective environment path overrides, not only the YAML defaults.
    for child in children.values():
        child["paths"] = paths.model_dump(mode="json")
    contract = {"protocol": PROTOCOL, "policies": POLICIES, "windows": children,
                "expected_cells_per_window": expected_cells(raw)}
    digest = stable_hash(contract)
    manifest_path = root / "rolling_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["identity"] != digest:
            raise RuntimeError("rolling experiment configuration changed; use a new experiment_id")
    elif root.exists() and any(root.glob("test_*/**/final_pool.json")):
        raise RuntimeError("existing rolling cells have no frozen protocol identity; use a new experiment_id")
    configs: list[tuple[int, Path, str]] = []
    for year, child in children.items():
        path = root / "window_configs" / f"test_{year}.yaml"
        if path.exists():
            if stable_hash(load_yaml(path)) != stable_hash(child):
                raise RuntimeError(f"resolved window configuration changed: {path}")
        else:
            write_yaml(path, child)
        configs.append((year, path, f"{experiment_id}/test_{year}"))
    if not manifest_path.exists():
        write_json(manifest_path, {"identity": digest, **contract})
    return configs


def run_rolling_matrix(config: str | Path, experiment_id: str, resume: bool = True,
                       poll_seconds: int = 30, methods: list[str] | None = None,
                       rewards: list[str] | None = None) -> dict[str, Any]:
    from .matrix.runner import run_matrix

    raw = load_yaml(config)
    cells = [cell for cell in expected_cells(raw)
             if (not methods or cell[0] in methods) and (not rewards or cell[1] in rewards)]
    if any(method in {"base_llm", "grpo_llm"} for method, _, _ in cells) and not raw["experiment"].get("auto_start_expensive_jobs", False):
        raise RuntimeError("expensive Base-LLM/GRPO cells are disabled by experiment.auto_start_expensive_jobs=false; enable before freezing the experiment")
    windows = {}
    for year, child, child_id in prepare_rolling_windows(config, experiment_id):
        windows[str(year)] = run_matrix(child, child_id, resume, poll_seconds, methods, rewards)
    return {"experiment_id": experiment_id, "protocol": PROTOCOL, "windows": windows}


def evaluate_rolling(experiment_id: str, config: str | Path,
                     methods: list[str] | None = None) -> dict[str, Any]:
    from .evaluation.finalize import finalize_experiment

    raw = load_yaml(config)
    root = load_paths(config).runs_root / experiment_id
    windows, failures, missing = {}, {}, []
    for year, child, child_id in prepare_rolling_windows(config, experiment_id):
        try:
            windows[str(year)] = finalize_experiment(child_id, child, methods=methods)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            failures[str(year)] = str(exc)
        for method, reward, seed in expected_cells(raw):
            directory = root / f"test_{year}" / method / reward / f"seed_{seed}"
            marker = directory / "test" / "finalization.json"
            if not (directory / "test" / "metrics.json").exists() or not marker.exists() or json.loads(marker.read_text()).get("status") != "complete":
                missing.append(f"test_{year}/{method}/{reward}/seed_{seed}")
        if str(year) in windows and any(item.get("status") != "complete" for item in windows[str(year)].values()):
            failures[str(year)] = "one or more evaluation cells failed"
    result = {"experiment_id": experiment_id, "protocol": PROTOCOL, "windows": windows,
              "failures": failures, "missing_cells": missing,
              "scope_methods": sorted(methods or {cell[0] for cell in expected_cells(raw)}),
              "complete": not failures and not missing and (not methods or set(methods) == {cell[0] for cell in expected_cells(raw)})}
    write_json(root / "rolling_evaluation.json", result)
    return result
