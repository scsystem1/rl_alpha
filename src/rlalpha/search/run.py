from __future__ import annotations

import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_paths, load_yaml
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.pool import PoolManager
from ..manifest import build_manifest
from ..rewards.r0 import R0Objective
from ..rewards.r1 import R1Objective
from ..rewards.r2_lcb import R2LCBObjective
from ..utils.hashing import stable_hash
from ..utils.io import atomic_write_text, write_json, write_yaml
from .coordinator import SearchCoordinator
from .gp import GPSearcher
from .random_search import RandomSearcher


def objective_for(reward: str, panel: SplitPanel):
    label = panel.target(panel.label)
    mask = panel.target(panel.common_mask) & pd.notna(label)
    if reward == "r0":
        return R0Objective(label, mask)
    exposures = panel.target(panel.exposures)
    if reward == "r1":
        return R1Objective(label, mask, exposures)
    if reward == "r2_lcb":
        return R2LCBObjective(label, mask, exposures, hac_lag=20)
    raise ValueError(f"unknown reward {reward}")


def searcher_for(method: str, seed: int, config: dict[str, Any]):
    if method == "random":
        return RandomSearcher(seed, int(config.get("max_depth", 6)))
    if method == "gp":
        return GPSearcher(seed, int(config.get("population_size", 128)), int(config.get("tournament_size", 5)), int(config.get("elitism", 4)))
    if method == "base_llm":
        from .base_llm import BaseLLMSearcher

        return BaseLLMSearcher.from_config(seed, config)
    if method == "grpo_llm":
        from .grpo.staged_controller import StagedGRPOSearcher

        return StagedGRPOSearcher.from_config(seed, config)
    raise ValueError(f"unknown method {method}")


def _score_validation(expressions: list[str], panel: SplitPanel, reward: str) -> dict[str, Any]:
    signals = [panel.evaluate(parse_expression(expression)) for expression in expressions]
    score = objective_for(reward, panel).score_pool(signals)
    return {"objective": score.objective, "mean_ic": score.mean_ic, "standard_error": score.standard_error, "weights": list(score.weights), "daily_ic": list(score.daily_ic)}


def run_search(config_path: str | Path, method: str, reward: str, seed: int, budget: int, experiment_id: str, resume: bool = True) -> dict[str, Any]:
    raw_config = load_yaml(config_path)
    paths = load_paths(config_path)
    method_config = load_yaml(paths.code_root / f"configs/search/{method}.yaml").get("search", {})
    model_config = load_yaml(paths.code_root / "configs/model/qwen3_5_2b.yaml") if method in {"base_llm", "grpo_llm"} else {}
    merged_config = {**method_config, **model_config, "method": method, "reward": reward, "seed": seed, "budget": budget}
    run_dir = paths.runs_root / experiment_id / method / reward / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    environment_dir = run_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "pip-freeze.txt", freeze.stdout)
    gpu_start = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "gpu-start.csv", gpu_start.stdout or gpu_start.stderr)
    merged_config["run_dir"] = str(run_dir)
    write_yaml(run_dir / "resolved_config.yaml", {"paths": paths.model_dump(mode="json"), "run": merged_config, "source": raw_config})
    store = PanelStore(paths.processed_root)
    train = store.load_split("train")
    validation = store.load_split("validation")
    objective = objective_for(reward, train)
    pool = PoolManager(objective, capacity=int(raw_config.get("experiment", {}).get("pool_capacity", 20)))
    searcher = searcher_for(method, seed, merged_config)
    coordinator = SearchCoordinator(searcher, pool, train.evaluate, train.target(train.common_mask), budget, run_dir)
    checkpoint = run_dir / "checkpoint.json"
    if resume and checkpoint.exists():
        coordinator.load_checkpoint()
        coordinator.ledger.limit = budget
    elif resume and seed == 0 and (source_experiment := raw_config.get("experiment", {}).get("continue_seed_zero_from")):
        source_dir = paths.runs_root / source_experiment / method / reward / "seed_0"
        if (source_dir / "checkpoint.json").exists():
            coordinator.run_dir = source_dir
            coordinator.load_checkpoint()
            coordinator.run_dir = run_dir
            coordinator.ledger.limit = budget
    snapshots_path = run_dir / "checkpoints/snapshots.json"
    snapshots = []
    if snapshots_path.exists():
        import json

        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    elif seed == 0 and (source_experiment := raw_config.get("experiment", {}).get("continue_seed_zero_from")):
        source_snapshots = paths.runs_root / source_experiment / method / reward / "seed_0/checkpoints/snapshots.json"
        if source_snapshots.exists():
            import json

            snapshots = json.loads(source_snapshots.read_text(encoding="utf-8"))
    started = time.monotonic()
    last_version = pool.version
    group_size = int(raw_config.get("experiment", {}).get("proposal_group_size", 8))
    while not coordinator.ledger.exhausted:
        coordinator.run_group(group_size)
        if pool.version != last_version:
            expressions = [entry.expression for entry in pool.entries]
            validation_score = _score_validation(expressions, validation, reward)
            snapshot = {"pool_version": pool.version, "expressions": expressions, "train": asdict(pool._score(pool.entries)), "validation": validation_score, "valid_unique_evaluations": coordinator.ledger.valid_unique_evaluations}
            snapshots.append(snapshot)
            write_json(snapshots_path, snapshots)
            last_version = pool.version
    previous_version = pool.version
    if int(getattr(searcher, "admission_group_interval", 1)) == 1:
        coordinator.flush_admission()
    if pool.version != previous_version:
        expressions = [entry.expression for entry in pool.entries]
        snapshots.append({"pool_version": pool.version, "expressions": expressions, "train": asdict(pool._score(pool.entries)), "validation": _score_validation(expressions, validation, reward), "valid_unique_evaluations": coordinator.ledger.valid_unique_evaluations})
        write_json(snapshots_path, snapshots)
    coordinator.save_checkpoint()
    pd.DataFrame(coordinator.records).to_parquet(run_dir / "candidates.parquet", index=False)
    selected = max(snapshots, key=lambda item: (item["validation"]["objective"], -len(item["expressions"]), -item["pool_version"])) if snapshots else {"pool_version": 0, "expressions": [], "train": {}, "validation": {}}
    write_json(run_dir / "final_pool.json", selected)
    write_json(run_dir / "train_metrics.json", {**coordinator.ledger.state_dict(), "pool_size": len(pool.entries), "pool_version": pool.version, "wall_seconds": time.monotonic() - started})
    write_json(run_dir / "validation_metrics.json", selected.get("validation", {}))
    manifest = build_manifest(paths)
    manifest.update({"experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed, "budget": coordinator.ledger.state_dict(), "model": model_config.get("model") if model_config else None, "splits": {name: {"start": str(split.start.date()), "end": str(split.end.date())} for name, split in __import__("rlalpha.data.splits", fromlist=["SPLITS"]).SPLITS.items()}, "conventions": {"label": "20 trading-day next-close total return", "signal": "formed after t close", "execution": "next trading-day close", "pnl_start": "trading day after execution"}})
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    write_yaml(run_dir / "manifest.yaml", manifest)
    gpu_end = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "gpu-end.csv", gpu_end.stdout or gpu_end.stderr)
    return {"run_dir": str(run_dir), "selected_pool_version": selected["pool_version"], **coordinator.ledger.state_dict()}
