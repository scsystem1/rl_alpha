from __future__ import annotations

import subprocess
import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_paths, load_yaml
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.pool import PoolManager
from ..manifest import build_manifest, git_info
from ..data.discovery import discover_data_files
from ..dsl.evaluator import EVALUATOR_SEMANTICS_VERSION
from ..rewards.r0 import R0Objective
from ..rewards.r1 import R1Objective
from ..rewards.r2_lcb import R2LCBObjective
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import atomic_write_text, write_json, write_yaml
from ..utils.experiment_log import append_event, update_progress, write_result_summary
from .coordinator import SearchCoordinator
from .gp import GPSearcher
from .random_search import RandomSearcher
from .prompts import prompt_contract


def _record_round(
    run_dir: Path,
    method: str,
    round_number: int,
    records: list[Any],
    admission: dict[str, Any],
    coordinator: Any,
    pool: PoolManager,
) -> None:
    normalized = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in records]
    selected_hash = admission.get("candidate_hash")
    selected_expression = next((str(item.get("expression")) for item in normalized if item.get("expr_hash") == selected_hash), None)
    score = pool._score(pool.entries)
    fields = {
        "round": round_number,
        "generated": len(normalized),
        "valid": sum(bool(item.get("valid")) for item in normalized),
        "evaluated": sum(bool(item.get("market_evaluated")) for item in normalized),
        "admitted": bool(admission.get("admitted")),
        "selected": selected_expression,
        "delta": admission.get("delta"),
        "pool_version": pool.version,
        "pool_size": len(pool.entries),
        "pool_objective": float(score.objective),
        "budget": f"{coordinator.ledger.valid_unique_evaluations}/{coordinator.ledger.limit}",
    }
    append_event(run_dir / "experiment.log", "round_complete", **fields)
    update_progress(
        run_dir / "progress.json",
        status="running",
        method=method,
        round=round_number,
        last_round=fields,
        valid_unique_evaluations=coordinator.ledger.valid_unique_evaluations,
        valid_unique_budget=coordinator.ledger.limit,
        pool_version=pool.version,
        pool_size=len(pool.entries),
    )


def _snapshot_record(pool: PoolManager, validation_score: dict[str, Any], valid_unique_evaluations: int, searcher: object) -> dict[str, Any]:
    train_score = asdict(pool._score(pool.entries))
    state = searcher.state_dict()
    factors = []
    for index, entry in enumerate(pool.entries):
        factors.append({
            "factor_id": entry.expr_hash,
            "proposal_id": entry.metadata.get("proposal_id"),
            "expression": entry.expression,
            "generator": entry.metadata.get("generator"),
            "parents": entry.metadata.get("parents", []),
            "search_weight": train_score.get("weights", [])[index] if index < len(train_score.get("weights", [])) else None,
        })
    pool_hash = stable_hash({"pool_version": pool.version, "factor_ids": [item["factor_id"] for item in factors]})
    return {
        "snapshot_id": f"snapshot_{stable_hash({'pool_hash': pool_hash, 'valid_unique_evaluations': valid_unique_evaluations})[:20]}",
        "pool_version": pool.version,
        "pool_snapshot_hash": pool_hash,
        "expressions": [entry.expression for entry in pool.entries],
        "factors": factors,
        "train": train_score,
        "validation": validation_score,
        "valid_unique_evaluations": valid_unique_evaluations,
        "stage": state.get("stage"),
        "group": state.get("groups_in_stage"),
        "optimizer_update": state.get("updates"),
        "checkpoint": state.get("checkpoint"),
    }


def _write_lineage(run_dir: Path, coordinator: SearchCoordinator, snapshots: list[dict[str, Any]], selected: dict[str, Any], method: str, reward: str, seed: int, experiment_id: str) -> dict[str, Any]:
    lineage_root = run_dir / "lineage"
    lineage_root.mkdir(parents=True, exist_ok=True)
    proposal_rows = []
    for record in coordinator.records:
        metadata = dict(record.get("metadata") or {})
        proposal_rows.append({
            "experiment_id": experiment_id,
            "cell_id": f"{method}/{reward}/seed_{seed}",
            "method": method,
            "reward": reward,
            "seed": seed,
            "proposal_id": metadata.get("proposal_id"),
            "factor_id": metadata.get("factor_id"),
            "generator": metadata.get("generator"),
            "raw_proposal_index": metadata.get("raw_proposal_index"),
            "group_index": metadata.get("group_index"),
            "pre_group_pool_version": metadata.get("pre_group_pool_version"),
            "pre_group_pool_snapshot_hash": metadata.get("pre_group_pool_snapshot_hash"),
            "raw_text": metadata.get("raw_text"),
            "parsed_expression": record.get("expression"),
            "valid": record.get("valid"),
            "reason_code": record.get("reason"),
            "market_evaluated": record.get("market_evaluated"),
            "delta_objective": record.get("delta_objective"),
            "shaped_reward": record.get("shaped_reward"),
            "parents": json.dumps(metadata.get("parents", []), sort_keys=True),
            "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
        })
    pd.DataFrame(proposal_rows).to_parquet(lineage_root / "proposals.parquet", index=False)
    admission_rows = [{"experiment_id": experiment_id, "cell_id": f"{method}/{reward}/seed_{seed}", **item} for item in coordinator.pool.history]
    pd.DataFrame(admission_rows).to_parquet(lineage_root / "admission_events.parquet", index=False)
    snapshot_rows = [{**{key: item.get(key) for key in ("snapshot_id", "pool_version", "pool_snapshot_hash", "valid_unique_evaluations", "stage", "group", "optimizer_update", "checkpoint")}, "expressions_json": json.dumps(item.get("expressions", [])), "factors_json": json.dumps(item.get("factors", []), sort_keys=True), "train_objective": item.get("train", {}).get("objective"), "validation_objective": item.get("validation", {}).get("objective")} for item in snapshots]
    pd.DataFrame(snapshot_rows).to_parquet(lineage_root / "pool_snapshots.parquet", index=False)
    final_pool_id = f"final_pool_{stable_hash({'experiment_id': experiment_id, 'cell': f'{method}/{reward}/seed_{seed}', 'snapshot_id': selected.get('snapshot_id')})[:20]}"
    admission_by_factor = {item.get("candidate_hash"): item for item in coordinator.pool.history if item.get("admitted")}
    final_factors = []
    weights = selected.get("train", {}).get("weights", [])
    for index, factor in enumerate(selected.get("factors", [])):
        admission = admission_by_factor.get(factor.get("factor_id"), {})
        final_factors.append({
            **factor,
            "factor_lineage_id": f"lineage_{stable_hash({'proposal_id': factor.get('proposal_id'), 'factor_id': factor.get('factor_id'), 'final_pool_id': final_pool_id})[:20]}",
            "admission_event_id": admission.get("admission_event_id"),
            "admitted_pool_version": admission.get("pool_version"),
            "final_weight": weights[index] if index < len(weights) else None,
            "lineage_status": "verified" if factor.get("proposal_id") and admission.get("admission_event_id") else "legacy_unknown",
        })
    final = {
        **selected,
        "final_pool_id": final_pool_id,
        "selected_from": {
            "method": method,
            "stage": selected.get("stage"),
            "group": selected.get("group"),
            "optimizer_update": selected.get("optimizer_update"),
            "checkpoint": selected.get("checkpoint"),
            "selection_rule": "maximum predeclared validation objective; tie smaller pool then earlier pool version",
        },
        "factors": final_factors,
    }
    pd.DataFrame([{**factor, "experiment_id": experiment_id, "cell_id": f"{method}/{reward}/seed_{seed}", "final_pool_id": final_pool_id} for factor in final_factors]).to_parquet(lineage_root / "final_pool_lineage.parquet", index=False)
    write_json(lineage_root / "final_pool_lineage.json", final)
    return final


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


def searcher_for(method: str, seed: int, config: dict[str, Any], alphagen_root: str | Path | None = None):
    if method == "random":
        return RandomSearcher(seed, int(config.get("max_depth", 6)))
    if method == "gp":
        if alphagen_root is None:
            raise ValueError("alphagen_root is required for the AlphaGen GP baseline")
        return GPSearcher(
            seed,
            alphagen_root,
            population_size=int(config.get("population_size", 8)),
            tournament_size=int(config.get("tournament_size", 5)),
            init_depth=tuple(config.get("init_depth", (2, 6))),
            p_crossover=float(config.get("p_crossover", 0.5882352941)),
            p_subtree_mutation=float(config.get("p_subtree_mutation", 0.1960784314)),
            p_hoist_mutation=float(config.get("p_hoist_mutation", 0.0196078431)),
            p_point_mutation=float(config.get("p_point_mutation", 0.1960784314)),
            p_reproduction=float(config.get("p_reproduction", 0.0)),
            p_point_replace=float(config.get("p_point_replace", 0.60)),
        )
    if method == "base_llm":
        from .base_llm import BaseLLMSearcher

        return BaseLLMSearcher.from_config(seed, config)
    if method == "grpo_llm":
        raise RuntimeError(
            "formal GRPO is stage-driven and cannot use the per-group Searcher interface; "
            "call run_search(), which dispatches VerlGRPOStageCoordinator"
        )
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
    reward_config = load_yaml(paths.code_root / f"configs/reward/{reward}.yaml")
    data_config = load_yaml(paths.code_root / "configs/data/sp500.yaml").get("data", {})
    evaluation_config = load_yaml(paths.code_root / "configs/eval/preliminary.yaml").get("evaluation", {})
    merged_config = {**method_config, **model_config, "method": method, "reward": reward, "seed": seed, "budget": budget}
    run_dir = paths.runs_root / experiment_id / method / reward / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    append_event(run_dir / "experiment.log", "search_started", experiment_id=experiment_id, method=method, reward=reward, seed=seed, budget=budget, resume=resume)
    update_progress(run_dir / "progress.json", status="initializing", method=method, reward=reward, seed=seed, budget=budget)
    environment_dir = run_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "pip-freeze.txt", freeze.stdout)
    gpu_start = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "gpu-start.csv", gpu_start.stdout or gpu_start.stderr)
    merged_config["run_dir"] = str(run_dir)
    effective_config = {"paths": paths.model_dump(mode="json"), "data": data_config, "experiment": raw_config.get("experiment", {}), "search": method_config, **model_config, **reward_config, "evaluation": evaluation_config, "invocation": {"experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed, "budget": budget}}
    if method == "grpo_llm":
        from .base_llm import resolve_model_path

        effective_config["model"]["path"] = str(resolve_model_path(effective_config))
    write_yaml(run_dir / "effective_config.yaml", effective_config)
    identity_inputs = {
        "schema_version": 2,
        "effective_config": effective_config,
        "evaluator_version": EVALUATOR_SEMANTICS_VERSION,
        "repositories": {
            "ours": git_info(paths.code_root),
            "alphagen": git_info(paths.alphagen_root),
            "quantevolver": git_info(paths.quantevolver_root),
        },
    }
    for name in ("build_manifest.yaml", "risk_build_manifest.yaml", "index.json"):
        path = paths.processed_root / "panel" / name
        if path.exists():
            identity_inputs[name] = file_fingerprint(path)
    run_identity = stable_hash(identity_inputs)
    identity_path = run_dir / "run_identity.json"
    if identity_path.exists():
        prior_identity = __import__("json").loads(identity_path.read_text(encoding="utf-8"))
        if prior_identity.get("run_identity") != run_identity:
            raise RuntimeError("run identity changed; use a new experiment ID/artifact root")
    else:
        existing_checkpoint = run_dir / "checkpoint.json"
        if existing_checkpoint.exists():
            raise RuntimeError("legacy checkpoint has no run identity and cannot be resumed")
        write_json(identity_path, {"run_identity": run_identity, "inputs": identity_inputs})
    store = PanelStore(paths.processed_root)
    train = store.load_split("train")
    validation = store.load_split("validation")
    objective = objective_for(reward, train)
    pool = PoolManager(objective, capacity=int(raw_config.get("experiment", {}).get("pool_capacity", 20)))
    if method == "grpo_llm":
        from .grpo.stage_coordinator import VerlGRPOStageCoordinator

        coordinator = VerlGRPOStageCoordinator(
            pool,
            train.evaluate,
            train.target(train.common_mask),
            budget,
            run_dir,
            effective_config,
            paths.quantevolver_root,
            paths.processed_root,
            reward,
            seed,
        )
        searcher = coordinator.searcher
    else:
        searcher = searcher_for(method, seed, merged_config, paths.alphagen_root)
        coordinator = SearchCoordinator(searcher, pool, train.evaluate, train.target(train.common_mask), budget, run_dir)
    checkpoint = run_dir / "checkpoint.json"
    if resume and checkpoint.exists():
        coordinator.load_checkpoint()
        coordinator.ledger.limit = budget
    snapshots_path = run_dir / "checkpoints/snapshots.json"
    snapshots = []
    if snapshots_path.exists():
        import json

        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    group_size = int(raw_config.get("experiment", {}).get("proposal_group_size", 8))
    if method in {"random", "gp", "base_llm", "grpo_llm"} and group_size != 8:
        raise ValueError(f"{method} fairness protocol requires proposal_group_size=8, got {group_size}")
    if method == "grpo_llm":
        while not coordinator.ledger.exhausted:
            records_before = len(coordinator.records)
            result = coordinator.run_stage()
            _record_round(run_dir, method, int(result["stage"]) + 1, coordinator.records[records_before:], result["admission"], coordinator, pool)
            expressions = [entry.expression for entry in pool.entries]
            validation_score = _score_validation(expressions, validation, reward)
            snapshot = _snapshot_record(pool, validation_score, coordinator.ledger.valid_unique_evaluations, searcher)
            snapshots.append(snapshot)
            write_json(snapshots_path, snapshots)
            coordinator.record_validation_event(snapshot["snapshot_id"], validation_score["objective"])
    else:
        last_version = pool.version
        while not coordinator.ledger.exhausted:
            history_before = len(pool.history)
            outcomes = coordinator.run_group(group_size)
            admission = pool.history[-1] if len(pool.history) > history_before else {"admitted": False, "reason": "deferred"}
            _record_round(run_dir, method, int(coordinator.group_index), outcomes, admission, coordinator, pool)
            if pool.version != last_version:
                expressions = [entry.expression for entry in pool.entries]
                validation_score = _score_validation(expressions, validation, reward)
                snapshot = _snapshot_record(pool, validation_score, coordinator.ledger.valid_unique_evaluations, searcher)
                snapshots.append(snapshot)
                write_json(snapshots_path, snapshots)
                last_version = pool.version
        previous_version = pool.version
        if int(getattr(searcher, "admission_group_interval", 1)) == 1:
            coordinator.flush_admission()
        if pool.version != previous_version:
            expressions = [entry.expression for entry in pool.entries]
            snapshots.append(_snapshot_record(pool, _score_validation(expressions, validation, reward), coordinator.ledger.valid_unique_evaluations, searcher))
            write_json(snapshots_path, snapshots)
    coordinator.save_checkpoint()
    pd.DataFrame(coordinator.records).to_parquet(run_dir / "candidates.parquet", index=False)
    selected = max(snapshots, key=lambda item: (item["validation"]["objective"], -len(item["expressions"]), -item["pool_version"])) if snapshots else {"pool_version": 0, "expressions": [], "factors": [], "train": {}, "validation": {}}
    selected = _write_lineage(run_dir, coordinator, snapshots, selected, method, reward, seed, experiment_id)
    write_json(run_dir / "final_pool.json", selected)
    train_metrics = {**coordinator.ledger.state_dict(), "pool_size": len(pool.entries), "pool_version": pool.version, "wall_seconds": time.monotonic() - started}
    write_json(run_dir / "train_metrics.json", train_metrics)
    write_json(run_dir / "validation_metrics.json", selected.get("validation", {}))
    prompt = prompt_contract() if method in {"base_llm", "grpo_llm"} else None
    manifest = build_manifest(paths, list(discover_data_files(paths.raw_data_root).values()), effective_config=effective_config, model_config=model_config.get("model") if model_config else None, prompt=prompt, reward_version=f"{reward}:joint-complete-case-v2", evaluator_version=EVALUATOR_SEMANTICS_VERSION)
    manifest.update({"experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed, "budget": coordinator.ledger.state_dict(), "model": model_config.get("model") if model_config else None, "splits": {name: {"start": str(split.start.date()), "end": str(split.end.date())} for name, split in __import__("rlalpha.data.splits", fromlist=["SPLITS"]).SPLITS.items()}, "conventions": {"label": "20 trading-day next-close total return", "signal": "formed after t close", "execution": "next trading-day close", "pnl_start": "trading day after execution"}})
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    write_yaml(run_dir / "manifest.yaml", manifest)
    result = {
        "status": "complete",
        "experiment_id": experiment_id,
        "method": method,
        "reward": reward,
        "seed": seed,
        "budget": budget,
        "search": train_metrics,
        "selected_pool_version": selected.get("pool_version"),
        "train_objective": selected.get("train", {}).get("objective"),
        "validation_objective": selected.get("validation", {}).get("objective"),
        "final_factors": selected.get("expressions", []),
    }
    write_json(run_dir / "result.json", result)
    write_result_summary(
        run_dir / "result.md",
        experiment_id=experiment_id,
        method=method,
        reward=reward,
        seed=seed,
        budget=budget,
        ledger=coordinator.ledger.state_dict(),
        pool_version=int(selected.get("pool_version", 0)),
        train_objective=selected.get("train", {}).get("objective"),
        validation_objective=selected.get("validation", {}).get("objective"),
        expressions=selected.get("expressions", []),
    )
    update_progress(run_dir / "progress.json", status="search_complete", valid_unique_evaluations=coordinator.ledger.valid_unique_evaluations, valid_unique_budget=budget, pool_version=pool.version, pool_size=len(pool.entries))
    append_event(run_dir / "experiment.log", "search_finished", valid_unique=coordinator.ledger.valid_unique_evaluations, raw_proposals=coordinator.ledger.raw_proposals, pool_version=pool.version, pool_size=len(pool.entries))
    gpu_end = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "gpu-end.csv", gpu_end.stdout or gpu_end.stderr)
    return {"run_dir": str(run_dir), "selected_pool_version": selected["pool_version"], **coordinator.ledger.state_dict()}
