from __future__ import annotations

import subprocess
import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_paths, load_yaml, merge_reward_config, resolve_data_evaluation
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.pool import PoolManager
from ..manifest import build_manifest, git_info
from ..data.discovery import discover_data_files
from ..dsl.evaluator import EVALUATOR_SEMANTICS_VERSION
from ..rewards.factory import objective_for, REWARD_POOL_SEMANTICS
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
    search_steps: int,
) -> None:
    normalized = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in records]
    selected_hash = admission.get("candidate_hash")
    selected_expression = next((str(item.get("expression")) for item in normalized if item.get("expr_hash") == selected_hash), None)
    score = pool.score
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
        "steps": f"{round_number}/{search_steps}",
    }
    append_event(run_dir / "experiment.log", "round_complete", **fields)
    update_progress(
        run_dir / "progress.json",
        status="running",
        method=method,
        round=round_number,
        completed_steps=round_number,
        search_steps=search_steps,
        last_round=fields,
        valid_unique_evaluations=coordinator.ledger.valid_unique_evaluations,
        pool_version=pool.version,
        pool_size=len(pool.entries),
    )


def _snapshot_record(pool: PoolManager, validation_score: dict[str, Any], valid_unique_evaluations: int, searcher: object) -> dict[str, Any]:
    prepared = pool.prepared_state()
    train_score = asdict(pool.score)
    support_diagnostics = getattr(pool.objective, "support_diagnostics", None)
    if prepared is not None and callable(support_diagnostics):
        train_score["support"] = support_diagnostics(prepared)
    diagnostics = getattr(pool.objective, "snapshot_diagnostics", None)
    if callable(diagnostics):
        train_score.update(diagnostics(prepared))
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
    snapshot_rows = [{**{key: item.get(key) for key in ("snapshot_id", "pool_version", "pool_snapshot_hash", "valid_unique_evaluations", "stage", "group", "optimizer_update", "checkpoint")}, "expressions_json": json.dumps(item.get("expressions", [])), "factors_json": json.dumps(item.get("factors", []), sort_keys=True), "train_objective": item.get("train", {}).get("objective"), "validation_objective": item.get("validation", {}).get("objective"), "train_valid_days": item.get("train", {}).get("support", {}).get("valid_days"), "train_observation_rate": item.get("train", {}).get("support", {}).get("observation_rate"), "validation_valid_days": item.get("validation", {}).get("support", {}).get("valid_days"), "validation_observation_rate": item.get("validation", {}).get("support", {}).get("observation_rate")} for item in snapshots]
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
            "selection_rule": selected.get("selection_rule", "support-qualified maximum validation objective with train-fitted ridge weights; tie smaller pool then earlier pool version"),
        },
        "factors": final_factors,
    }
    pd.DataFrame([{**factor, "experiment_id": experiment_id, "cell_id": f"{method}/{reward}/seed_{seed}", "final_pool_id": final_pool_id} for factor in final_factors]).to_parquet(lineage_root / "final_pool_lineage.parquet", index=False)
    write_json(lineage_root / "final_pool_lineage.json", final)
    return final


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
            "formal GRPO is cell-persistent and cannot use the per-group Searcher interface; "
            "call run_search(), which dispatches VerlGRPOStageCoordinator"
        )
    raise ValueError(f"unknown method {method}")


def _score_validation(
    expressions: list[str],
    panel: SplitPanel,
    reward: str,
    train_weights: list[float] | tuple[float, ...],
    signal_cache: dict[str, Any] | None = None,
    reward_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signal_cache = {} if signal_cache is None else signal_cache
    signals = []
    for expression in expressions:
        if expression not in signal_cache:
            signal_cache[expression] = panel.evaluate(parse_expression(expression))
        signals.append(signal_cache[expression])
    objective = objective_for(reward, panel, reward_config, evaluation=True)
    state = objective.prepare_pool(signals)
    score = objective.score_prepared_with_weights(state, train_weights)
    return {
        "objective": score.objective,
        "mean_ic": score.mean_ic,
        "standard_error": score.standard_error,
        "weights": list(score.weights),
        "daily_ic": list(score.daily_ic),
        "support": objective.support_diagnostics(state),
        "ridge_weight_source": "train",
    }


def _select_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        item for item in snapshots
        if bool(item.get("train", {}).get("support", {}).get("valid"))
        and bool(item.get("validation", {}).get("support", {}).get("valid"))
        and np.isfinite(float(item.get("validation", {}).get("objective", float("nan"))))
    ]
    if snapshots and not eligible:
        raise RuntimeError("no pool snapshot satisfies train/validation support requirements")
    return max(
        eligible,
        key=lambda item: (
            item["validation"]["objective"],
            -len(item["expressions"]),
            -item["pool_version"],
        ),
    ) if eligible else {"pool_version": 0, "expressions": [], "factors": [], "train": {}, "validation": {}}


def _terminal_snapshot(pool, coordinator, searcher, ridge_fit_period: str = "calibration"):
    snapshot = _snapshot_record(pool, {}, coordinator.ledger.valid_unique_evaluations, searcher)
    snapshot["selection_rule"] = "fixed_budget_terminal_pool"
    snapshot["calibration_policy"] = (
        "full-train ridge fit after search; no distinct validation/calibration period"
        if ridge_fit_period == "train"
        else "calibration-only ridge fit after search; no search-period calibration access"
    )
    snapshot["status"] = "complete" if pool.entries else "empty_pool"
    return snapshot


def _record_gpu_environment(path: Path) -> None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        output = result.stdout or result.stderr
    except FileNotFoundError:
        output = "nvidia-smi unavailable\n"
    atomic_write_text(path, output)


def run_search(config_path: str | Path, method: str, reward: str, seed: int, steps: int, experiment_id: str, resume: bool = True) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError(f"search steps must be positive, got {steps}")
    raw_config = load_yaml(config_path)
    if raw_config.get("rolling"):
        raise ValueError("rolling search requires a resolved window configuration; use matrix run or a frozen window_configs/test_YEAR.yaml")
    if raw_config.get("protocol") == "recent_alpha_v1":
        from ..rolling import RECENT_METHOD_REWARDS

        if method not in RECENT_METHOD_REWARDS or reward not in RECENT_METHOD_REWARDS[method]:
            raise ValueError(f"recent_alpha_v1 does not support cell {(method, reward)}")
    if method == "quantevolver":
        from .quantevolver.run import run_quantevolver

        return run_quantevolver(config_path, reward, seed, steps, experiment_id, resume)
    if method == "alphasage":
        from .alphasage.run import run_alphasage

        return run_alphasage(config_path, reward, seed, steps, experiment_id, resume)
    recent = raw_config.get("protocol") == "recent_alpha_v1"
    group_size = int(raw_config.get("experiment", {}).get("proposal_group_size", 8))
    if method in {"random", "gp", "base_llm", "grpo_llm"} and group_size != 8:
        raise ValueError(f"{method} fairness protocol requires proposal_group_size=8, got {group_size}")
    candidate_limit = int(steps) * group_size
    paths = load_paths(config_path)
    method_config = load_yaml(paths.code_root / f"configs/search/{method}.yaml").get("search", {})
    model_config = load_yaml(paths.code_root / "configs/model/qwen3_5_2b.yaml") if method in {"base_llm", "grpo_llm"} else {}
    data_config, evaluation_config = resolve_data_evaluation(raw_config, paths.code_root)
    reward_config = {"reward": merge_reward_config(raw_config, paths.code_root, reward)}
    merged_config = {**method_config, **model_config, "method": method, "reward": reward, "seed": seed, "search_steps": steps}
    run_dir = paths.runs_root / experiment_id / method / reward / f"seed_{seed}"
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"--no-resume refuses non-empty cell directory {run_dir}; use a new experiment ID or an empty directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    append_event(run_dir / "experiment.log", "search_started", experiment_id=experiment_id, method=method, reward=reward, seed=seed, search_steps=steps, candidates_per_step=group_size, resume=resume)
    update_progress(run_dir / "progress.json", status="initializing", method=method, reward=reward, seed=seed, search_steps=steps, candidates_per_step=group_size)
    environment_dir = run_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    atomic_write_text(environment_dir / "pip-freeze.txt", freeze.stdout)
    _record_gpu_environment(environment_dir / "gpu-start.csv")
    merged_config["run_dir"] = str(run_dir)
    effective_config = {"paths": paths.model_dump(mode="json"), "data": data_config, "experiment": raw_config.get("experiment", {}), "search": method_config, **model_config, **reward_config, "evaluation": evaluation_config, "invocation": {"experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed, "search_steps": steps, "candidates_per_step": group_size}}
    if recent:
        effective_config["protocol"] = raw_config["protocol"]
    if method == "grpo_llm":
        from .base_llm import resolve_model_path

        effective_config["model"]["path"] = str(resolve_model_path(effective_config))
    identity_inputs = {
        "schema_version": 3,
        "final_pool_selection_version": (
            f"terminal-pool-{evaluation_config.get('ridge_fit_period', 'calibration')}-fit-v2"
            if recent else "full-train-ridge-oof-mean-validation-v2"
        ),
        "prompt_contract": prompt_contract() if method in {"base_llm", "grpo_llm"} else None,
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
    write_yaml(run_dir / "effective_config.yaml", effective_config)
    store = PanelStore(paths.processed_root)
    train = store.load_interval("train", *data_config["train"]) if recent else store.load_split("train")
    validation = None if recent else store.load_split("validation")
    reward_options = dict(reward_config.get("reward", {}))
    objective = objective_for(reward, train, reward_options)
    pool = PoolManager(
        objective,
        capacity=int(raw_config.get("experiment", {}).get("pool_capacity", 20)),
        replacement_top_k=int(method_config.get("replacement_top_k", 3)),
        admission_recheck_top_k=int(method_config.get("admission_recheck_top_k", 3)),
    )
    if method == "grpo_llm":
        from .grpo.stage_coordinator import VerlGRPOStageCoordinator

        coordinator = VerlGRPOStageCoordinator(
            pool,
            train.evaluate,
            train.target(train.common_mask),
            candidate_limit,
            run_dir,
            effective_config,
            paths.quantevolver_root,
            paths.processed_root,
            reward,
            seed,
            train_start=str(data_config["train"][0]) if recent else None,
            train_end=str(data_config["train"][1]) if recent else None,
            max_training_steps=steps,
        )
        searcher = coordinator.searcher
    else:
        searcher = searcher_for(method, seed, merged_config, paths.alphagen_root)
        coordinator = SearchCoordinator(searcher, pool, train.evaluate, train.target(train.common_mask), candidate_limit, run_dir)
    checkpoint = run_dir / "checkpoint.json"
    if resume and checkpoint.exists():
        coordinator.load_checkpoint()
        coordinator.ledger.limit = candidate_limit
    snapshots_path = run_dir / "checkpoints/snapshots.jsonl"
    snapshots: list[dict[str, Any]] = []
    if snapshots_path.exists():
        snapshots = [json.loads(line) for line in snapshots_path.read_text(encoding="utf-8").splitlines()]
    validation_signal_cache: dict[str, Any] = {}

    def score_snapshot(expressions, weights):
        # Calibration is a later fitting stage, never search feedback.
        if recent:
            return {}
        return _score_validation(expressions, validation, reward, weights, validation_signal_cache, reward_options)

    def append_snapshot(snapshot: dict[str, Any]) -> None:
        snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshots_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
            handle.flush()
    previous_wall_seconds = 0.0
    if recent and (run_dir / "train_metrics.json").exists():
        previous_wall_seconds = float(json.loads((run_dir / "train_metrics.json").read_text())["wall_seconds"])
    started = time.monotonic()
    if method == "grpo_llm":
        cell_result = ({"pool_snapshots": [], "checkpoint": str(coordinator.checkpoint)}
            if recent and coordinator.updates == steps else coordinator.run_cell())
        for snapshot in cell_result["pool_snapshots"]:
            validation_score = score_snapshot(
                list(snapshot["expressions"]),
                list(snapshot["train"].get("weights", [])),
            )
            completed = {**snapshot, "validation": validation_score}
            snapshots.append(completed)
            append_snapshot(completed)
        append_event(
            run_dir / "experiment.log",
            "persistent_grpo_complete",
            optimizer_updates=coordinator.updates,
            ray_initializations=1,
            checkpoint=cell_result["checkpoint"],
        )
    else:
        last_version = pool.version
        while coordinator.group_index < steps:
            history_before = len(pool.history)
            outcomes = coordinator.run_group(group_size)
            admission = pool.history[-1] if len(pool.history) > history_before else {"admitted": False, "reason": "deferred"}
            _record_round(run_dir, method, int(coordinator.group_index), outcomes, admission, coordinator, pool, steps)
            if pool.version != last_version:
                expressions = [entry.expression for entry in pool.entries]
                validation_score = score_snapshot(
                    expressions,
                    list(pool.score.weights),
                )
                snapshot = _snapshot_record(pool, validation_score, coordinator.ledger.valid_unique_evaluations, searcher)
                snapshots.append(snapshot)
                append_snapshot(snapshot)
                last_version = pool.version
        previous_version = pool.version
        if int(getattr(searcher, "admission_group_interval", 1)) == 1:
            coordinator.flush_admission()
        if pool.version != previous_version:
            expressions = [entry.expression for entry in pool.entries]
            snapshot = _snapshot_record(
                pool,
                score_snapshot(
                    expressions,
                    list(pool.score.weights),
                ),
                coordinator.ledger.valid_unique_evaluations,
                searcher,
            )
            snapshots.append(snapshot)
            append_snapshot(snapshot)
    coordinator.save_checkpoint()
    if recent:
        selected = _terminal_snapshot(
            pool, coordinator, searcher,
            str(evaluation_config.get("ridge_fit_period", "calibration")),
        )
        snapshots.append(selected)
        append_snapshot(selected)
    else:
        selected = _select_snapshot(snapshots)
    pd.DataFrame(coordinator.records).to_parquet(run_dir / "candidates.parquet", index=False)
    snapshot_frame = pd.DataFrame(snapshots)
    if recent:
        # There is no calibration score during search; Arrow also cannot
        # serialize a column consisting entirely of empty structs.
        snapshot_frame = snapshot_frame.drop(columns=["validation"], errors="ignore")
    snapshot_frame.to_parquet(run_dir / "checkpoints/snapshots.parquet", index=False)
    selected = _write_lineage(run_dir, coordinator, snapshots, selected, method, reward, seed, experiment_id)
    write_json(run_dir / "final_pool.json", selected)
    completed_steps = int(coordinator.updates if method == "grpo_llm" else coordinator.group_index)
    train_metrics = {**coordinator.ledger.state_dict(), "search_steps": int(steps), "completed_steps": completed_steps, "candidates_per_step": group_size, "pool_size": len(pool.entries), "pool_version": pool.version, "wall_seconds": previous_wall_seconds + time.monotonic() - started}
    write_json(run_dir / "train_metrics.json", train_metrics)
    if recent and not selected["expressions"]:
        update_progress(run_dir / "progress.json", status="failed_empty_pool", search_steps=steps, pool_size=0)
        write_json(run_dir / "result.json", {"status": "failed_empty_pool", "experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed})
        raise RuntimeError("fixed-budget search finished with an empty terminal pool; no snapshot fallback is allowed")
    if not recent:
        write_json(run_dir / "validation_metrics.json", selected.get("validation", {}))
    prompt = prompt_contract() if method in {"base_llm", "grpo_llm"} else None
    manifest = build_manifest(paths, list(discover_data_files(paths.raw_data_root).values()), effective_config=effective_config, model_config=model_config.get("model") if model_config else None, prompt=prompt, reward_version=f"{reward}:{REWARD_POOL_SEMANTICS}", evaluator_version=EVALUATOR_SEMANTICS_VERSION)
    manifest.update({"experiment_id": experiment_id, "method": method, "reward": reward, "seed": seed, "search_steps": int(steps), "completed_steps": completed_steps, "candidates_per_step": group_size, "search_accounting": coordinator.ledger.state_dict(), "model": model_config.get("model") if model_config else None, "splits": {name: {"start": str(split.start.date()), "end": str(split.end.date())} for name, split in __import__("rlalpha.data.splits", fromlist=["SPLITS"]).SPLITS.items()}, "conventions": {"label": "20 trading-day next-close total return", "signal": "formed after t close", "execution": "next trading-day close", "pnl_start": "trading day after execution"}})
    if recent:
        manifest["splits"] = {name: {"start": str(data_config[name][0]), "end": str(data_config[name][1])} for name in ("train", "validation", "test")}
        manifest["protocol"] = "recent_alpha_v1"
        manifest["conventions"].update({
            "pool_selection": "terminal",
            "ridge_weight_source": f"{evaluation_config.get('ridge_fit_period', 'calibration')}_only",
            "annual_liquidation": True,
        })
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    write_yaml(run_dir / "manifest.yaml", manifest)
    result = {
        "status": "complete",
        "experiment_id": experiment_id,
        "method": method,
        "reward": reward,
        "seed": seed,
        "search_steps": int(steps),
        "candidates_per_step": group_size,
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
        search_steps=steps,
        ledger=train_metrics,
        pool_version=int(selected.get("pool_version", 0)),
        train_objective=selected.get("train", {}).get("objective"),
        validation_objective=selected.get("validation", {}).get("objective"),
        expressions=selected.get("expressions", []),
    )
    update_progress(run_dir / "progress.json", status="search_complete", completed_steps=completed_steps, search_steps=steps, candidates_per_step=group_size, valid_unique_evaluations=coordinator.ledger.valid_unique_evaluations, pool_version=pool.version, pool_size=len(pool.entries))
    append_event(run_dir / "experiment.log", "search_finished", completed_steps=completed_steps, valid_unique=coordinator.ledger.valid_unique_evaluations, raw_proposals=coordinator.ledger.raw_proposals, pool_version=pool.version, pool_size=len(pool.entries))
    _record_gpu_environment(environment_dir / "gpu-end.csv")
    return {"run_dir": str(run_dir), "selected_pool_version": selected["pool_version"], **train_metrics}
