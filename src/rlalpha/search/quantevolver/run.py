from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import load_paths, load_yaml
from ...data.discovery import discover_data_files
from ...data.store import PanelStore
from ...dsl.evaluator import EVALUATOR_SEMANTICS_VERSION
from ...dsl.parser import parse_expression
from ...dsl.validity import _daily_rank_corr_exact_serial, validate_signal
from ...manifest import build_manifest
from ...utils.experiment_log import append_event, update_progress, write_result_summary
from ...utils.hashing import stable_hash
from ...utils.io import atomic_write_text, write_json, write_yaml
from ..base_llm import resolve_model_path
from ..grpo.verl_config import build_verl_grpo_config
from ..grpo.verl_trainer import run_quant_evolver_verl_trainer
from .prompts import build_messages, prompt_contract, task_for_round


def _latest_checkpoint(root: Path) -> tuple[int, Path | None]:
    candidates: list[tuple[int, Path]] = []
    for path in (root / "checkpoints/verl").glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if (path / "actor").is_dir():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])


def _truncate_history(path: Path, completed_steps: int, rollout_n: int) -> None:
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    retained = [item for item in records if int(item.get("round", -1)) < int(completed_steps)]
    expected = int(completed_steps) * int(rollout_n)
    if len(retained) != expected:
        raise RuntimeError(
            f"QuantEvolver resume history has {len(retained)} committed rows, expected {expected}"
        )
    atomic_write_text(path, "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in retained))


def _daily_rankic(signal: np.ndarray, label: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    support = mask & np.isfinite(signal) & np.isfinite(label)
    valid_days = support.sum(axis=1) >= 3
    values = _daily_rank_corr_exact_serial(signal, label, mask, valid_days)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    return mean, float(mean / (np.std(finite) + 1e-8) * np.sqrt(len(finite))), int(len(finite))


def _pooled_abs_corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    common = mask & np.isfinite(left) & np.isfinite(right)
    if int(common.sum()) < 3:
        return float("nan")
    x, y = left[common], right[common]
    if np.var(x) <= 1e-24 or np.var(y) <= 1e-24:
        return 1.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def _select_final_pool(
    history_path: Path,
    processed_root: Path,
    capacity: int,
    correlation_threshold: float = 0.70,
) -> dict[str, Any]:
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best: dict[str, dict[str, Any]] = {}
    for item in records:
        if not item.get("valid") or not item.get("expression"):
            continue
        expression = str(item["expression"])
        prior = best.get(expression)
        if prior is None or float(item.get("raw_reward", -1.0)) > float(prior.get("raw_reward", -1.0)):
            best[expression] = item
    mined = [item for item in best.values() if item.get("mined")]
    candidates = mined or list(best.values())
    if not candidates:
        raise RuntimeError("QuantEvolver produced no valid unique candidates")

    validation = PanelStore(processed_root).load_split("validation")
    mask = validation.target(validation.common_mask)
    label = validation.target(validation.label)
    scored: list[tuple[float, float, int, dict[str, Any], np.ndarray]] = []
    for item in candidates:
        try:
            node = parse_expression(str(item["expression"]))
            signal = np.asarray(validation.evaluate(node), dtype=float)
            validity = validate_signal(signal, mask, [])
            if not validity.valid:
                continue
            mean, icir, valid_days = _daily_rankic(signal, label, mask)
            if np.isfinite(mean):
                scored.append((mean, icir, valid_days, item, signal))
        except Exception:
            continue
    scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
    selected: list[tuple[float, float, int, dict[str, Any], np.ndarray]] = []
    for candidate in scored:
        if all(
            _pooled_abs_corr(candidate[4], prior[4], mask) < correlation_threshold
            for prior in selected
        ):
            selected.append(candidate)
        if len(selected) >= int(capacity):
            break
    if not selected:
        raise RuntimeError("no QuantEvolver candidate passed validation selection")

    expressions = [str(item[3]["expression"]) for item in selected]
    factors = []
    for index, (mean, icir, valid_days, source, _signal) in enumerate(selected):
        factor_id = str(source["expr_hash"])
        factors.append({
            "factor_id": factor_id,
            "proposal_id": f"qe_round_{int(source['round']):04d}_rollout_{int(source['rollout_index']):02d}",
            "expression": str(source["expression"]),
            "generator": "quantevolver",
            "parents": [str(source.get("seed_expr") or "")],
            "search_weight": 1.0 / len(selected),
            "factor_lineage_id": f"qe_lineage_{factor_id[:20]}",
            "lineage_status": "verified",
            "validation_rank_ic": mean,
            "validation_rank_ic_ir": icir,
            "validation_valid_days": valid_days,
        })
    pool_hash = stable_hash({"expressions": expressions, "selection": "validation-rankic-decorrelation-0.70"})
    return {
        "snapshot_id": f"snapshot_{pool_hash[:20]}",
        "final_pool_id": f"final_pool_{pool_hash[:20]}",
        "pool_version": len(records) // 8,
        "pool_snapshot_hash": pool_hash,
        "expressions": expressions,
        "factors": factors,
        "train": {
            "objective": float(np.mean([float(item[3].get("mean_rank_ic", np.nan)) for item in selected])),
            "weights": [1.0 / len(selected)] * len(selected),
            "selection_semantics": "QuantEvolver mined-factor database",
        },
        "validation": {
            "objective": float(np.mean([item[0] for item in selected])),
            "mean_rank_ic": float(np.mean([item[0] for item in selected])),
            "weights": [1.0 / len(selected)] * len(selected),
            "selection_semantics": "rank by validation RankIC, greedy abs-correlation<0.70",
        },
        "selected_from": {
            "method": "quantevolver",
            "selection_rule": "validation RankIC ranking, 0.70 decorrelation, equal-weight paper protocol",
            "mined_candidates": len(mined),
            "valid_unique_candidates": len(best),
        },
    }


def run_quantevolver(
    config_path: str | Path,
    reward: str,
    seed: int,
    steps: int,
    experiment_id: str,
    resume: bool = True,
) -> dict[str, Any]:
    if reward != "qe_native":
        raise ValueError(f"QuantEvolver baseline requires reward=qe_native, got {reward}")
    raw_config = load_yaml(config_path)
    experiment = raw_config["experiment"]
    group_size = int(experiment.get("proposal_group_size", 8))
    if group_size != 8:
        raise ValueError(f"QuantEvolver fairness protocol requires 8 completions, got {group_size}")
    paths = load_paths(config_path)
    run_dir = paths.runs_root / experiment_id / "quantevolver" / reward / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "mined_factor_history.jsonl"
    append_event(run_dir / "experiment.log", "search_started", method="quantevolver", reward=reward, seed=seed, search_steps=steps, candidates_per_step=group_size)
    update_progress(run_dir / "progress.json", status="initializing", method="quantevolver", reward=reward, seed=seed, search_steps=steps, candidates_per_step=group_size)

    model_config = load_yaml(paths.code_root / "configs/model/qwen3_5_2b.yaml")
    model_config["model"]["path"] = str(resolve_model_path(model_config))
    search_config = load_yaml(paths.code_root / "configs/search/quantevolver.yaml")["search"]
    reward_config = load_yaml(paths.code_root / "configs/reward/qe_native.yaml")
    evaluation_config = load_yaml(paths.code_root / "configs/eval/preliminary.yaml")["evaluation"]
    effective = {
        "paths": paths.model_dump(mode="json"),
        "experiment": experiment,
        "search": search_config,
        **model_config,
        **reward_config,
        "evaluation": evaluation_config,
        "invocation": {
            "experiment_id": experiment_id,
            "method": "quantevolver",
            "reward": reward,
            "seed": int(seed),
            "search_steps": int(steps),
            "candidates_per_step": group_size,
        },
    }
    write_yaml(run_dir / "effective_config.yaml", effective)
    session = run_dir / "grpo_session"
    session.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "reward_semantics": "quantevolver-dico-rankic-v1",
        "processed_root": str(paths.processed_root.resolve()),
        "history_path": str(history_path.resolve()),
        "signal_cache_root": str((run_dir / "cache/signals").resolve()),
        "rollout_n": group_size,
        "invalid_penalty": -1.0,
        "exact_repeat_penalty": 0.10,
        "family_free_quota": 8,
        "family_low_quality_threshold": 0.08,
        "family_repeat_penalty": 0.02,
        "family_good_new_threshold": 0.10,
        "family_new_bonus": 0.02,
        "elite_top_k": 20,
        "behavior_good_score_threshold": 0.12,
        "behavior_corr_threshold": 0.85,
        "behavior_corr_penalty": 0.08,
        "behavior_low_corr_threshold": 0.50,
        "behavior_low_corr_bonus": 0.02,
        "mined_rank_ic_threshold": 0.01,
        "mined_coverage_threshold": 0.60,
    }
    spec["spec_hash"] = stable_hash(spec)
    spec_path = session / "reward_spec.json"
    write_json(spec_path, spec)

    rows = []
    for round_index in range(int(steps)):
        task = task_for_round(round_index, seed)
        rows.append({
            "data_source": f"quantevolver/{task['family']}/{task['time_split']}",
            "prompt": build_messages(task),
            "reward_model": {"ground_truth": [], "style": "rule"},
            "extra_info": {
                **task,
                "index": round_index,
                "split": "train",
                "spec_path": str(spec_path.resolve()),
                "reward_batch_key": f"{spec['spec_hash']}:{round_index}",
                "expected_samples": group_size,
            },
        })
    train_file = session / "train_tasks.parquet"
    validation_file = session / "validation_tasks.parquet"
    pd.DataFrame(rows).to_parquet(train_file, index=False)
    pd.DataFrame(rows[:1]).to_parquet(validation_file, index=False)

    completed_step, checkpoint = _latest_checkpoint(session)
    if not resume and (completed_step or history_path.exists()):
        raise RuntimeError(f"--no-resume refuses existing QuantEvolver artifacts in {run_dir}")
    _truncate_history(history_path, completed_step, group_size)
    metrics_path = session / "verl_metrics.jsonl"
    config = build_verl_grpo_config(
        paths.quantevolver_root,
        effective,
        train_file,
        validation_file,
        session,
        experiment_name=f"quantevolver_seed_{seed}",
        resume_from_path=checkpoint,
        prompt_groups=1,
        expected_global_step=int(steps),
        total_training_steps=int(steps),
        reward_function_path=Path(__file__).with_name("reward_function.py"),
        agent_loop_config_path=Path(__file__).parents[1] / "grpo/agent_loop_config.yaml",
        metrics_path=metrics_path,
        online_dataset=False,
    )
    from omegaconf import OmegaConf

    atomic_write_text(session / "effective_verl_config.yaml", OmegaConf.to_yaml(config, resolve=True))
    environment = run_dir / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([__import__("sys").executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    atomic_write_text(environment / "pip-freeze.txt", freeze.stdout)
    started = time.monotonic()
    trainer = run_quant_evolver_verl_trainer(config, expected_global_step=int(steps))
    wall_seconds = time.monotonic() - started

    selected = _select_final_pool(history_path, paths.processed_root, int(experiment.get("pool_capacity", 20)))
    write_json(run_dir / "final_pool.json", selected)
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique_market = {str(item["expr_hash"]) for item in records if item.get("market_evaluated") and item.get("expr_hash")}
    train_metrics = {
        "search_steps": int(steps),
        "completed_steps": int(steps),
        "candidates_per_step": group_size,
        "raw_proposals": len(records),
        "valid_unique_evaluations": len(unique_market),
        "invalid": sum(not bool(item.get("valid")) for item in records),
        "duplicates": sum(float(item.get("exact_repeat_penalty", 0.0)) > 0 for item in records),
        "mined_candidates": sum(bool(item.get("mined")) for item in records),
        "pool_size": len(selected["expressions"]),
        "pool_version": int(selected["pool_version"]),
        "wall_seconds": wall_seconds,
        "checkpoint": trainer["checkpoint"],
    }
    write_json(run_dir / "train_metrics.json", train_metrics)
    write_json(run_dir / "validation_metrics.json", selected["validation"])
    write_json(run_dir / "checkpoint.json", {
        "schema_version": 1,
        "reward_semantics": "quantevolver-dico-rankic-v1",
        "paired_optimizer_step": int(steps),
        "trainer_checkpoint": trainer["checkpoint"],
        "pool_history": [
            {
                "admitted": bool(item.get("mined")),
                "candidate_hash": item.get("expr_hash"),
                "round": item.get("round"),
                "raw_reward": item.get("raw_reward"),
            }
            for item in records
        ],
    })
    pd.DataFrame(records).to_parquet(run_dir / "candidates.parquet", index=False)
    manifest = build_manifest(
        paths,
        list(discover_data_files(paths.raw_data_root).values()),
        effective_config=effective,
        model_config=model_config["model"],
        prompt=prompt_contract(),
        reward_version="quantevolver-dico-rankic-v1",
        evaluator_version=EVALUATOR_SEMANTICS_VERSION,
    )
    manifest.update({
        "experiment_id": experiment_id,
        "method": "quantevolver",
        "reward": reward,
        "seed": int(seed),
        "search_steps": int(steps),
        "completed_steps": int(steps),
        "candidates_per_step": group_size,
        "search_accounting": train_metrics,
        "model": model_config["model"],
        "baseline_contract": {
            "source": "QuantEvolver public repository",
            "source_commit": "4eb0e78842138ada5334349585b114ad923564e8",
            "adaptation": "native seeded task bank and DiCo RankIC reward over RLAlpha panel/DSL",
        },
    })
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    write_yaml(run_dir / "manifest.yaml", manifest)
    result = {
        "status": "complete",
        "experiment_id": experiment_id,
        "method": "quantevolver",
        "reward": reward,
        "seed": int(seed),
        "search_steps": int(steps),
        "candidates_per_step": group_size,
        "search": train_metrics,
        "selected_pool_version": selected["pool_version"],
        "train_objective": selected["train"]["objective"],
        "validation_objective": selected["validation"]["objective"],
        "final_factors": selected["expressions"],
    }
    write_json(run_dir / "result.json", result)
    write_result_summary(
        run_dir / "result.md",
        experiment_id=experiment_id,
        method="quantevolver",
        reward=reward,
        seed=seed,
        search_steps=steps,
        ledger=train_metrics,
        pool_version=int(selected["pool_version"]),
        train_objective=selected["train"]["objective"],
        validation_objective=selected["validation"]["objective"],
        expressions=selected["expressions"],
    )
    update_progress(run_dir / "progress.json", status="search_complete", completed_steps=int(steps), search_steps=int(steps), candidates_per_step=group_size, valid_unique_evaluations=len(unique_market), pool_version=int(selected["pool_version"]), pool_size=len(selected["expressions"]))
    append_event(run_dir / "experiment.log", "search_finished", completed_steps=int(steps), valid_unique=len(unique_market), raw_proposals=len(records), pool_size=len(selected["expressions"]))
    return {"run_dir": str(run_dir), **train_metrics}
