"""Create a non-destructive derived evaluation with an exact selected pool size."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rlalpha.config import load_paths, load_yaml
from rlalpha.evaluation.finalize import finalize_cell
from rlalpha.reporting.build import build_report
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import write_json, write_yaml


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _cells(raw: dict[str, Any]) -> list[tuple[str, str, int]]:
    experiment = raw["experiment"]
    if "cells" in experiment:
        pairs = experiment["cells"]
    else:
        pairs = [
            (method, reward)
            for method in experiment["methods"]
            for reward in experiment["rewards"]
        ]
    return [
        (str(method), str(reward), int(seed))
        for method, reward in pairs
        for seed in experiment["seeds"]
    ]


def _snapshots(path: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("pool_snapshot_hash") or item["snapshot_id"])
            unique.setdefault(key, item)
    return list(unique.values())


def _eligible_snapshots(path: Path, pool_size: int) -> list[dict[str, Any]]:
    eligible = []
    for item in _snapshots(path):
        if len(item.get("expressions", [])) != pool_size:
            continue
        train = item.get("train", {})
        validation = item.get("validation", {})
        objective = float(validation.get("objective", float("nan")))
        if not train.get("support", {}).get("valid"):
            continue
        if not validation.get("support", {}).get("valid") or not math.isfinite(objective):
            continue
        if len(train.get("weights", [])) != pool_size:
            continue
        eligible.append(item)
    return eligible


def _select(items: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        items,
        key=lambda item: (
            float(item["validation"]["objective"]),
            -int(item["pool_version"]),
        ),
    )


def _derived_final_pool(
    selected: dict[str, Any],
    source_cell: Path,
    source_experiment: str,
    destination_experiment: str,
    method: str,
    reward: str,
    seed: int,
    pool_size: int,
) -> dict[str, Any]:
    admissions_path = source_cell / "lineage/admission_events.parquet"
    admissions: dict[str, dict[str, Any]] = {}
    if admissions_path.exists():
        admissions = {
            str(item.get("candidate_hash")): item
            for item in pd.read_parquet(admissions_path).to_dict("records")
            if item.get("admitted")
        }
    final_pool_id = "final_pool_" + stable_hash(
        {
            "destination_experiment": destination_experiment,
            "cell": f"{method}/{reward}/seed_{seed}",
            "snapshot_id": selected["snapshot_id"],
            "required_pool_size": pool_size,
        }
    )[:20]
    weights = list(selected["train"]["weights"])
    factors = []
    for index, source_factor in enumerate(selected.get("factors", [])):
        factor = dict(source_factor)
        admission = admissions.get(str(factor.get("factor_id")), {})
        factor.update(
            {
                "factor_lineage_id": "lineage_"
                + stable_hash(
                    {
                        "proposal_id": factor.get("proposal_id"),
                        "factor_id": factor.get("factor_id"),
                        "final_pool_id": final_pool_id,
                    }
                )[:20],
                "admission_event_id": admission.get("admission_event_id"),
                "admitted_pool_version": admission.get("pool_version"),
                "final_weight": weights[index],
                "lineage_status": (
                    "verified"
                    if factor.get("proposal_id") and admission.get("admission_event_id")
                    else "legacy_unknown"
                ),
            }
        )
        factors.append(factor)
    result = dict(selected)
    result.update(
        {
            "final_pool_id": final_pool_id,
            "factors": factors,
            "selected_from": {
                "method": method,
                "stage": selected.get("stage"),
                "group": selected.get("group"),
                "optimizer_update": selected.get("optimizer_update"),
                "checkpoint": selected.get("checkpoint"),
                "selection_rule": (
                    f"support-qualified maximum validation objective among snapshots "
                    f"with exactly {pool_size} factors; train-fitted ridge weights frozen; "
                    "tie earlier pool version"
                ),
                "source_experiment": source_experiment,
                "source_cell": str(source_cell),
            },
        }
    )
    if len(result["expressions"]) != pool_size or len(result["factors"]) != pool_size:
        raise RuntimeError("derived final pool does not have the required size")
    return result


def _copy_metadata(source: Path, destination: Path, destination_experiment: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "train_metrics.json",
        "validation_metrics.json",
        "effective_config.yaml",
        "run_identity.json",
        "checkpoint.json",
        "candidates.parquet",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
    manifest_path = source / "manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifest["source_experiment_id"] = manifest.get("experiment_id")
        manifest["experiment_id"] = destination_experiment
        manifest["derived_artifact"] = True
        manifest.pop("manifest_hash", None)
        manifest["manifest_hash"] = stable_hash(manifest)
        write_yaml(destination / "manifest.yaml", manifest)
    for name in ("progress.json", "result.json"):
        path = source / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment_id"] = destination_experiment
        payload.pop("evaluation", None)
        if name == "progress.json":
            payload["status"] = "complete"
            payload["evaluation_status"] = "pending"
        write_json(destination / name, payload)


def _evaluate_one(job: tuple[str, str, str, str, int, str, int, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    cell_text, processed_root, source_experiment, method, seed, reward, pool_size, evaluation_config = job
    cell = Path(cell_text)
    key = f"{method}/{reward}/seed_{seed}"
    pool_path = cell / "final_pool.json"
    before = _digest(pool_path)
    scope_hash = stable_hash(
        {
            "cell": key,
            "source_experiment": source_experiment,
            "selection_rule": "exact_pool_size",
            "required_pool_size": pool_size,
            "support": "fixed-universe-zero-fill-psd-gram-v6",
        }
    )
    metrics = finalize_cell(
        cell,
        processed_root,
        finalization_scope_hash=scope_hash,
        evaluation_config=evaluation_config,
    )
    after = _digest(pool_path)
    if before != after:
        raise RuntimeError(f"evaluation changed frozen final pool for {key}")
    if int(metrics["pool_size"]) != pool_size:
        raise RuntimeError(f"evaluation reported the wrong pool size for {key}")
    return key, {
        "status": "complete",
        "pool_size": int(metrics["pool_size"]),
        "primary_pearson_rnic": metrics["primary_pearson_rnic"]["mean"],
        "rank_rnic": metrics["rank_rnic"]["mean"],
        "fully_neutral_10bps_sharpe": metrics["portfolios"]["fully_neutral"]["10bps"]["sharpe"],
        "final_pool_sha256": after,
    }


def _load_completed_result(cell: Path, key: str, pool_size: int) -> dict[str, Any]:
    """Load an already completed cell without reopening the test evaluation."""
    pool_path = cell / "final_pool.json"
    metrics_path = cell / "test/metrics.json"
    marker_path = cell / "test/finalization.json"
    if not pool_path.exists() or not metrics_path.exists() or not marker_path.exists():
        raise RuntimeError(f"report-only artifacts are incomplete for {key}")
    final_pool = json.loads(pool_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError(f"test finalization is not complete for {key}")
    if len(final_pool.get("expressions", [])) != pool_size:
        raise RuntimeError(f"derived final pool does not have size {pool_size} for {key}")
    if int(metrics["pool_size"]) != pool_size:
        raise RuntimeError(f"saved test metrics report the wrong pool size for {key}")
    return {
        "status": "complete",
        "pool_size": int(metrics["pool_size"]),
        "primary_pearson_rnic": metrics["primary_pearson_rnic"]["mean"],
        "rank_rnic": metrics["rank_rnic"]["mean"],
        "fully_neutral_10bps_sharpe": metrics["portfolios"]["fully_neutral"]["10bps"]["sharpe"],
        "final_pool_sha256": _digest(pool_path),
    }


def _finish_report(
    destination_root: Path,
    destination_experiment: str,
    source_experiment: str,
    pool_size: int,
    results: dict[str, Any],
    config: Path,
) -> None:
    write_json(
        destination_root / "evaluation_summary.json",
        {key: {"status": "complete", "metrics": value} for key, value in sorted(results.items())},
    )
    # The report builder requires this experiment-level transaction marker.
    write_json(
        destination_root / "test_finalization.json",
        {
            "status": "complete",
            "completed_cells": len(results),
            "failed_cells": 0,
            "source_experiment": source_experiment,
            "selection_rule": "exact_pool_size",
            "required_pool_size": pool_size,
        },
    )
    build_report(destination_experiment, config)
    write_json(
        destination_root / "evaluation_status.json",
        {
            "status": "complete",
            "source_experiment": source_experiment,
            "destination_experiment": destination_experiment,
            "required_pool_size": pool_size,
            "evaluated_cells": len(results),
            "source_final_pools_unchanged": True,
            "report": str(destination_root / "report.md"),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluated_cells": len(results),
                "report": str(destination_root / "report.md"),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--destination-experiment", required=True)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Build aggregate outputs from already completed per-cell test artifacts.",
    )
    args = parser.parse_args()

    if args.source_experiment == args.destination_experiment:
        raise RuntimeError("source and destination experiment IDs must differ")
    config = args.config.resolve()
    raw = load_yaml(config)
    paths = load_paths(config)
    source_root = paths.runs_root / args.source_experiment
    destination_root = paths.runs_root / args.destination_experiment
    targets = _cells(raw)
    availability: dict[str, int] = {}
    eligible_by_cell: dict[str, list[dict[str, Any]]] = {}
    for method, reward, seed in targets:
        key = f"{method}/{reward}/seed_{seed}"
        path = source_root / key / "checkpoints/snapshots.jsonl"
        eligible = _eligible_snapshots(path, args.pool_size)
        availability[key] = len(eligible)
        eligible_by_cell[key] = eligible
    missing = sorted(key for key, count in availability.items() if count == 0)
    print(json.dumps({"required_pool_size": args.pool_size, "eligible_snapshots": availability}, indent=2, sort_keys=True))
    if missing:
        raise RuntimeError(f"no eligible exact-size snapshot for cells: {missing}")
    if args.check_only:
        return
    if destination_root.exists() and any(destination_root.iterdir()) and not (args.resume or args.report_only):
        raise RuntimeError(f"destination is non-empty: {destination_root}; use --resume or a new ID")
    destination_root.mkdir(parents=True, exist_ok=True)

    source_guard_paths = [
        source_root / method / reward / f"seed_{seed}" / "final_pool.json"
        for method, reward, seed in targets
    ]
    source_hashes_before = {str(path): _digest(path) for path in source_guard_paths}
    if args.report_only:
        results = {}
        for method, reward, seed in targets:
            key = f"{method}/{reward}/seed_{seed}"
            results[key] = _load_completed_result(destination_root / key, key, args.pool_size)
        source_hashes_after = {str(path): _digest(path) for path in source_guard_paths}
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("a source final_pool.json changed while rebuilding the derived report")
        _finish_report(
            destination_root,
            args.destination_experiment,
            args.source_experiment,
            args.pool_size,
            results,
            config,
        )
        return

    selections: dict[str, Any] = {}
    for method, reward, seed in targets:
        key = f"{method}/{reward}/seed_{seed}"
        source_cell = source_root / key
        destination_cell = destination_root / key
        selected = _select(eligible_by_cell[key])
        _copy_metadata(source_cell, destination_cell, args.destination_experiment)
        final_pool = _derived_final_pool(
            selected,
            source_cell,
            args.source_experiment,
            args.destination_experiment,
            method,
            reward,
            seed,
            args.pool_size,
        )
        write_json(destination_cell / "final_pool.json", final_pool)
        write_json(destination_cell / "validation_metrics.json", final_pool["validation"])
        selections[key] = {
            "snapshot_id": selected["snapshot_id"],
            "pool_version": selected["pool_version"],
            "pool_size": len(selected["expressions"]),
            "valid_unique_evaluations": selected.get("valid_unique_evaluations"),
            "validation_objective": selected["validation"]["objective"],
            "eligible_exact_size_snapshots": availability[key],
        }

    # Freeze all derived pools before opening the test split for any cell.
    write_json(
        destination_root / "pool_reselection.json",
        {
            "status": "frozen",
            "source_experiment": args.source_experiment,
            "destination_experiment": args.destination_experiment,
            "required_pool_size": args.pool_size,
            "selection_rule": "maximum validation objective among support-qualified exact-size snapshots",
            "cells": selections,
        },
    )
    evaluation_config = load_yaml(paths.code_root / "configs/eval/preliminary.yaml")["evaluation"]
    jobs = [
        (
            str(destination_root / method / reward / f"seed_{seed}"),
            str(paths.processed_root),
            args.source_experiment,
            method,
            seed,
            reward,
            args.pool_size,
            evaluation_config,
        )
        for method, reward, seed in targets
    ]
    results: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_evaluate_one, job) for job in jobs]
        for future in as_completed(futures):
            key, metrics = future.result()
            results[key] = metrics
            print(json.dumps({"evaluation_complete": key, **metrics}, sort_keys=True), flush=True)
    source_hashes_after = {str(path): _digest(path) for path in source_guard_paths}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("a source final_pool.json changed during derived evaluation")
    _finish_report(
        destination_root,
        args.destination_experiment,
        args.source_experiment,
        args.pool_size,
        results,
        config,
    )


if __name__ == "__main__":
    main()
