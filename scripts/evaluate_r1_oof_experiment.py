"""Freeze, evaluate, and report a formal 12-cell OOF experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rlalpha.config import load_paths, load_yaml
from rlalpha.evaluation.finalize import finalize_cell, finalize_experiment
from rlalpha.matrix.runner import _cell_acceptance, _expected_cell_identity
from rlalpha.reporting.build import build_report
from rlalpha.utils.experiment_log import update_progress
from rlalpha.utils.hashing import stable_hash


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _evaluate_one(arguments: tuple[str, str, str, int, str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    cell_text, processed_root, reward, seed, experiment_id, evaluation_config = arguments
    cell = Path(cell_text)
    pool_path = cell / "final_pool.json"
    before = _digest(pool_path)
    key = f"{cell.parts[-3]}/{reward}/seed_{seed}"
    scope_hash = stable_hash({"cell": key, "support": "fixed-universe-zero-fill-psd-gram-v6"})
    metrics = finalize_cell(
        cell,
        processed_root,
        finalization_scope_hash=scope_hash,
        evaluation_config=evaluation_config,
    )
    after = _digest(pool_path)
    if before != after:
        raise RuntimeError(f"evaluation changed the frozen pool for {key}")
    return key, {
        "primary_pearson_rnic": metrics["primary_pearson_rnic"]["mean"],
        "rank_rnic": metrics["rank_rnic"]["mean"],
        "fully_neutral_10bps_sharpe": metrics["portfolios"]["fully_neutral"]["10bps"]["sharpe"],
        "final_pool_sha256": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = args.config.resolve()
    raw = load_yaml(config)
    paths = load_paths(config)
    experiment = raw["experiment"]
    steps = int(experiment["search_steps"])
    targets = [
        (str(method), str(reward), int(seed))
        for method, reward in experiment["cells"]
        for seed in experiment["seeds"]
    ]
    root = paths.runs_root / args.experiment_id

    # Opening test is an experiment-wide action: first prove that all searches
    # are complete and bind every cell to its exact code/config/data identity.
    for method, reward, seed in targets:
        cell = root / method / reward / f"seed_{seed}"
        accepted, reason = _cell_acceptance(cell, steps)
        if not accepted:
            raise RuntimeError(f"search acceptance failed for {method}/{reward}/seed_{seed}: {reason}")
        state_path = cell / "progress.json"
        prior = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        identity = _expected_cell_identity(config, paths, method, reward, seed, steps)
        update_progress(
            state_path,
            **{
                **prior,
                "status": "complete",
                "method": method,
                "reward": reward,
                "seed": seed,
                "search_steps": steps,
                "cell_identity": identity,
            },
        )

    evaluation_config = load_yaml(paths.code_root / "configs/eval/preliminary.yaml")["evaluation"]
    jobs = [
        (
            str(root / method / reward / f"seed_{seed}"),
            str(paths.processed_root),
            reward,
            seed,
            args.experiment_id,
            evaluation_config,
        )
        for method, reward, seed in targets
    ]
    completed: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_evaluate_one, job): job for job in jobs}
        for future in as_completed(futures):
            key, metrics = future.result()
            completed[key] = metrics
            print(json.dumps({"evaluation_complete": key, **metrics}, sort_keys=True), flush=True)

    # This second pass is cache-only for each cell.  It writes the official
    # experiment-wide finalization transaction and summary after every cell is
    # known to have evaluated successfully.
    summary = finalize_experiment(args.experiment_id, config)
    failures = {key: value for key, value in summary.items() if value.get("status") != "complete"}
    if failures:
        raise RuntimeError(f"combined evaluation failed: {failures}")
    build_report(args.experiment_id, config)
    output = root / "evaluation_status.json"
    output.write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment_id": args.experiment_id,
                "evaluated_cells": len(completed),
                "cells": completed,
                "report": str(root / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "evaluated_cells": len(completed), "report": str(root / "report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
