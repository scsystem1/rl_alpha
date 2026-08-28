from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from ..config import load_paths, load_yaml
from ..evaluation.statistics import paired_summary
from ..utils.io import atomic_write_text
from ..utils.experiment_log import append_event, update_progress


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _seed_ci(values: pd.Series) -> str:
    finite = np.asarray(values.dropna(), dtype=float)
    if len(finite) < 2:
        return json.dumps([float(finite[0]), float(finite[0])]) if len(finite) else json.dumps([None, None])
    half_width = float(t.ppf(0.975, len(finite) - 1) * finite.std(ddof=1) / np.sqrt(len(finite)))
    return json.dumps([float(finite.mean() - half_width), float(finite.mean() + half_width)])


def _average_daily(cells: dict[tuple[str, str, int], Path], method: str, reward: str, seeds: set[int], filename: str, column: str) -> pd.DataFrame:
    frames = []
    for seed in sorted(seeds):
        path = cells[(method, reward, seed)] / "test" / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date", column]).rename(columns={column: f"seed_{seed}"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "value"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner", validate="one_to_one")
    # A cross-seed daily result is valid only when every declared seed has a
    # value.  Pandas' default skipna=True would silently turn a missing held
    # return in one seed into a usable portfolio return.
    merged["value"] = merged.drop(columns="date").mean(axis=1, skipna=False)
    return merged[["date", "value"]]


def _sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    volatility = values.std(ddof=1) if len(values) > 1 else float("nan")
    return float(np.sqrt(252.0) * values.mean() / volatility) if np.isfinite(volatility) and volatility > 0 else float("nan")


def _paired_sharpe(first: np.ndarray, second: np.ndarray, samples: int = 2000, block: int = 20, seed: int = 0) -> tuple[float, list[float]]:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or not len(first) or not np.isfinite(first).all() or not np.isfinite(second).all():
        return float("nan"), [float("nan"), float("nan")]
    delta = _sharpe(first) - _sharpe(second)
    if len(first) < 2:
        return delta, [float("nan"), float("nan")]
    block = min(block, len(first))
    starts = np.arange(len(first) - block + 1)
    n_blocks = int(np.ceil(len(first) / block))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        positions = np.concatenate([np.arange(start, start + block) for start in chosen])[: len(first)]
        draws[index] = _sharpe(first[positions]) - _sharpe(second[positions])
    return delta, [float(value) for value in np.nanquantile(draws, [0.025, 0.975])]


def _comparison_pairs(keys: set[tuple[str, str]]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    pairs = set()
    for reward in sorted({reward for _, reward in keys}):
        baseline = ("random", reward)
        if baseline in keys:
            pairs.update((key, baseline) for key in keys if key[1] == reward and key != baseline)
    grpo_r0 = ("grpo_llm", "r0")
    if grpo_r0 in keys:
        pairs.update((key, grpo_r0) for key in keys if key[0] == "grpo_llm" and key != grpo_r0)
    return sorted(pairs)


def build_report(experiment_id: str, config: str | Path, methods: list[str] | None = None) -> dict[str, Any]:
    paths = load_paths(config)
    raw_config = load_yaml(config)
    root = paths.runs_root / experiment_id
    append_event(root / "experiment.log", "report_started", experiment_id=experiment_id)
    cells: dict[tuple[str, str, int], Path] = {}
    search_rows, pool_rows, portfolio_rows = [], [], []
    selected_methods = set(methods or ())
    for metrics_path in sorted(root.glob("*/*/seed_*/test/metrics.json")):
        cell = metrics_path.parents[1]
        method, reward, seed_name = cell.relative_to(root).parts
        if selected_methods and method not in selected_methods:
            continue
        seed = int(seed_name.removeprefix("seed_"))
        key = method, reward, seed
        cells[key] = cell
        metrics = _read_json(metrics_path)
        train = _read_json(cell / "train_metrics.json")
        selected = _read_json(cell / "final_pool.json")
        checkpoint = _read_json(cell / "checkpoint.json")
        candidate_path = cell / "candidates.parquet"
        valid = int(pd.read_parquet(candidate_path, columns=["valid"])["valid"].sum()) if candidate_path.exists() else train.get("valid_unique_evaluations")
        admitted = sum(bool(item.get("admitted")) for item in checkpoint.get("pool_history", []))
        search_rows.append({
            "method": method, "reward": reward, "seed": seed,
            "raw_proposals": train.get("raw_proposals"), "valid": valid,
            "unique": train.get("valid_unique_evaluations"), "admitted": admitted,
            "pool_size": metrics.get("pool_size"), "tokens": train.get("tokens", 0),
            "gpu_hours": float(train.get("gpu_seconds", 0.0)) / 3600.0,
            "wall_hours": float(train.get("wall_seconds", 0.0)) / 3600.0,
        })
        pool_rows.append({
            "method": method, "reward": reward, "seed": seed,
            "train_objective": selected.get("train", {}).get("objective"),
            "validation_objective": selected.get("validation", {}).get("objective"),
            "test_raw_ic_diagnostic": metrics.get("raw_ic_diagnostic", metrics.get("raw_ic", {})).get("mean"),
            "test_primary_pearson_rnic": metrics.get("primary_pearson_rnic", metrics.get("rnic", {})).get("mean"),
            "nw_t": metrics.get("rnic", {}).get("hac_t"),
            "test_rnic_95_ci": json.dumps(metrics.get("rnic", {}).get("bootstrap_95_ci")),
            "average_pair_correlation": metrics.get("average_pair_correlation"),
            "neutralization_retention": metrics.get("neutralization_retention"),
            "pool_size": metrics.get("pool_size"),
        })
        for portfolio in ("dollar_neutral", "fully_neutral"):
            portfolio_metrics = metrics.get("portfolios", {}).get(portfolio, {})
            for cost in ("0bps", "10bps"):
                values = portfolio_metrics.get(cost, {})
                portfolio_rows.append({
                    "method": method, "reward": reward, "seed": seed, "portfolio": portfolio, "cost": cost,
                    "annual_return": values.get("annual_return"), "sharpe": values.get("sharpe"),
                    "max_drawdown": values.get("max_drawdown"), "turnover": values.get("average_turnover"),
                    "average_gross": values.get("average_gross"), "average_net": values.get("average_net"),
                    "max_risk_exposure": portfolio_metrics.get("max_realized_risk_exposure"),
                    "infeasible_days": values.get("infeasible_days"),
                    "missing_held_returns": values.get("missing_held_returns"),
                    "missing_held_return_days": values.get("missing_held_return_days"),
                    "held_return_weight_coverage": values.get("held_return_weight_coverage"),
                    "maximum_missing_held_return_weight": values.get("maximum_missing_held_return_weight"),
                    "missing_held_exposure_days": portfolio_metrics.get("missing_held_exposure_days"),
                })

    experiment = raw_config.get("experiment")
    if experiment is not None:
        if "cells" in experiment:
            expected = {(method, reward, int(seed)) for method, reward in experiment["cells"] for seed in experiment["seeds"]}
        else:
            expected = {(method, reward, int(seed)) for method in experiment["methods"] for reward in experiment["rewards"] for seed in experiment["seeds"]}
        if selected_methods:
            unknown = selected_methods - {method for method, _, _ in expected}
            if unknown:
                raise ValueError(f"methods are not configured for this experiment: {sorted(unknown)}")
            expected = {cell for cell in expected if cell[0] in selected_methods}
        missing = sorted(expected - set(cells))
        unexpected = sorted(set(cells) - expected)
        if missing or unexpected:
            raise RuntimeError(f"formal aggregate refused: missing_cells={missing}, unexpected_cells={unexpected}")
        scope_name = "_".join(sorted(selected_methods)) or "all"
        transaction = _read_json(root / ("test_finalization.json" if scope_name == "all" else f"test_finalization_{scope_name}.json"))
        if transaction.get("status") != "complete":
            raise RuntimeError("formal aggregate refused: experiment test finalization is not complete")
        budget = int(experiment["valid_unique_budget"])
        comparability = set()
        for key, cell in cells.items():
            state = _read_json(cell / "progress.json")
            train = _read_json(cell / "train_metrics.json")
            marker = _read_json(cell / "test/finalization.json")
            if state.get("status") != "complete" or int(state.get("budget", -1)) != budget:
                raise RuntimeError(f"formal aggregate refused: incomplete or wrong-budget state for {key}")
            if int(train.get("valid_unique_evaluations", -1)) < budget:
                raise RuntimeError(f"formal aggregate refused: budget not met for {key}")
            if marker.get("status") != "complete":
                raise RuntimeError(f"formal aggregate refused: cell finalization incomplete for {key}")
            import yaml

            manifest_path = cell / "manifest.yaml"
            if not manifest_path.exists():
                raise RuntimeError(f"formal aggregate refused: manifest missing for {key}")
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            panel_identity = tuple((Path(item["path"]).name, item["sha256"]) for item in manifest.get("panel_artifacts", []))
            comparability.add((panel_identity, manifest.get("evaluator_version")))
        if len(comparability) != 1:
            raise RuntimeError("formal aggregate refused: panel/evaluator fingerprints differ across cells")

    search = pd.DataFrame(search_rows)
    pools = pd.DataFrame(pool_rows)
    portfolios = pd.DataFrame(portfolio_rows)
    artifact_suffix = "" if not selected_methods else "_" + "_".join(sorted(selected_methods))
    for name, frame in (("search_efficiency", search), ("pool_quality", pools), ("portfolio_results", portfolios)):
        frame.to_parquet(root / f"{name}{artifact_suffix}.parquet", index=False)
        frame.to_csv(root / f"{name}{artifact_suffix}.csv", index=False)

    paired_rows = []
    keys = {(method, reward) for method, reward, _ in cells}
    for (first_method, first_reward), (second_method, second_reward) in _comparison_pairs(keys):
        first_seeds = {seed for method, reward, seed in cells if (method, reward) == (first_method, first_reward)}
        second_seeds = {seed for method, reward, seed in cells if (method, reward) == (second_method, second_reward)}
        seeds = first_seeds & second_seeds
        first_ic = _average_daily(cells, first_method, first_reward, seeds, "rnic_daily.parquet", "rnic")
        second_ic = _average_daily(cells, second_method, second_reward, seeds, "rnic_daily.parquet", "rnic")
        joined_ic = first_ic.merge(second_ic, on="date", suffixes=("_first", "_second"), validate="one_to_one")
        summary = paired_summary(joined_ic["value_first"].to_numpy(), joined_ic["value_second"].to_numpy(), bootstrap_samples=2000)
        first_return = _average_daily(cells, first_method, first_reward, seeds, "fully_neutral_daily.parquet", "net_return_10bps")
        second_return = _average_daily(cells, second_method, second_reward, seeds, "fully_neutral_daily.parquet", "net_return_10bps")
        joined_return = first_return.merge(second_return, on="date", suffixes=("_first", "_second"), validate="one_to_one")
        delta_sharpe, sharpe_ci = _paired_sharpe(joined_return["value_first"].to_numpy(), joined_return["value_second"].to_numpy())
        ic_ci = summary["bootstrap_95_ci"]
        significance = "uncertain" if ic_ci[0] <= 0 <= ic_ci[1] else ("positive" if summary["mean"] > 0 else "negative")
        dimension = "search_algorithm" if first_reward == second_reward else "reward"
        paired_rows.append({
            "comparison": f"{first_method}-{first_reward} minus {second_method}-{second_reward}",
            "matched_seeds": json.dumps(sorted(seeds)), "paired_dates": summary["n"],
            "delta_rnic": summary["mean"], "delta_rnic_hac_t": summary["hac_t"],
            "delta_rnic_95_ci": json.dumps(ic_ci), "delta_fully_neutral_10bps_sharpe": delta_sharpe,
            "delta_sharpe_95_ci": json.dumps(sharpe_ci), "interpretation": f"{dimension}:{significance}",
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_parquet(root / f"paired_comparisons{artifact_suffix}.parquet", index=False)
    paired.to_csv(root / f"paired_comparisons{artifact_suffix}.csv", index=False)

    if len(pools):
        summary = pools.groupby(["method", "reward"], as_index=False).agg(
            seeds=("seed", "count"), test_rnic_mean=("test_primary_pearson_rnic", "mean"), test_rnic_seed_std=("test_primary_pearson_rnic", "std"),
            nw_t_mean=("nw_t", "mean"), pool_size_mean=("pool_size", "mean"),
        )
        cis = pools.groupby(["method", "reward"])["test_primary_pearson_rnic"].apply(_seed_ci).rename("test_rnic_seed_95_ci").reset_index()
        summary = summary.merge(cis, on=["method", "reward"])
    else:
        summary = pd.DataFrame()
    summary.to_parquet(root / f"cross_method_summary{artifact_suffix}.parquet", index=False)
    summary.to_csv(root / f"cross_method_summary{artifact_suffix}.csv", index=False)

    sections = [f"# {experiment_id}", f"Completed cells: {len(cells)}"]
    for title, frame in (("Search efficiency", search), ("Factor pool quality", pools), ("Portfolio results", portfolios), ("Paired comparisons", paired), ("Across-seed summary", summary)):
        sections.extend([f"## {title}", frame.to_markdown(index=False) if len(frame) else "No completed rows."])
    sections.extend(["## Interpretation notes", "Paired RNIC intervals use same-date HAC and 20-day moving-block bootstrap. Sharpe differences use paired 20-day block resampling of fully-neutral 10bps daily returns. `uncertain` means the paired RNIC interval includes zero; it is not evidence of equivalence. GPU and wall time must be considered alongside statistical uncertainty."])
    scope_name = "_".join(sorted(selected_methods)) or "all"
    report_name = "report.md" if scope_name == "all" else f"report_{scope_name}.md"
    atomic_write_text(root / report_name, "\n\n".join(sections) + "\n")
    update_progress(root / "progress.json", status="complete", completed_cells=len(cells), report=report_name)
    append_event(root / "experiment.log", "report_finished", completed_cells=len(cells), paired_comparisons=len(paired), report=report_name)
    return {"experiment_id": experiment_id, "completed_cells": len(cells), "paired_comparisons": len(paired), "report": str(root / report_name)}
