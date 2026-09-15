"""Evaluate an exact GRPO optimizer-update pool from a completed rolling run.

The source experiment is treated as immutable.  Each selected snapshot is copied
to a derived cell, ridge weights are refit with the source window's frozen
configuration, and the original annual test is evaluated by ``finalize_cell``.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import multiprocessing as mp
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlalpha.config import load_yaml, resolve_data_evaluation
from rlalpha.evaluation.finalize import finalize_cell
from rlalpha.evaluation.portfolio import return_metrics
from rlalpha.evaluation.statistics import bootstrap_date_indices, series_summary
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import atomic_write_text, write_json


METHOD = "grpo_llm"
REWARD = "r2_paired_oof"
IC_COLUMNS = ("raw_ic", "raw_rank_ic", "rnic", "rank_rnic")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshots(path: Path, optimizer_update: int) -> list[dict[str, Any]]:
    selected = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if int(item.get("optimizer_update", -1)) == optimizer_update:
                    selected.append(item)
    if not selected:
        raise RuntimeError(f"no snapshot at optimizer update {optimizer_update}: {path}")
    hashes = {str(item["pool_snapshot_hash"]) for item in selected}
    if len(hashes) != 1:
        raise RuntimeError(f"conflicting snapshots at optimizer update {optimizer_update}: {path}")
    return selected


def _derived_pool(snapshot: dict[str, Any], source_cell: Path, update: int) -> dict[str, Any]:
    result = dict(snapshot)
    result.update(
        {
            "final_pool_id": "final_pool_" + stable_hash(
                {
                    "source_cell": str(source_cell),
                    "snapshot_id": snapshot["snapshot_id"],
                    "optimizer_update": update,
                }
            )[:20],
            # finalize_cell deliberately requires this exact policy for recent-alpha.
            "selection_rule": "fixed_budget_terminal_pool",
            "calibration_policy": (
                "derived exact-update pool; full-train ridge fit after search; "
                "no distinct validation/calibration period"
            ),
            "status": "complete",
            "selected_from": {
                "source_cell": str(source_cell),
                "snapshot_id": snapshot["snapshot_id"],
                "optimizer_update": update,
                "selection_rule": "exact optimizer-update snapshot after committed update",
            },
        }
    )
    if int(result.get("optimizer_update", -1)) != update or not result.get("expressions"):
        raise RuntimeError(f"invalid derived pool from {source_cell}")
    return result


def _evaluate(job: tuple[str, str, str, int, int, str]) -> dict[str, Any]:
    source_text, destination_text, config_text, year, seed, processed_text = job
    source, destination, config = Path(source_text), Path(destination_text), Path(config_text)
    child = load_yaml(config)
    code_root = Path(child["paths"]["code_root"])
    data, evaluation = resolve_data_evaluation(child, code_root)
    # Per-factor intervals are diagnostics for this checkpoint comparison and
    # dominate runtime at 2,000 draws.  Headline and paired pool-level series
    # are bootstrapped with 2,000 draws in _report below.
    evaluation = {**evaluation, "bootstrap_samples": 100}
    snapshot = _snapshots(source / "checkpoints/snapshots.jsonl", UPDATE)[-1]
    pool = _derived_pool(snapshot, source, UPDATE)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "final_pool.json", pool)
    write_json(
        destination / "source_snapshot.json",
        {
            "source_cell": str(source),
            "source_snapshot_file": str(source / "checkpoints/snapshots.jsonl"),
            "source_snapshot_sha256": _sha256(source / "checkpoints/snapshots.jsonl"),
            "snapshot_id": snapshot["snapshot_id"],
            "pool_snapshot_hash": snapshot["pool_snapshot_hash"],
            "optimizer_update": UPDATE,
            "pool_version": snapshot["pool_version"],
            "pool_size": len(snapshot["expressions"]),
        },
    )
    marker = destination / "test/finalization.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            marker.unlink()
    metrics = finalize_cell(
        destination,
        processed_text,
        finalization_scope_hash=stable_hash(
            {"source": str(source), "year": year, "seed": seed, "optimizer_update": UPDATE}
        ),
        evaluation_config=evaluation,
        data_config=data,
        protocol=child.get("protocol"),
    )
    return {
        "year": year,
        "seed": seed,
        "pool_version": int(snapshot["pool_version"]),
        "pool_size": len(snapshot["expressions"]),
        "snapshot_id": snapshot["snapshot_id"],
        "rnic": float(metrics["primary_pearson_rnic"]["mean"]),
        "rank_rnic": float(metrics["rank_rnic"]["mean"]),
        "sharpe_0bps": float(metrics["portfolios"]["dollar_neutral"]["0bps"]["sharpe"]),
        "sharpe_10bps": float(metrics["portfolios"]["dollar_neutral"]["10bps"]["sharpe"]),
    }


def _evaluate_year(jobs: list[tuple[str, str, str, int, int, str]]) -> list[dict[str, Any]]:
    """Evaluate one year's seeds while reusing its immutable train/test panels."""
    from rlalpha.data.store import PanelStore

    faulthandler.register(signal.SIGUSR1, all_threads=True)

    original = PanelStore.load_interval
    cache: dict[tuple[str, str, str, int], Any] = {}

    def cached_load(self, name, start, end, history=252):
        key = (str(name), str(start), str(end), int(history))
        if key not in cache:
            cache[key] = original(self, name, start, end, history)
        return cache[key]

    PanelStore.load_interval = cached_load
    try:
        return [_evaluate(job) for job in jobs]
    finally:
        PanelStore.load_interval = original


def _summary(values: pd.DataFrame, column: str) -> dict[str, float]:
    values = values.sort_values("date")
    dates = pd.DatetimeIndex(values["date"])
    indices = bootstrap_date_indices(dates, block=20, samples=2000, seed=0)
    result = series_summary(
        values[column].to_numpy(float),
        hac_lag=20,
        bootstrap_samples=2000,
        seed=0,
        dates=dates,
        bootstrap_block=20,
        bootstrap_indices=indices,
    )
    low, high = result["bootstrap_95_ci"]
    return {
        "mean": float(result["mean"]),
        "hac_t": float(result["hac_t"]),
        "p_value": float(result["p_value"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(result["n"]),
    }


def _load_budget(root: Path, years: tuple[int, ...], seeds: tuple[int, ...]) -> tuple[dict, dict]:
    ic, portfolio = {}, {}
    for year in years:
        for seed in seeds:
            test = root / f"test_{year}" / METHOD / REWARD / f"seed_{seed}" / "test"
            left = pd.read_parquet(test / "rnic_daily.parquet")
            right = pd.read_parquet(test / "dollar_neutral_daily.parquet")
            left["date"], right["date"] = pd.to_datetime(left["date"]), pd.to_datetime(right["date"])
            ic[(year, seed)], portfolio[(year, seed)] = left, right
    return ic, portfolio


def _aggregate_budget(
    label: str,
    ic: dict,
    portfolio: dict,
    years: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    daily_ic = pd.concat(
        [frame.assign(seed=seed) for (year, seed), frame in ic.items()], ignore_index=True
    ).groupby("date", as_index=False)[list(IC_COLUMNS)].mean()
    headline: dict[str, Any] = {"budget": label}
    for metric in IC_COLUMNS:
        for key, value in _summary(daily_ic[["date", metric]], metric).items():
            headline[f"{metric}_{key}"] = value
    seed_returns = {0: [], 10: []}
    yearly = []
    for year in years:
        year_ic = pd.concat([ic[(year, seed)] for seed in seeds]).groupby("date", as_index=False)[list(IC_COLUMNS)].mean()
        row = {"budget": label, "test_year": year}
        for metric in IC_COLUMNS:
            row[metric] = float(year_ic[metric].mean())
        for cost in (0, 10):
            seed_metrics = []
            for seed in seeds:
                result = return_metrics(portfolio[(year, seed)][f"net_return_{cost}bps"].to_numpy(float))
                seed_metrics.append(result)
            row[f"annual_return_{cost}bps"] = float(np.mean([item["annual_return"] for item in seed_metrics]))
            row[f"sharpe_{cost}bps"] = float(np.mean([item["sharpe"] for item in seed_metrics]))
        yearly.append(row)
    for seed in seeds:
        path = pd.concat([portfolio[(year, seed)] for year in years], ignore_index=True).sort_values("date")
        for cost in (0, 10):
            seed_returns[cost].append(return_metrics(path[f"net_return_{cost}bps"].to_numpy(float)))
    for cost in (0, 10):
        headline[f"annual_return_{cost}bps"] = float(np.mean([x["annual_return"] for x in seed_returns[cost]]))
        headline[f"sharpe_{cost}bps"] = float(np.mean([x["sharpe"] for x in seed_returns[cost]]))
    return headline, yearly, {"daily_ic": daily_ic}


def _fmt(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _report(
    destination: Path,
    source: Path,
    years: tuple[int, ...],
    seeds: tuple[int, ...],
    selections: list[dict[str, Any]],
) -> None:
    ic100, portfolio100 = _load_budget(destination, years, seeds)
    ic200, portfolio200 = _load_budget(source, years, seeds)
    h100, y100, cache100 = _aggregate_budget("100", ic100, portfolio100, years, seeds)
    h200, y200, cache200 = _aggregate_budget("200", ic200, portfolio200, years, seeds)
    headlines = pd.DataFrame([h100, h200])
    headlines.to_csv(destination / "budget_headlines.csv", index=False)
    yearly = pd.DataFrame(y100 + y200)
    yearly.to_csv(destination / "yearly_results.csv", index=False)

    paired = cache100["daily_ic"].merge(cache200["daily_ic"], on="date", suffixes=("_100", "_200"), validate="one_to_one")
    paired_rows = []
    for metric in IC_COLUMNS:
        column = f"delta_{metric}"
        paired[column] = paired[f"{metric}_100"] - paired[f"{metric}_200"]
        paired_rows.append({"metric": metric, **_summary(paired[["date", column]], column)})
    pd.DataFrame(paired_rows).to_csv(destination / "paired_100_minus_200_ic.csv", index=False)
    paired.to_parquet(destination / "paired_daily_ic.parquet", index=False)

    pool_rows = []
    for item in selections:
        year, seed = int(item["year"]), int(item["seed"])
        pool100 = json.loads((destination / f"test_{year}" / METHOD / REWARD / f"seed_{seed}" / "final_pool.json").read_text())
        pool200 = json.loads((source / f"test_{year}" / METHOD / REWARD / f"seed_{seed}" / "final_pool.json").read_text())
        left, right = set(pool100["expressions"]), set(pool200["expressions"])
        pool_rows.append({
            **item,
            "pool_version_200": int(pool200["pool_version"]),
            "retained_from_100": len(left & right),
            "jaccard_100_200": len(left & right) / len(left | right),
        })
    pools = pd.DataFrame(pool_rows).sort_values(["year", "seed"])
    pools.to_csv(destination / "pool_changes.csv", index=False)

    delta_rows = []
    for year in years:
        a, b = yearly[(yearly.budget == "100") & (yearly.test_year == year)].iloc[0], yearly[(yearly.budget == "200") & (yearly.test_year == year)].iloc[0]
        delta_rows.append({
            "test_year": year,
            "rnic_100": a.rnic, "rnic_200": b.rnic, "delta_rnic": a.rnic - b.rnic,
            "rank_rnic_100": a.rank_rnic, "rank_rnic_200": b.rank_rnic, "delta_rank_rnic": a.rank_rnic - b.rank_rnic,
            "sharpe_10bps_100": a.sharpe_10bps, "sharpe_10bps_200": b.sharpe_10bps,
            "delta_sharpe_10bps": a.sharpe_10bps - b.sharpe_10bps,
        })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(destination / "yearly_100_vs_200.csv", index=False)
    paired_rnic = next(item for item in paired_rows if item["metric"] == "rnic")
    paired_rank = next(item for item in paired_rows if item["metric"] == "rank_rnic")

    headline_view = pd.DataFrame([
        {
            "steps": row["budget"], "RNIC": _fmt(row["rnic_mean"]), "Rank RNIC": _fmt(row["rank_rnic_mean"]),
            "AnnRet 0bps": f"{100*row['annual_return_0bps']:.2f}%", "Sharpe 0bps": _fmt(row["sharpe_0bps"], 3),
            "AnnRet 10bps": f"{100*row['annual_return_10bps']:.2f}%", "Sharpe 10bps": _fmt(row["sharpe_10bps"], 3),
        } for _, row in headlines.iterrows()
    ])
    delta_view = delta.copy()
    for column in delta_view.columns[1:]:
        delta_view[column] = delta_view[column].map(lambda value: _fmt(float(value), 4))
    text = [
        "# GRPO-R2 exact step-100 pool evaluation",
        "",
        f"Source: `{source}`. The 100-step pool is the exact snapshot after optimizer update 100; all source artifacts remain unchanged. Ridge weights are refitted on the same two-calendar-year train window, followed by the identical next-calendar-year test and dollar-neutral 0/10bps portfolio evaluation used at 200 steps.",
        "",
        "## Overall result",
        "",
        headline_view.to_markdown(index=False),
        "",
        f"Paired daily 100−200 RNIC: **{paired_rnic['mean']:+.4f}** (HAC t={paired_rnic['hac_t']:.2f}, p={paired_rnic['p_value']:.4g}, 95% block-bootstrap CI [{paired_rnic['ci_low']:.4f}, {paired_rnic['ci_high']:.4f}]).",
        f"Paired daily 100−200 rank RNIC: **{paired_rank['mean']:+.4f}** (HAC t={paired_rank['hac_t']:.2f}, p={paired_rank['p_value']:.4g}, 95% block-bootstrap CI [{paired_rank['ci_low']:.4f}, {paired_rank['ci_high']:.4f}]).",
        "",
        "## By test year",
        "",
        delta_view.to_markdown(index=False),
        "",
        "## Pool evolution from step 100 to 200",
        "",
        pools[["year", "seed", "pool_version", "pool_version_200", "pool_size", "retained_from_100", "jaccard_100_200"]].to_markdown(index=False),
        "",
        "Machine-readable outputs: `budget_headlines.csv`, `yearly_results.csv`, `yearly_100_vs_200.csv`, `paired_100_minus_200_ic.csv`, `paired_daily_ic.parquet`, and `pool_changes.csv`.",
    ]
    atomic_write_text(destination / "REPORT.md", "\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    global UPDATE
    UPDATE = int(args.update)
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination:
        raise RuntimeError("source and destination must differ")
    manifest = json.loads((source / "rolling_manifest.json").read_text(encoding="utf-8"))
    all_years = tuple(int(year) for year in manifest["windows"])
    all_seeds = tuple(int(seed) for seed in manifest["windows"][str(all_years[0])]["experiment"]["seeds"])
    years = tuple(args.years) if args.years else all_years
    seeds = tuple(args.seeds) if args.seeds else all_seeds
    if set(years) - set(all_years) or set(seeds) - set(all_seeds):
        raise ValueError("requested years/seeds are outside the source rolling contract")
    cells = [(year, seed) for year in years for seed in seeds]
    destination.mkdir(parents=True, exist_ok=True)
    jobs = []
    source_hashes = {}
    for year, seed in cells:
        source_cell = source / f"test_{year}" / METHOD / REWARD / f"seed_{seed}"
        destination_cell = destination / f"test_{year}" / METHOD / REWARD / f"seed_{seed}"
        source_hashes[str(source_cell / "final_pool.json")] = _sha256(source_cell / "final_pool.json")
        marker = destination_cell / "test/finalization.json"
        if args.resume and marker.exists():
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                continue
        jobs.append((
            str(source_cell), str(destination_cell), str(source / "window_configs" / f"test_{year}.yaml"),
            year, seed, str(manifest["windows"][str(year)]["paths"]["processed_root"]),
        ))
    results = []
    if jobs:
        jobs_by_year = [
            [job for job in jobs if int(job[3]) == year]
            for year in years
            if any(int(job[3]) == year for job in jobs)
        ]
        # Numba/OpenMP kernels used by the factor evaluator can deadlock when a
        # process is forked after the runtime has initialized.  Spawn gives each
        # yearly worker a clean numerical runtime.
        if args.workers == 1:
            for group in jobs_by_year:
                for result in _evaluate_year(group):
                    results.append(result)
                    print(json.dumps({"status": "complete", **result}, sort_keys=True), flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=max(1, args.workers), mp_context=mp.get_context("spawn")
            ) as executor:
                futures = {executor.submit(_evaluate_year, group): group for group in jobs_by_year}
                for future in as_completed(futures):
                    for result in future.result():
                        results.append(result)
                        print(json.dumps({"status": "complete", **result}, sort_keys=True), flush=True)
    for year, seed in cells:
        cell = destination / f"test_{year}" / METHOD / REWARD / f"seed_{seed}"
        snapshot = json.loads((cell / "source_snapshot.json").read_text(encoding="utf-8"))
        results.append({"year": year, "seed": seed, "pool_version": snapshot["pool_version"], "pool_size": snapshot["pool_size"], "snapshot_id": snapshot["snapshot_id"]})
    unique = {(item["year"], item["seed"]): item for item in results}
    if len(unique) != len(cells):
        raise RuntimeError("derived evaluation is incomplete")
    if source_hashes != {path: _sha256(Path(path)) for path in source_hashes}:
        raise RuntimeError("source final pools changed during derived evaluation")
    selections = list(unique.values())
    if set(years) != set(all_years) or set(seeds) != set(all_seeds):
        write_json(destination / f"partial_{'_'.join(map(str, years))}_{'_'.join(map(str, seeds))}.json", {
            "status": "complete", "source": str(source), "optimizer_update": UPDATE,
            "method": METHOD, "reward": REWARD, "cells": selections,
        })
        print(json.dumps({"status": "partial_complete", "cells": len(selections)}, sort_keys=True))
        return
    write_json(destination / "evaluation_manifest.json", {
        "status": "complete", "source": str(source), "optimizer_update": UPDATE,
        "method": METHOD, "reward": REWARD, "cells": selections, "source_final_pools_unchanged": True,
        "per_factor_bootstrap_samples": 100, "aggregate_bootstrap_samples": 2000,
    })
    _report(destination, source, years, seeds, selections)
    print(json.dumps({"status": "complete", "report": str(destination / "REPORT.md")}, sort_keys=True))


UPDATE = 100


if __name__ == "__main__":
    main()
