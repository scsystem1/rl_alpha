from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import load_paths, load_yaml
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.calculator import FactorCalculator
from ..factors.combiner import RidgeCombiner
from ..factors.transform import (
    IndependentFactorTransformPipeline,
    TransformConfig,
    combine_available_signals,
)
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import write_json
from ..utils.experiment_log import append_event, update_progress, write_result_summary
from .portfolio import PortfolioBacktester, portfolio_metrics
from .statistics import benjamini_hochberg, factor_significance, series_summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_code_fingerprints() -> list[dict[str, Any]]:
    package = Path(__file__).parents[1]
    relatives = (
        "data/store.py", "dsl/evaluator.py", "dsl/parser.py",
        "factors/calculator.py", "factors/combiner.py", "factors/transform.py", "risk/neutralize.py",
        "evaluation/portfolio.py", "evaluation/statistics.py", "evaluation/finalize.py",
    )
    return [file_fingerprint(package / relative) for relative in relatives]


def _panel_fingerprints(processed_root: Path) -> list[dict[str, Any]]:
    panel = processed_root / "panel"
    return [file_fingerprint(path) for path in (panel / "index.json", panel / "build_manifest.yaml", panel / "risk_build_manifest.yaml")]


def _daily_correlations(signal: np.ndarray, label: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.full(len(signal), np.nan)
    rank = np.full(len(signal), np.nan)
    for day in range(len(signal)):
        common = mask[day] & np.isfinite(signal[day]) & np.isfinite(label[day])
        if common.sum() < 3:
            continue
        left, right = signal[day, common], label[day, common]
        pearson[day] = np.corrcoef(left, right)[0, 1]
        rank[day] = np.corrcoef(rankdata(left), rankdata(right))[0, 1]
    return pearson, rank


def _signals(panel: SplitPanel, expressions: list[str]) -> list[np.ndarray]:
    return [panel.evaluate(parse_expression(expression)) for expression in expressions]


def _average_pair_correlation(signals: list[np.ndarray]) -> float:
    correlations = []
    for left_index, left in enumerate(signals):
        for right in signals[left_index + 1 :]:
            common = np.isfinite(left) & np.isfinite(right)
            if common.sum() >= 3:
                correlations.append(abs(float(np.corrcoef(left[common], right[common])[0, 1])))
    return float(np.mean(correlations)) if correlations else float("nan")


def _max_abs_exposure(realized: np.ndarray) -> float:
    return float(np.nanmax(np.abs(realized))) if np.isfinite(realized).any() else float("nan")


def _cell_identity(run_dir: Path) -> dict[str, object]:
    parts = run_dir.parts
    seed_name = parts[-1]
    return {
        "method": parts[-3] if len(parts) >= 3 else "unknown",
        "reward": parts[-2] if len(parts) >= 2 else "unknown",
        "seed": int(seed_name.removeprefix("seed_")) if seed_name.startswith("seed_") and seed_name.removeprefix("seed_").isdigit() else -1,
    }


def _configured_cells(experiment: dict[str, Any]) -> set[tuple[str, str, int]]:
    seeds = [int(seed) for seed in experiment["seeds"]]
    if "cells" in experiment:
        return {(str(method), str(reward), seed) for method, reward in experiment["cells"] for seed in seeds}
    return {
        (str(method), str(reward), seed)
        for method in experiment["methods"]
        for reward in experiment["rewards"]
        for seed in seeds
    }


def _assert_experiment_frozen(config: str | Path, paths: Any, root: Path, experiment: dict[str, Any], methods: list[str] | None = None) -> list[Path]:
    """Refuse test access until the requested method cells are frozen."""
    from ..matrix.runner import _cell_acceptance, _expected_cell_identity

    expected = _configured_cells(experiment)
    selected = set(methods or ())
    if selected:
        unknown = selected - {method for method, _, _ in expected}
        if unknown:
            raise ValueError(f"methods are not configured for this experiment: {sorted(unknown)}")
        expected = {cell for cell in expected if cell[0] in selected}
    actual = {}
    for final_pool in root.glob("*/*/seed_*/final_pool.json"):
        relative = final_pool.parent.relative_to(root).parts
        if len(relative) != 3 or not relative[2].startswith("seed_"):
            continue
        actual[(relative[0], relative[1], int(relative[2].removeprefix("seed_")))] = final_pool
    missing = sorted(expected - set(actual))
    if missing:
        raise RuntimeError(f"test opening refused: missing_cells={missing}")
    budget = int(experiment["valid_unique_budget"])
    comparability = set()
    for method, reward, seed in sorted(expected):
        directory = root / method / reward / f"seed_{seed}"
        state_path = directory / "progress.json"
        if not state_path.exists():
            raise RuntimeError(f"test opening refused: cell state missing for {(method, reward, seed)}")
        state = _read_json(state_path)
        identity = _expected_cell_identity(Path(config).resolve(), paths, method, reward, seed, budget)
        if state.get("status") != "complete" or int(state.get("budget", -1)) != budget or state.get("cell_identity") != identity:
            raise RuntimeError(f"test opening refused: incomplete or incompatible state for {(method, reward, seed)}")
        accepted, reason = _cell_acceptance(directory, budget)
        if not accepted:
            raise RuntimeError(f"test opening refused: {(method, reward, seed)}: {reason}")
        import yaml

        manifest = yaml.safe_load((directory / "manifest.yaml").read_text(encoding="utf-8")) or {}
        panel_identity = tuple((Path(item["path"]).name, item["sha256"]) for item in manifest.get("panel_artifacts", []))
        comparability.add((panel_identity, manifest.get("evaluator_version")))
    if len(comparability) != 1:
        raise RuntimeError("test opening refused: panel/evaluator fingerprints differ across cells")
    return [actual[key] for key in sorted(expected)]


def _write_factor_statistics(
    test_dir: Path,
    run_dir: Path,
    selected: dict[str, Any],
    expressions: list[str],
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    raw_signals: list[np.ndarray],
    raw_label: np.ndarray,
    transformed_signals: tuple[np.ndarray, ...],
    transformed_label: np.ndarray,
    common_mask: np.ndarray,
    diagnostics: tuple[dict[str, Any], ...],
    evaluation_config: dict[str, Any],
) -> None:
    identity = _cell_identity(run_dir)
    final_pool_id = str(selected.get("final_pool_id") or f"final_pool_{stable_hash({'cell': identity, 'pool_version': selected.get('pool_version'), 'expressions': expressions})[:20]}")
    lineage_by_expression = {str(item.get("expression")): item for item in selected.get("factors", [])}
    raw_calculator = FactorCalculator(raw_label, common_mask)
    raw_prepared = [raw_calculator.standardize(signal) for signal in raw_signals]
    daily_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for index, expression in enumerate(expressions):
        lineage = lineage_by_expression.get(expression, {})
        factor_id = str(lineage.get("factor_id") or stable_hash({"canonical_expression": expression}))
        lineage_id = str(lineage.get("factor_lineage_id") or lineage.get("proposal_id") or "legacy_unknown")
        raw_ic, raw_rank = _daily_correlations(raw_prepared[index], raw_label, common_mask)
        rnic, rank_rnic = _daily_correlations(transformed_signals[index], transformed_label, common_mask)
        n_obs = (common_mask & np.isfinite(transformed_signals[index]) & np.isfinite(transformed_label)).sum(axis=1)
        diag_by_day = {int(item.get("date")): item for item in diagnostics if str(item.get("date", "")).isdigit()}
        frame = pd.DataFrame({
            "date": dates,
            "n_obs": n_obs,
            "raw_pearson_ic": raw_ic,
            "raw_rank_ic": raw_rank,
            "pearson_rnic": rnic,
            "rank_rnic": rank_rnic,
            "residualization_status": [diag_by_day.get(day, {}).get("status", "missing") for day in range(len(dates))],
            "exposure_rank": [diag_by_day.get(day, {}).get("rank") for day in range(len(dates))],
            "condition_number": [diag_by_day.get(day, {}).get("condition_number") for day in range(len(dates))],
            "max_residual_exposure": [diag_by_day.get(day, {}).get("max_residual_exposure") for day in range(len(dates))],
        })
        for key, value in {**identity, "final_pool_id": final_pool_id, "factor_id": factor_id, "factor_lineage_id": lineage_id, "expression": expression}.items():
            frame.insert(len(frame.columns), key, value)
        daily_rows.append(frame)
        direction = float(np.sign(weights[index]))
        for metric, values in (("pearson_rnic", rnic), ("rank_rnic", rank_rnic)):
            record = factor_significance(values, hac_lag=int(evaluation_config["hac_lag"]), bootstrap_block=int(evaluation_config["bootstrap_block_length"]), bootstrap_samples=int(evaluation_config["bootstrap_samples"]), seed=int(evaluation_config["bootstrap_seed"]) + (int(identity["seed"]) if int(identity["seed"]) >= 0 else 0))
            summary_rows.append({
                **identity,
                "final_pool_id": final_pool_id,
                "factor_id": factor_id,
                "factor_lineage_id": lineage_id,
                "expression": expression,
                "metric": metric,
                "final_ridge_weight": float(weights[index]),
                "formula_rnic_mean": record["mean"],
                "direction_adjusted_rnic_mean": direction * float(record["mean"]),
                **record,
            })
    daily = pd.concat(daily_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    summary["q_value"] = np.nan
    for metric in summary["metric"].unique():
        selected_rows = summary["metric"].eq(metric)
        summary.loc[selected_rows, "q_value"] = benjamini_hochberg(summary.loc[selected_rows, "p_value"].to_numpy())
    fdr_threshold = float(evaluation_config["fdr_threshold"])
    summary["fdr_threshold"] = fdr_threshold
    summary["significant_fdr_5pct"] = summary["q_value"] <= fdr_threshold
    daily.to_parquet(test_dir / "factor_rnic_daily.parquet", index=False)
    summary.to_parquet(test_dir / "factor_significance.parquet", index=False)
    summary.to_csv(test_dir / "factor_significance.csv", index=False)
    write_json(test_dir / "factor_significance_metadata.json", {
        "schema_version": 1,
        "primary_metrics": ["pearson_rnic", "rank_rnic"],
        "hac_lag": evaluation_config["hac_lag"],
        "bootstrap": {"method": "moving_block", "block_length": evaluation_config["bootstrap_block_length"], "samples": evaluation_config["bootstrap_samples"], "seed": int(evaluation_config["bootstrap_seed"]) + int(identity["seed"])},
        "multiple_testing": {"method": "Benjamini-Hochberg", "scope": "final_pool_by_metric", "threshold": fdr_threshold},
        "test_direction_policy": "formula direction and ridge-weight sign were frozen before test; no test sign flip",
    })


def finalize_cell(
    run_dir: str | Path,
    processed_root: str | Path,
    bootstrap_samples: int = 2000,
    trade_mask_override: np.ndarray | None = None,
    fit_mask_override: np.ndarray | None = None,
    finalization_scope_hash: str | None = None,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir, processed_root = Path(run_dir), Path(processed_root)
    final_pool_path = run_dir / "final_pool.json"
    selected = json.loads(final_pool_path.read_text(encoding="utf-8"))
    evaluation_config = evaluation_config or {
        "ridge_lambda": 1e-3, "hac_lag": 20, "rebalance_days": 5, "holding_days": 20,
        "one_way_cost_bps": [0, 10], "fully_neutral_max_weight": 0.02,
        "net_tolerance": 1e-8, "exposure_tolerance": 1e-6, "gross_tolerance": 1e-6,
        "weight_tolerance": 1e-6, "bootstrap_block_length": 20, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": 0, "fdr_threshold": 0.05,
    }
    bootstrap_samples = int(evaluation_config["bootstrap_samples"])
    expressions = list(selected.get("expressions", []))
    if not expressions:
        raise ValueError(f"selected pool is empty: {run_dir}")
    input_hash = stable_hash({"schema_version": 9, "final_pool": file_fingerprint(final_pool_path), "panel": _panel_fingerprints(processed_root), "evaluation_code": _evaluation_code_fingerprints(), "evaluation_config": evaluation_config, "finalization_scope_hash": finalization_scope_hash, "support_policy": "independent-factor-availability-v1", "missing_return_policy": "zero-return-stale-value-v1"})
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    marker = test_dir / "finalization.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("input_hash") != input_hash:
            raise RuntimeError("test finalization input changed after test was opened")
        if state.get("status") == "complete":
            return json.loads((test_dir / "metrics.json").read_text(encoding="utf-8"))
    write_json(marker, {"status": "started", "input_hash": input_hash})

    store = PanelStore(processed_root)
    train, validation, test = (store.load_split(name) for name in ("train", "validation", "test"))
    fit_signals = [np.concatenate(pair, axis=0) for pair in zip(_signals(train, expressions), _signals(validation, expressions))]
    fit_mask = np.concatenate([train.target(train.common_mask), validation.target(validation.common_mask)])
    fit_exposures = np.concatenate([train.target(train.exposures), validation.target(validation.exposures)])
    fit_label = np.concatenate([train.target(train.label), validation.target(validation.label)])
    if fit_mask_override is not None:
        if fit_mask_override.shape != fit_mask.shape:
            raise ValueError("shared fit mask shape differs from train+validation panel")
        fit_mask &= fit_mask_override
    combiner = RidgeCombiner(
        float(evaluation_config["ridge_lambda"]),
        IndependentFactorTransformPipeline(
            TransformConfig(
                version="daily-cs-independent-availability-v1",
                neutralize=True,
                post_residual_standardize=True,
            )
        ),
    )
    weights = combiner.fit(fit_signals, fit_label, fit_mask, fit_exposures)
    fit_transformed = combiner.last_fit_result
    if fit_transformed is None:
        raise RuntimeError("combiner did not retain its fitted transform result")
    write_json(run_dir / "combiner.json", combiner.to_dict())

    trade_mask = test.target(test.common_mask)
    if trade_mask_override is not None:
        if trade_mask_override.shape != trade_mask.shape:
            raise ValueError("shared trade mask shape differs from test split")
        trade_mask &= trade_mask_override
    test_exposures = test.target(test.exposures)
    test_signals = _signals(test, expressions)
    raw_label = test.target(test.label)
    transformed_ic = combiner.pipeline.transform_ic(test_signals, raw_label, trade_mask, test_exposures)
    assert transformed_ic.label is not None
    combined_ic, ic_available = combine_available_signals(transformed_ic.signals, weights)
    ic_mask = transformed_ic.mask & ic_available
    combined_ic[~ic_mask] = np.nan
    pearson, rank = _daily_correlations(combined_ic, transformed_ic.label, ic_mask)
    transformed_portfolio = combiner.pipeline.transform_portfolio(test_signals, trade_mask, test_exposures)
    combined, portfolio_available = combine_available_signals(transformed_portfolio.signals, weights)
    portfolio_mask = transformed_portfolio.mask & portfolio_available
    combined[~portfolio_mask] = np.nan
    portfolio_diagnostics = transformed_portfolio.diagnostics
    raw_common = trade_mask & np.isfinite(raw_label)
    raw_calculator = FactorCalculator(raw_label, raw_common)
    raw_prepared = [raw_calculator.standardize(signal) for signal in test_signals]
    raw_combined, raw_available = combine_available_signals(raw_prepared, weights)
    raw_common &= raw_available
    raw_combined[~raw_common] = np.nan
    raw_pearson, raw_rank = _daily_correlations(raw_combined, raw_label, raw_common)
    pd.DataFrame({"date": test.target_dates, "raw_ic": raw_pearson, "raw_rank_ic": raw_rank, "rnic": pearson, "rank_rnic": rank}).to_parquet(test_dir / "rnic_daily.parquet", index=False)
    _write_factor_statistics(test_dir, run_dir, selected, expressions, weights, test.target_dates, test_signals, raw_label, transformed_ic.signals, transformed_ic.label, ic_mask, transformed_ic.diagnostics, evaluation_config)

    backtester = PortfolioBacktester(int(evaluation_config["rebalance_days"]), int(evaluation_config["holding_days"]))
    returns = test.target(test.daily_return)
    dollar = backtester.run(combined, returns, portfolio_mask)
    neutral_tolerances = {"net_tolerance": float(evaluation_config["net_tolerance"]), "exposure_tolerance": float(evaluation_config["exposure_tolerance"]), "gross_tolerance": float(evaluation_config["gross_tolerance"]), "weight_tolerance": float(evaluation_config["weight_tolerance"])}
    fully = backtester.run(combined, returns, portfolio_mask, test_exposures, fully_neutral=True, max_weight=float(evaluation_config["fully_neutral_max_weight"]), neutral_tolerances=neutral_tolerances)
    daily_frames = {}
    portfolio_results = {}
    audit_frames = []
    for name, result in (("dollar_neutral", dollar), ("fully_neutral", fully)):
        frame = pd.DataFrame({
            "date": test.target_dates,
            "gross_return": result.gross_returns,
            "turnover": result.turnover,
            "missing_held_returns": result.missing_held_returns,
            "missing_held_return_weight": result.missing_held_return_weight,
            "infeasible": result.infeasible,
        })
        for cost in map(float, evaluation_config["one_way_cost_bps"]):
            frame[f"net_return_{int(cost)}bps"] = result.gross_returns - cost / 10000.0 * result.turnover
        frame.to_parquet(test_dir / f"{name}_daily.parquet", index=False)
        daily_frames[name] = frame
        portfolio_results[name] = {f"{int(cost)}bps": portfolio_metrics(result, cost) for cost in map(float, evaluation_config["one_way_cost_bps"])}
        if result.audits:
            audit_frame = pd.DataFrame(result.audits)
            audit_frame.insert(0, "portfolio", name)
            audit_frames.append(audit_frame)
        if name == "fully_neutral":
            successful = [audit for audit in result.audits if audit.get("accepted")]
            portfolio_results[name]["solver_audit"] = {
                "rebalance_events": len(result.audits),
                "accepted_events": len(successful),
                "failure_rate": 1.0 - len(successful) / max(1, len(result.audits)),
                "maximum_abs_net": max((abs(float(audit.get("net", 0.0))) for audit in successful), default=float("nan")),
                "maximum_gross": max((float(audit.get("gross", 0.0)) for audit in successful), default=float("nan")),
                "maximum_weight": max((float(audit.get("max_weight", 0.0)) for audit in successful), default=float("nan")),
                "maximum_risk_exposure": max((float(audit.get("max_risk_exposure", 0.0)) for audit in successful), default=float("nan")),
            }
    if audit_frames:
        pd.concat(audit_frames, ignore_index=True).to_parquet(test_dir / "portfolio_solver_audits.parquet", index=False)
    exposures = []
    for name, result in (("dollar_neutral", dollar), ("fully_neutral", fully)):
        finite_exposure = np.isfinite(test_exposures).all(axis=2)
        held_missing = (np.abs(result.weights) > 0) & ~finite_exposure
        auditable = ~held_missing.any(axis=1)
        realized = np.einsum("tna,tn->ta", np.nan_to_num(test_exposures), result.weights)
        realized[~auditable] = np.nan
        frame = pd.DataFrame(realized, columns=test.exposure_names)
        frame.insert(1, "missing_held_exposures", held_missing.sum(axis=1))
        frame.insert(0, "portfolio", name)
        frame.insert(0, "date", test.target_dates)
        exposures.append(frame)
        portfolio_results[name]["max_realized_risk_exposure"] = _max_abs_exposure(realized)
        portfolio_results[name]["missing_held_exposure_days"] = int((~auditable).sum())
    pd.concat(exposures, ignore_index=True).to_parquet(test_dir / "exposures.parquet", index=False)
    raw_summary = series_summary(raw_pearson, bootstrap_samples=bootstrap_samples)
    rnic_summary = series_summary(pearson, bootstrap_samples=bootstrap_samples)
    raw_mean, residual_mean = float(raw_summary["mean"]), float(rnic_summary["mean"])
    retention = abs(residual_mean) / abs(raw_mean) if np.isfinite(raw_mean) and abs(raw_mean) > 1e-12 else float("nan")
    metrics = {
        "input_hash": input_hash,
        "pool_version": selected.get("pool_version"),
        "pool_size": len(expressions),
        "expressions": expressions,
        "ridge_weights": weights.tolist(),
        "transform_pipeline": combiner.pipeline.to_dict(),
        "fit_observations": int(fit_transformed.mask.sum()),
        "fit_valid_days": int(fit_transformed.mask.any(axis=1).sum()),
        "test_ic_observations": int(ic_mask.sum()),
        "test_ic_valid_days": int(ic_mask.any(axis=1).sum()),
        "test_trade_observations": int(portfolio_mask.sum()),
        "test_trade_valid_days": int(portfolio_mask.any(axis=1).sum()),
        "average_pair_correlation": _average_pair_correlation(list(fit_transformed.signals)),
        "raw_ic_diagnostic": raw_summary,
        "primary_pearson_rnic": rnic_summary,
        "raw_ic": raw_summary,
        "rnic": rnic_summary,
        "rank_rnic": series_summary(rank, bootstrap_samples=bootstrap_samples),
        "neutralization_retention": retention,
        "portfolios": portfolio_results,
        "evaluation_support_policy": "Each factor is transformed on its own daily support; unavailable factors are omitted and active weights are renormalized. A constant factor never invalidates the rest of the pool.",
        "limitations": ["Historical borrow availability is unavailable; short eligibility assumes borrowability."],
        "neutralization_diagnostics": {"ic_days": len(transformed_ic.diagnostics), "portfolio_days": len(portfolio_diagnostics)},
    }
    write_json(test_dir / "metrics.json", metrics)
    write_json(marker, {"status": "complete", "input_hash": input_hash, "metrics_hash": stable_hash(metrics)})
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    result.update({"status": "evaluated", "evaluation": metrics})
    write_json(result_path, result)
    identity = _cell_identity(run_dir)
    ledger = result.get("search", {})
    write_result_summary(
        run_dir / "result.md",
        experiment_id=str(result.get("experiment_id", run_dir.parents[2].name)),
        method=str(result.get("method", identity["method"])),
        reward=str(result.get("reward", identity["reward"])),
        seed=int(result.get("seed", identity["seed"])),
        budget=int(result.get("budget", ledger.get("limit", ledger.get("valid_unique_evaluations", 0)))),
        ledger=ledger,
        pool_version=int(selected.get("pool_version", 0)),
        train_objective=result.get("train_objective"),
        validation_objective=result.get("validation_objective"),
        expressions=expressions,
        evaluation=metrics,
    )
    # Matrix/report completion gates keep ``status=complete``; evaluation has
    # its own monotonic field so adding readable progress cannot invalidate an
    # otherwise accepted cell.
    update_progress(run_dir / "progress.json", status="complete", evaluation_status="complete")
    append_event(run_dir / "experiment.log", "evaluation_finished", primary_rnic=metrics["primary_pearson_rnic"].get("mean"), rank_rnic=metrics["rank_rnic"].get("mean"), fully_neutral_10bps_sharpe=metrics["portfolios"]["fully_neutral"]["10bps"].get("sharpe"))
    return metrics


def finalize_experiment(experiment_id: str, config: str | Path, methods: list[str] | None = None) -> dict[str, Any]:
    raw_config = load_yaml(config)
    paths = load_paths(config)
    evaluation_config = load_yaml(paths.code_root / "configs/eval/preliminary.yaml")["evaluation"]
    root = paths.runs_root / experiment_id
    append_event(root / "experiment.log", "evaluation_started", experiment_id=experiment_id)
    experiment = raw_config["experiment"]
    final_pools = _assert_experiment_frozen(config, paths, root, experiment, methods)
    scope_input_hash = stable_hash({"schema_version": 2, "experiment_id": experiment_id, "final_pools": [file_fingerprint(path) for path in final_pools], "panel": _panel_fingerprints(paths.processed_root), "evaluation_code": _evaluation_code_fingerprints()})
    scope_name = "_".join(sorted(set(methods or ()))) or "all"
    transaction_path = root / ("test_finalization.json" if scope_name == "all" else f"test_finalization_{scope_name}.json")
    summary_path = root / ("evaluation_summary.json" if scope_name == "all" else f"evaluation_summary_{scope_name}.json")
    if transaction_path.exists():
        transaction = _read_json(transaction_path)
        if transaction.get("scope_input_hash") != scope_input_hash:
            raise RuntimeError("frozen experiment inputs changed after test finalization started")
        if transaction.get("status") == "complete":
            return _read_json(summary_path)
    write_json(transaction_path, {"status": "started", "scope_input_hash": scope_input_hash})

    results: dict[str, Any] = {}
    for final_pool in final_pools:
        cell = final_pool.parent
        key = str(cell.relative_to(root))
        try:
            cell_scope_hash = stable_hash({"cell": key, "support": "cell-final-pool-complete-case-v1"})
            results[key] = {"status": "complete", "metrics": finalize_cell(cell, paths.processed_root, finalization_scope_hash=cell_scope_hash, evaluation_config=evaluation_config)}
        except Exception as exc:
            results[key] = {"status": "failed", "error": str(exc)}
    write_json(summary_path, results)
    status = "complete" if all(item["status"] == "complete" for item in results.values()) else "failed"
    write_json(transaction_path, {"status": status, "scope_input_hash": scope_input_hash, "support_policy": "each cell final pool complete-case; sample counts are descriptive", "completed_cells": sum(item["status"] == "complete" for item in results.values()), "failed_cells": sum(item["status"] == "failed" for item in results.values())})
    append_event(root / "experiment.log", "evaluation_finished", status=status, completed=sum(item["status"] == "complete" for item in results.values()), failed=sum(item["status"] == "failed" for item in results.values()))
    return results
