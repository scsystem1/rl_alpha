from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import EvaluationConfig, load_paths, load_yaml
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.calculator import FactorCalculator
from ..factors.combiner import RidgeCombiner
from ..factors.transform import (
    FIXED_UNIVERSE_TRANSFORM_VERSION,
    IndependentFactorTransformPipeline,
    TransformConfig,
    combine_fixed_signals,
    prepare_fixed_universe_inputs,
)
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import write_json
from ..utils.experiment_log import append_event, update_progress, write_result_summary
from .portfolio import PortfolioBacktester, portfolio_metrics
from .statistics import benjamini_hochberg, bootstrap_date_indices, factor_significance, series_summary


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


def cell_finalization_scope_hash(cell_key: str) -> str:
    """Stable scope shared by evaluation and report verification."""
    return stable_hash({"cell": cell_key, "support": "fixed-universe-zero-fill-psd-gram-v6"})


def cell_input_hash(
    run_dir: str | Path,
    processed_root: str | Path,
    evaluation_config: dict[str, Any],
    data_config: dict[str, Any] | None = None,
    protocol: str | None = None,
    finalization_scope_hash: str | None = None,
) -> str:
    """Fingerprint the current frozen pool and resolved evaluation inputs.

    ``evaluation_config`` must include effective defaults, exactly as supplied
    to finalization. Reports use this read-only check instead of reopening data
    or trusting a completion flag left by an earlier pool/configuration.
    """
    recent = protocol == "recent_alpha_v1"
    return stable_hash({
        "schema_version": 12,
        "final_pool": file_fingerprint(Path(run_dir) / "final_pool.json"),
        "panel": _panel_fingerprints(Path(processed_root)),
        "evaluation_code": _evaluation_code_fingerprints(),
        "evaluation_config": evaluation_config,
        "data_config": data_config,
        "protocol": protocol,
        "finalization_scope_hash": finalization_scope_hash,
        "support_policy": "fixed-universe-zero-fill-psd-gram-v6",
        "missing_return_policy": "zero-return-stale-value-v1",
        "fit_policy": "calibration_only" if recent else "train_and_validation",
        "liquidation_policy": "last_close" if recent else "none",
    })


def _verified_cached_metrics(test_dir: Path, input_hash: str) -> dict[str, Any]:
    marker_path, metrics_path = test_dir / "finalization.json", test_dir / "metrics.json"
    if not marker_path.is_file() or not metrics_path.is_file():
        raise RuntimeError(f"finalized cell artifacts are missing: {test_dir}")
    marker, metrics = _read_json(marker_path), _read_json(metrics_path)
    if marker.get("status") != "complete" or marker.get("input_hash") != input_hash:
        raise RuntimeError(f"finalized cell marker is incomplete or incompatible: {test_dir}")
    if metrics.get("input_hash") != input_hash or stable_hash(metrics) != marker.get("metrics_hash"):
        raise RuntimeError(f"finalized metrics changed after test was completed: {test_dir}")
    return metrics


def _daily_correlations(signal: np.ndarray, label: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.full(len(signal), np.nan)
    rank = np.full(len(signal), np.nan)
    for day in range(len(signal)):
        common = mask[day] & np.isfinite(signal[day]) & np.isfinite(label[day])
        if common.sum() < 3:
            continue
        left, right = signal[day, common], label[day, common]
        if np.var(right) <= 1e-24:
            continue
        if np.var(left) <= 1e-24:
            # A fixed-universe all-zero opinion is no predictive information,
            # not an opportunity to remove this date from the metric.
            pearson[day] = 0.0
            rank[day] = 0.0
            continue
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
    search_steps = int(experiment.get("search_steps", 250))
    comparability = set()
    for method, reward, seed in sorted(expected):
        directory = root / method / reward / f"seed_{seed}"
        state_path = directory / "progress.json"
        if not state_path.exists():
            raise RuntimeError(f"test opening refused: cell state missing for {(method, reward, seed)}")
        state = _read_json(state_path)
        identity = _expected_cell_identity(Path(config).resolve(), paths, method, reward, seed, search_steps)
        if state.get("status") != "complete" or int(state.get("search_steps", -1)) != search_steps or state.get("cell_identity") != identity:
            raise RuntimeError(f"test opening refused: incomplete or incompatible state for {(method, reward, seed)}")
        accepted, reason = _cell_acceptance(directory, search_steps)
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
    trade_mask: np.ndarray | None = None,
    bootstrap_indices: np.ndarray | None = None,
) -> None:
    identity = _cell_identity(run_dir)
    final_pool_id = str(selected.get("final_pool_id") or f"final_pool_{stable_hash({'cell': identity, 'pool_version': selected.get('pool_version'), 'expressions': expressions})[:20]}")
    lineage_by_expression = {str(item.get("expression")): item for item in selected.get("factors", [])}
    raw_calculator = FactorCalculator(raw_label, common_mask if trade_mask is None else trade_mask)
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
            record = factor_significance(values, hac_lag=int(evaluation_config["hac_lag"]), bootstrap_block=int(evaluation_config["bootstrap_block_length"]), bootstrap_samples=int(evaluation_config["bootstrap_samples"]), seed=int(evaluation_config["bootstrap_seed"]), dates=dates, bootstrap_indices=bootstrap_indices)
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
        "bootstrap": {"method": "year_stratified_moving_block_on_trading_grid", "block_length": evaluation_config["bootstrap_block_length"], "samples": evaluation_config["bootstrap_samples"], "seed": int(evaluation_config["bootstrap_seed"]), "shared_date_draws": True},
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
    data_config: dict[str, Any] | None = None,
    protocol: str | None = None,
) -> dict[str, Any]:
    run_dir, processed_root = Path(run_dir), Path(processed_root)
    final_pool_path = run_dir / "final_pool.json"
    selected = json.loads(final_pool_path.read_text(encoding="utf-8"))
    evaluation_config = {**EvaluationConfig().model_dump(), "bootstrap_samples": bootstrap_samples, **(evaluation_config or {})}
    recent = protocol == "recent_alpha_v1"
    if recent and (data_config is None or selected.get("selection_rule") != "fixed_budget_terminal_pool"):
        raise ValueError("recent alpha evaluation requires explicit dates and a fixed-budget terminal pool")
    bootstrap_samples = int(evaluation_config["bootstrap_samples"])
    expressions = list(selected.get("expressions", []))
    if not expressions:
        raise ValueError(f"selected pool is empty: {run_dir}")
    input_hash = cell_input_hash(run_dir, processed_root, evaluation_config, data_config, protocol, finalization_scope_hash)
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    marker = test_dir / "finalization.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("input_hash") != input_hash:
            raise RuntimeError("test finalization input changed after test was opened")
        if state.get("status") == "complete":
            return _verified_cached_metrics(test_dir, input_hash)
    write_json(marker, {"status": "started", "input_hash": input_hash})

    store = PanelStore(processed_root)
    if recent:
        assert data_config is not None
        validation = store.load_interval("validation", *data_config["validation"])
        test = store.load_interval("test", *data_config["test"])
        fit_panels = [validation]
    else:
        train, validation, test = (store.load_split(name) for name in ("train", "validation", "test"))
        fit_panels = [train, validation]
    fit_signals = [np.concatenate(parts, axis=0) for parts in zip(*(_signals(panel, expressions) for panel in fit_panels))]
    fit_mask = np.concatenate([panel.target(panel.common_mask) for panel in fit_panels])
    fit_exposures = np.concatenate([panel.target(panel.exposures) for panel in fit_panels])
    fit_label = np.concatenate([panel.target(panel.label) for panel in fit_panels])
    if fit_mask_override is not None:
        if fit_mask_override.shape != fit_mask.shape:
            raise ValueError("shared fit mask shape differs from fitting panel")
        fit_mask &= fit_mask_override
    combiner = RidgeCombiner(
        float(evaluation_config["ridge_lambda"]),
        IndependentFactorTransformPipeline(
            TransformConfig(
                version=FIXED_UNIVERSE_TRANSFORM_VERSION,
                neutralize=True,
                post_residual_standardize=True,
            )
        ),
    )
    weights = combiner.fit(fit_signals, fit_label, fit_mask, fit_exposures)
    fit_transformed = combiner.last_fit_result
    if fit_transformed is None:
        raise RuntimeError("combiner did not retain its fitted transform result")
    if not np.isfinite(weights).all() or not np.isfinite(weights).any() or not fit_transformed.metric_mask.any():
        raise ValueError("calibration has no usable fitted observations or weights")
    write_json(run_dir / "combiner.json", combiner.to_dict())

    trade_mask = test.target(test.common_mask)
    if trade_mask_override is not None:
        if trade_mask_override.shape != trade_mask.shape:
            raise ValueError("shared trade mask shape differs from test split")
        trade_mask &= trade_mask_override
    test_exposures = test.target(test.exposures)
    test_signals = _signals(test, expressions)
    raw_label = test.target(test.label)
    # Freeze the deployment composite exactly once without observing label
    # availability. Portfolio and RNIC then consume that same frozen signal.
    combined, portfolio_mask, portfolio_diagnostics = combiner.transform(
        test_signals, trade_mask, test_exposures
    )
    objective_composites, transformed_label, ic_mask, metric_diagnostics = prepare_fixed_universe_inputs(
        (combined,), raw_label, trade_mask, test_exposures, neutralize=True
    )
    combined_ic = objective_composites[0]
    ic_diagnostics = portfolio_diagnostics + metric_diagnostics
    combined_ic[~ic_mask] = np.nan
    pearson, rank = _daily_correlations(combined_ic, transformed_label, ic_mask)
    transformed_ic = combiner.pipeline.transform_ic(
        test_signals, raw_label, trade_mask, test_exposures
    )
    if transformed_ic.objective_signals is None:
        raise RuntimeError("IC transform did not produce objective factor signals")
    raw_common = trade_mask & np.isfinite(raw_label)
    raw_prepared = [
        FactorCalculator(raw_label, trade_mask & np.isfinite(signal)).standardize(signal)
        for signal in test_signals
    ]
    raw_combined, _ = combine_fixed_signals(raw_prepared, weights)
    raw_combined[~raw_common] = np.nan
    raw_pearson, raw_rank = _daily_correlations(raw_combined, raw_label, raw_common)
    pd.DataFrame({"date": test.target_dates, "raw_ic": raw_pearson, "raw_rank_ic": raw_rank, "rnic": pearson, "rank_rnic": rank}).to_parquet(test_dir / "rnic_daily.parquet", index=False)
    bootstrap_indices = bootstrap_date_indices(test.target_dates, int(evaluation_config["bootstrap_block_length"]), bootstrap_samples, int(evaluation_config["bootstrap_seed"]))
    _write_factor_statistics(
        test_dir, run_dir, selected, expressions, weights, test.target_dates,
        test_signals, raw_label, transformed_ic.objective_signals,
        transformed_label, ic_mask, ic_diagnostics, evaluation_config, trade_mask, bootstrap_indices,
    )

    backtester = PortfolioBacktester(int(evaluation_config["rebalance_days"]), int(evaluation_config["holding_days"]))
    returns = test.target(test.daily_return)
    dollar = backtester.run(combined, returns, portfolio_mask, liquidate_last_close=recent)
    backtests = {"dollar_neutral": dollar}
    if not recent:
        neutral_tolerances = {"net_tolerance": float(evaluation_config["net_tolerance"]), "exposure_tolerance": float(evaluation_config["exposure_tolerance"]), "gross_tolerance": float(evaluation_config["gross_tolerance"]), "weight_tolerance": float(evaluation_config["weight_tolerance"])}
        backtests["fully_neutral"] = backtester.run(combined, returns, portfolio_mask, test_exposures, fully_neutral=True, max_weight=float(evaluation_config["fully_neutral_max_weight"]), neutral_tolerances=neutral_tolerances)
    daily_frames = {}
    portfolio_results = {}
    audit_frames = []
    for name, result in backtests.items():
        frame = pd.DataFrame({
            "date": test.target_dates,
            "gross_return": result.gross_returns,
            "turnover": result.turnover,
            "missing_held_returns": result.missing_held_returns,
            "missing_held_return_weight": result.missing_held_return_weight,
            "infeasible": result.infeasible,
            "liquidation_turnover": result.liquidation_turnover,
            "gross_weight": np.abs(result.weights).sum(axis=1),
            "net_weight": result.weights.sum(axis=1),
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
    for name, result in backtests.items():
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
    summary_options = {"hac_lag": int(evaluation_config["hac_lag"]), "bootstrap_samples": bootstrap_samples, "seed": int(evaluation_config["bootstrap_seed"]), "bootstrap_block": int(evaluation_config["bootstrap_block_length"]), "dates": test.target_dates, "bootstrap_indices": bootstrap_indices}
    raw_summary = series_summary(raw_pearson, **summary_options)
    rnic_summary = series_summary(pearson, **summary_options)
    raw_mean, residual_mean = float(raw_summary["mean"]), float(rnic_summary["mean"])
    retention = abs(residual_mean) / abs(raw_mean) if np.isfinite(raw_mean) and abs(raw_mean) > 1e-12 else float("nan")
    metrics = {
        "input_hash": input_hash,
        "evaluation_schema_version": 12,
        "protocol": protocol,
        "data_intervals": data_config,
        "fit_period": "calibration" if recent else "train_and_validation",
        "annual_liquidation": "last_close" if recent else "none",
        "pool_version": selected.get("pool_version"),
        "pool_size": len(expressions),
        "expressions": expressions,
        "ridge_weights": weights.tolist(),
        "transform_pipeline": combiner.pipeline.to_dict(),
        "fit_observations": int(fit_transformed.metric_mask.sum()) if fit_transformed.metric_mask is not None else 0,
        "fit_valid_days": int(fit_transformed.metric_mask.any(axis=1).sum()) if fit_transformed.metric_mask is not None else 0,
        "test_ic_observations": int(ic_mask.sum()),
        "test_ic_valid_days": int(ic_mask.any(axis=1).sum()),
        "test_trade_observations": int(trade_mask.sum()),
        "test_trade_valid_days": int(trade_mask.any(axis=1).sum()),
        "test_portfolio_executable_observations": int(portfolio_mask.sum()),
        "test_portfolio_executable_valid_days": int(portfolio_mask.any(axis=1).sum()),
        "average_pair_correlation": _average_pair_correlation(list(fit_transformed.signals)),
        "moment_diagnostics": combiner.moment_diagnostics_,
        "raw_ic_diagnostic": raw_summary,
        "primary_pearson_rnic": rnic_summary,
        "raw_ic": raw_summary,
        "raw_rank_ic": series_summary(raw_rank, **summary_options),
        "rnic": rnic_summary,
        "rank_rnic": series_summary(rank, **summary_options),
        "neutralization_retention": retention,
        "portfolios": portfolio_results,
        "evaluation_support_policy": "Factors are transformed independently on a label-free trade universe; missing transformed residuals are zero opinions; one fixed weight vector is applied without asset-wise renormalization; weights use a fixed-universe PSD Gram; primary RNIC jointly projects the deployment composite and label on the fixed metric universe.",
        "evaluation_input_schema": {
            "trade_mask": "membership & eligibility & finite(exposures); label- and factor-independent",
            "metric_mask": "trade_mask & finite(transformed_label)",
            "factor_null": "zero opinion for Gram, ridge composite, and primary RNIC",
            "portfolio_executable_mask": "trade_mask & any finite factor with nonzero frozen weight",
        },
        "diagnostic_semantics": {
            "test_trade_observations": "factor-independent label-free trade universe",
            "test_ic_observations": "fixed primary RNIC metric universe",
            "test_portfolio_executable_observations": "names eligible for portfolio selection after signal availability",
        },
        "limitations": ["Historical borrow availability is unavailable; short eligibility assumes borrowability."],
        "neutralization_diagnostics": {
            "ic_days": len({item.get("date") for item in ic_diagnostics}),
            "portfolio_days": len({item.get("date") for item in portfolio_diagnostics}),
        },
    }
    if recent:
        cal_signal, cal_label, cal_mask, _ = combiner.transform_metric_composite(fit_signals, fit_label, fit_mask, fit_exposures)
        cal_rnic, cal_rank = _daily_correlations(cal_signal, cal_label, cal_mask)
        metrics["calibration_diagnostic"] = {
            "semantics": "in_sample_weight_fit; not validation or pool selection",
            "rnic_mean": float(np.nanmean(cal_rnic)) if np.isfinite(cal_rnic).any() else float("nan"),
            "rank_rnic_mean": float(np.nanmean(cal_rank)) if np.isfinite(cal_rank).any() else float("nan"),
            "valid_days": int(np.isfinite(cal_rnic).sum()),
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
        search_steps=int(result.get("search_steps", ledger.get("search_steps", ledger.get("completed_steps", 0)))),
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
    append_event(run_dir / "experiment.log", "evaluation_finished", primary_rnic=metrics["primary_pearson_rnic"].get("mean"), rank_rnic=metrics["rank_rnic"].get("mean"), dollar_neutral_10bps_sharpe=metrics["portfolios"]["dollar_neutral"].get("10bps", {}).get("sharpe"))
    return metrics


def finalize_experiment(experiment_id: str, config: str | Path, methods: list[str] | None = None) -> dict[str, Any]:
    raw_config = load_yaml(config)
    if raw_config.get("rolling"):
        from ..rolling import evaluate_rolling

        return evaluate_rolling(experiment_id, config, methods=methods)
    paths = load_paths(config)
    from ..config import resolve_data_evaluation

    data_config, evaluation_config = resolve_data_evaluation(raw_config, paths.code_root)
    root = paths.runs_root / experiment_id
    append_event(root / "experiment.log", "evaluation_started", experiment_id=experiment_id)
    experiment = raw_config["experiment"]
    final_pools = _assert_experiment_frozen(config, paths, root, experiment, methods)
    scope_input_hash = stable_hash({"schema_version": 3, "experiment_id": experiment_id, "final_pools": [file_fingerprint(path) for path in final_pools], "panel": _panel_fingerprints(paths.processed_root), "evaluation_code": _evaluation_code_fingerprints(), "data_config": data_config, "evaluation_config": evaluation_config, "protocol": raw_config.get("protocol")})
    scope_name = "_".join(sorted(set(methods or ()))) or "all"
    transaction_path = root / ("test_finalization.json" if scope_name == "all" else f"test_finalization_{scope_name}.json")
    summary_path = root / ("evaluation_summary.json" if scope_name == "all" else f"evaluation_summary_{scope_name}.json")
    if transaction_path.exists():
        transaction = _read_json(transaction_path)
        if transaction.get("scope_input_hash") != scope_input_hash:
            raise RuntimeError("frozen experiment inputs changed after test finalization started")
        if transaction.get("status") == "complete":
            summary = _read_json(summary_path)
            expected_keys = {str(path.parent.relative_to(root)) for path in final_pools}
            if set(summary) != expected_keys:
                raise RuntimeError("cached evaluation summary cells differ from the frozen experiment")
            for final_pool in final_pools:
                cell = final_pool.parent
                key = str(cell.relative_to(root))
                input_hash = cell_input_hash(cell, paths.processed_root, evaluation_config, data_config,
                                             raw_config.get("protocol"), cell_finalization_scope_hash(key))
                metrics = _verified_cached_metrics(cell / "test", input_hash)
                entry = summary[key]
                if entry.get("status") != "complete" or stable_hash(entry.get("metrics")) != stable_hash(metrics):
                    raise RuntimeError(f"cached evaluation summary differs from finalized cell metrics: {key}")
            return summary
    write_json(transaction_path, {"status": "started", "scope_input_hash": scope_input_hash})

    results: dict[str, Any] = {}
    for final_pool in final_pools:
        cell = final_pool.parent
        key = str(cell.relative_to(root))
        try:
            cell_scope_hash = cell_finalization_scope_hash(key)
            results[key] = {"status": "complete", "metrics": finalize_cell(cell, paths.processed_root, finalization_scope_hash=cell_scope_hash, evaluation_config=evaluation_config, data_config=data_config, protocol=raw_config.get("protocol"))}
        except Exception as exc:
            results[key] = {"status": "failed", "error": str(exc)}
    write_json(summary_path, results)
    status = "complete" if all(item["status"] == "complete" for item in results.values()) else "failed"
    write_json(transaction_path, {"status": status, "scope_input_hash": scope_input_hash, "support_policy": "common fixed metric universe; factor residual nulls are zero opinions; no asset-wise weight renormalization", "completed_cells": sum(item["status"] == "complete" for item in results.values()), "failed_cells": sum(item["status"] == "failed" for item in results.values())})
    append_event(root / "experiment.log", "evaluation_finished", status=status, completed=sum(item["status"] == "complete" for item in results.values()), failed=sum(item["status"] == "failed" for item in results.values()))
    return results
