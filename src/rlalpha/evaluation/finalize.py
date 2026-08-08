from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import load_paths
from ..data.store import PanelStore, SplitPanel
from ..dsl.parser import parse_expression
from ..factors.calculator import FactorCalculator
from ..factors.combiner import RidgeCombiner
from ..risk.neutralize import RiskNeutralizer
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import write_json
from .portfolio import PortfolioBacktester, portfolio_metrics
from .statistics import series_summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_code_fingerprints() -> list[dict[str, Any]]:
    package = Path(__file__).parents[1]
    relatives = (
        "data/store.py", "dsl/evaluator.py", "dsl/parser.py",
        "factors/calculator.py", "factors/combiner.py", "risk/neutralize.py",
        "evaluation/portfolio.py", "evaluation/statistics.py", "evaluation/finalize.py",
    )
    return [file_fingerprint(package / relative) for relative in relatives]


def _panel_fingerprints(processed_root: Path) -> list[dict[str, Any]]:
    panel = processed_root / "panel"
    return [file_fingerprint(path) for path in (panel / "index.json", panel / "build_manifest.yaml", panel / "risk_build_manifest.yaml")]


def _standardized_residual_signals(signals: list[np.ndarray], mask: np.ndarray, exposures: np.ndarray) -> list[np.ndarray]:
    dummy = np.zeros(mask.shape)
    standardized = [FactorCalculator(dummy, mask).standardize(signal) for signal in signals]
    outputs = [np.full(mask.shape, np.nan) for _ in signals]
    neutralizer = RiskNeutralizer()
    for day in range(mask.shape[0]):
        values = np.column_stack([signal[day] for signal in standardized])
        common = mask[day] & np.isfinite(exposures[day]).all(axis=1) & np.isfinite(values).all(axis=1)
        if common.sum() <= exposures.shape[2]:
            continue
        residuals, _ = neutralizer.residualize_matrix(day, values, exposures[day], common)
        for index in range(len(signals)):
            outputs[index][day] = residuals[:, index]
    return outputs


def _residual_label(label: np.ndarray, mask: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    output = np.full_like(label, np.nan, dtype=float)
    neutralizer = RiskNeutralizer()
    for day in range(len(label)):
        common = mask[day] & np.isfinite(label[day]) & np.isfinite(exposures[day]).all(axis=1)
        if common.sum() <= exposures.shape[2]:
            continue
        residual, _ = neutralizer.residualize_vector(day, label[day], exposures[day], common)
        output[day] = residual
    return output


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


def finalize_cell(run_dir: str | Path, processed_root: str | Path, bootstrap_samples: int = 2000, trade_mask_override: np.ndarray | None = None, finalization_scope_hash: str | None = None) -> dict[str, Any]:
    run_dir, processed_root = Path(run_dir), Path(processed_root)
    final_pool_path = run_dir / "final_pool.json"
    selected = json.loads(final_pool_path.read_text(encoding="utf-8"))
    expressions = list(selected.get("expressions", []))
    if not expressions:
        raise ValueError(f"selected pool is empty: {run_dir}")
    input_hash = stable_hash({"schema_version": 6, "final_pool": file_fingerprint(final_pool_path), "panel": _panel_fingerprints(processed_root), "evaluation_code": _evaluation_code_fingerprints(), "bootstrap_samples": bootstrap_samples, "finalization_scope_hash": finalization_scope_hash})
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
    fit_residual_signals = _standardized_residual_signals(fit_signals, fit_mask, fit_exposures)
    fit_residual_label = _residual_label(fit_label, fit_mask, fit_exposures)
    fit_ic_mask = fit_mask & np.isfinite(fit_residual_label)
    weights = RidgeCombiner(1e-3).fit(fit_residual_signals, fit_residual_label, fit_ic_mask)

    trade_mask = test.target(test.common_mask)
    if trade_mask_override is not None:
        if trade_mask_override.shape != trade_mask.shape:
            raise ValueError("shared trade mask shape differs from test split")
        trade_mask &= trade_mask_override
    test_exposures = test.target(test.exposures)
    test_signals = _signals(test, expressions)
    residual_signals = _standardized_residual_signals(test_signals, trade_mask, test_exposures)
    stacked = np.stack(residual_signals, axis=-1)
    combined = np.nansum(stacked * weights[None, None, :], axis=-1)
    combined[~np.isfinite(stacked).any(axis=-1)] = np.nan
    residual_label = _residual_label(test.target(test.label), trade_mask, test_exposures)
    ic_mask = trade_mask & np.isfinite(residual_label)
    pearson, rank = _daily_correlations(combined, residual_label, ic_mask)
    raw_label = test.target(test.label)
    raw_calculator = FactorCalculator(raw_label, trade_mask)
    raw_prepared = [raw_calculator.standardize(signal) for signal in test_signals]
    raw_stacked = np.stack(raw_prepared, axis=-1)
    raw_combined = np.nansum(raw_stacked * weights[None, None, :], axis=-1)
    raw_combined[~np.isfinite(raw_stacked).any(axis=-1)] = np.nan
    raw_pearson, raw_rank = _daily_correlations(raw_combined, raw_label, trade_mask & np.isfinite(raw_label))
    pd.DataFrame({"date": test.target_dates, "raw_ic": raw_pearson, "raw_rank_ic": raw_rank, "rnic": pearson, "rank_rnic": rank}).to_parquet(test_dir / "rnic_daily.parquet", index=False)

    backtester = PortfolioBacktester(5, 20)
    returns = test.target(test.daily_return)
    dollar = backtester.run(combined, returns, trade_mask)
    fully = backtester.run(combined, returns, trade_mask, test_exposures, fully_neutral=True, max_weight=0.02)
    daily_frames = {}
    portfolio_results = {}
    for name, result in (("dollar_neutral", dollar), ("fully_neutral", fully)):
        frame = pd.DataFrame({"date": test.target_dates, "gross_return": result.gross_returns, "turnover": result.turnover, "missing_held_returns": result.missing_held_returns, "infeasible": result.infeasible})
        for cost in (0.0, 10.0):
            frame[f"net_return_{int(cost)}bps"] = result.gross_returns - cost / 10000.0 * result.turnover
        frame.to_parquet(test_dir / f"{name}_daily.parquet", index=False)
        daily_frames[name] = frame
        portfolio_results[name] = {f"{int(cost)}bps": portfolio_metrics(result, cost) for cost in (0.0, 10.0)}
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
        "average_pair_correlation": _average_pair_correlation(fit_residual_signals),
        "raw_ic": raw_summary,
        "rnic": rnic_summary,
        "rank_rnic": series_summary(rank, bootstrap_samples=bootstrap_samples),
        "neutralization_retention": retention,
        "portfolios": portfolio_results,
        "limitations": ["Historical borrow availability is unavailable; short eligibility assumes borrowability."],
    }
    write_json(test_dir / "metrics.json", metrics)
    write_json(marker, {"status": "complete", "input_hash": input_hash, "metrics_hash": stable_hash(metrics)})
    return metrics


def finalize_experiment(experiment_id: str, config: str | Path) -> dict[str, Any]:
    paths = load_paths(config)
    root = paths.runs_root / experiment_id
    final_pools = sorted(root.glob("*/*/seed_*/final_pool.json"))
    if not final_pools:
        raise ValueError(f"experiment has no frozen pools: {root}")
    scope_input_hash = stable_hash({"schema_version": 2, "experiment_id": experiment_id, "final_pools": [file_fingerprint(path) for path in final_pools], "panel": _panel_fingerprints(paths.processed_root), "evaluation_code": _evaluation_code_fingerprints()})
    transaction_path = root / "test_finalization.json"
    if transaction_path.exists():
        transaction = _read_json(transaction_path)
        if transaction.get("scope_input_hash") != scope_input_hash:
            raise RuntimeError("frozen experiment inputs changed after test finalization started")
        if transaction.get("status") == "complete":
            return _read_json(root / "evaluation_summary.json")
    write_json(transaction_path, {"status": "started", "scope_input_hash": scope_input_hash})

    test_panel = PanelStore(paths.processed_root).load_split("test")
    shared_trade_mask = test_panel.target(test_panel.common_mask).copy()
    expressions = sorted({expression for path in final_pools for expression in _read_json(path).get("expressions", [])})
    for expression in expressions:
        shared_trade_mask &= np.isfinite(test_panel.evaluate(parse_expression(expression)))
    universe_hash = hashlib.sha256(np.packbits(shared_trade_mask).tobytes()).hexdigest()
    finalization_scope_hash = stable_hash({"scope_input_hash": scope_input_hash, "universe_hash": universe_hash})
    write_json(root / "test_universe.json", {"scope_input_hash": scope_input_hash, "universe_hash": universe_hash, "formula_count": len(expressions), "dates": len(shared_trade_mask), "assets": shared_trade_mask.shape[1], "eligible_observations": int(shared_trade_mask.sum()), "eligible_by_date": shared_trade_mask.sum(axis=1).tolist()})

    results: dict[str, Any] = {}
    for final_pool in final_pools:
        cell = final_pool.parent
        key = str(cell.relative_to(root))
        try:
            results[key] = {"status": "complete", "metrics": finalize_cell(cell, paths.processed_root, trade_mask_override=shared_trade_mask, finalization_scope_hash=finalization_scope_hash)}
        except Exception as exc:
            results[key] = {"status": "failed", "error": str(exc)}
    write_json(root / "evaluation_summary.json", results)
    status = "complete" if all(item["status"] == "complete" for item in results.values()) else "failed"
    write_json(transaction_path, {"status": status, "scope_input_hash": scope_input_hash, "finalization_scope_hash": finalization_scope_hash, "completed_cells": sum(item["status"] == "complete" for item in results.values()), "failed_cells": sum(item["status"] == "failed" for item in results.values())})
    return results
