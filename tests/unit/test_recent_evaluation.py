from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

import rlalpha.evaluation.finalize as finalizer
from rlalpha.data.store import SplitPanel
from rlalpha.config import EvaluationConfig
from rlalpha.evaluation.portfolio import PortfolioBacktester, portfolio_metrics, return_metrics
from rlalpha.factors.combiner import RidgeCombiner
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import write_json


def test_last_close_liquidation_preserves_last_day_pnl_and_net_holdings():
    scores = np.tile(np.arange(20), (25, 1)).astype(float)
    scores[10:] *= -1  # sleeves offset while holdings are being replaced
    returns = np.zeros_like(scores)
    returns[-1] = np.linspace(-0.02, 0.02, 20)
    tester = PortfolioBacktester(5, 20)
    unclosed = tester.run(scores, returns, np.ones_like(scores, bool))
    closed = tester.run(scores, returns, np.ones_like(scores, bool), liquidate_last_close=True)
    np.testing.assert_allclose(closed.weights, unclosed.weights)
    np.testing.assert_allclose(closed.gross_returns, unclosed.gross_returns)
    assert closed.liquidation_turnover[-1] == pytest.approx(np.abs(unclosed.weights[-1]).sum())
    assert closed.liquidation_turnover[:-1].sum() == 0
    assert closed.turnover.sum() - unclosed.turnover.sum() == pytest.approx(closed.liquidation_turnover[-1])
    net = closed.gross_returns - 0.001 * closed.turnover
    assert portfolio_metrics(closed, 10)["total_return"] == pytest.approx(np.prod(1 + net) - 1)
    assert np.all(tester.run(scores, returns, np.ones_like(scores, bool), liquidate_last_close=True).weights[:2] == 0)


def test_return_path_metrics_and_annual_concatenation():
    first, second = np.array([-0.10, 0.2, 0.01]), np.array([0.03, -0.02, 0.04])
    overall = return_metrics(np.concatenate([first, second]))
    assert overall["total_return"] + 1 == pytest.approx((return_metrics(first)["total_return"] + 1) * (return_metrics(second)["total_return"] + 1))
    assert overall["cagr"] == pytest.approx((overall["total_return"] + 1) ** (252 / 6) - 1)
    assert overall["max_drawdown"] == pytest.approx(-0.1)
    assert overall["annual_return"] == overall["annualized_mean_return"]
    assert np.isnan(return_metrics(np.zeros(10))["sharpe"])
    assert return_metrics(np.array([-0.1]))["total_return"] == pytest.approx(-0.1)
    assert return_metrics(np.array([-0.1]))["annual_return"] == pytest.approx(-25.2)
    assert return_metrics(np.array([0.1, np.nan]))["invalid_return_path"]
    assert np.isnan(return_metrics(np.array([0.1, np.nan]))["total_return"])


def _panel(name, seed, start, days):
    rng = np.random.default_rng(seed)
    assets = 40
    dates = pd.bdate_range(start, periods=days)
    close, volume = rng.normal(size=(days, assets)), rng.normal(size=(days, assets))
    exposure = rng.normal(size=(days, assets))
    label = 0.03 * close + 0.01 * volume + 0.1 * exposure + rng.normal(0, 0.04, (days, assets))
    label[-21:] = np.nan
    exposures = np.stack([np.ones_like(exposure), exposure], axis=2)
    return SplitPanel(name, dates, np.arange(assets), {"$close": close, "$volume": volume}, rng.normal(0, 0.01, (days, assets)), label, np.ones_like(close, bool), np.ones_like(close, bool), exposures, ("intercept", "style"), slice(None), name)


def test_recent_finalization_calibration_only_four_ics_and_no_qp(tmp_path, monkeypatch):
    panels = {"validation": _panel("validation", 1, "2020-07-01", 100), "test": _panel("test", 2, "2021-01-01", 110)}
    calls = []

    class Store:
        def __init__(self, root):
            pass

        def load_interval(self, name, start, end, history=252):
            calls.append((name, start, end, history))
            assert name in {"validation", "test"}
            return panels[name]

        def load_split(self, *args, **kwargs):
            raise AssertionError("legacy split access forbidden")

    monkeypatch.setattr(finalizer, "PanelStore", Store)
    monkeypatch.setattr(finalizer, "_panel_fingerprints", lambda root: [])
    monkeypatch.setattr(finalizer, "_evaluation_code_fingerprints", lambda: [])
    monkeypatch.setattr("rlalpha.evaluation.portfolio.project_fully_neutral", lambda *args, **kwargs: pytest.fail("QP is forbidden"))
    data = {"train": ["2018-07-01", "2020-06-30"], "validation": ["2020-07-01", "2020-12-31"], "test": ["2021-01-01", "2021-12-31"]}

    def run(name):
        directory = tmp_path / name / "random" / "r1_oof" / "seed_0"
        directory.mkdir(parents=True)
        (directory / "final_pool.json").write_text(json.dumps({"expressions": ["$close", "$volume"], "pool_version": 7, "selection_rule": "fixed_budget_terminal_pool"}))
        metrics = finalizer.finalize_cell(directory, tmp_path, bootstrap_samples=40, protocol="recent_alpha_v1", data_config=data)
        return directory, metrics

    directory, metrics = run("original")
    assert [call[0] for call in calls] == ["validation", "test"]
    assert metrics["fit_period"] == "calibration"
    assert metrics["calibration_diagnostic"]["semantics"].startswith("in_sample")
    assert list(metrics["portfolios"]) == ["dollar_neutral"]
    assert not list((directory / "test").glob("fully_neutral*"))
    daily = pd.read_parquet(directory / "test/rnic_daily.parquet")
    assert daily[["raw_ic", "raw_rank_ic", "rnic", "rank_rnic"]].iloc[-21:].isna().all().all()
    for metric in ("raw_ic", "raw_rank_ic", "rnic", "rank_rnic"):
        assert metrics[metric]["mean"] == pytest.approx(daily[metric].mean())
        assert metrics[metric]["n"] == 89
    assert "p_value" in metrics["rnic"]
    assert "p_value" in metrics["rank_rnic"]

    # Direct one-day arithmetic verifies the raw weights and residual-before-rank order.
    panel = panels["test"]
    combiner = RidgeCombiner.from_dict(json.loads((directory / "combiner.json").read_text()))
    signals = [panel.features["$close"], panel.features["$volume"]]
    raw = []
    for signal in signals:
        clipped = np.clip(signal[0], *np.quantile(signal[0], [0.01, 0.99]))
        raw.append((clipped - clipped.mean()) / clipped.std())
    raw_composite = np.asarray(raw).T @ np.asarray(metrics["ridge_weights"])
    assert daily.raw_ic.iloc[0] == pytest.approx(np.corrcoef(raw_composite, panel.label[0])[0, 1])
    assert daily.raw_rank_ic.iloc[0] == pytest.approx(spearmanr(raw_composite, panel.label[0]).statistic)
    combined, _, _ = combiner.transform(signals, panel.common_mask, panel.exposures)
    x = panel.exposures[0]
    left = combined[0] - x @ np.linalg.lstsq(x, combined[0], rcond=None)[0]
    # Retain the established RNIC target winsorization before projection.
    target = np.clip(panel.label[0], *np.quantile(panel.label[0], [0.01, 0.99]))
    target = (target - target.mean()) / target.std()
    right = target - x @ np.linalg.lstsq(x, target, rcond=None)[0]
    assert daily.rnic.iloc[0] == pytest.approx(np.corrcoef(left, right)[0, 1])
    assert daily.rank_rnic.iloc[0] == pytest.approx(spearmanr(left, right).statistic)

    original_returns = pd.read_parquet(directory / "test/dollar_neutral_daily.parquet")
    assert original_returns.liquidation_turnover.iloc[-1] == pytest.approx(original_returns.gross_weight.iloc[-1])
    assert len(original_returns) == 110
    assert np.isfinite(original_returns.net_return_10bps).all()
    assert len(pd.read_parquet(directory / "test/factor_significance.parquet")) == 4

    # Changing test labels alters evaluation, never fitted weights or traded returns.
    changed = panel.label.copy()
    changed[:-21] *= -1
    panels["test"] = replace(panel, label=changed)
    changed_dir, changed_metrics = run("changed_test")
    np.testing.assert_allclose(changed_metrics["ridge_weights"], metrics["ridge_weights"])
    np.testing.assert_allclose(pd.read_parquet(changed_dir / "test/dollar_neutral_daily.parquet").net_return_10bps, original_returns.net_return_10bps)
    assert changed_metrics["rnic"]["mean"] == pytest.approx(-metrics["rnic"]["mean"])

    assert finalizer.finalize_cell(directory, tmp_path, bootstrap_samples=40, protocol="recent_alpha_v1", data_config=data)["input_hash"] == metrics["input_hash"]
    expected_hash = finalizer.cell_input_hash(directory, tmp_path, {**EvaluationConfig().model_dump(), "bootstrap_samples": 40}, data, "recent_alpha_v1")
    assert expected_hash == metrics["input_hash"]
    with pytest.raises(RuntimeError, match="input changed"):
        finalizer.finalize_cell(directory, tmp_path, bootstrap_samples=40, protocol="recent_alpha_v1", data_config={**data, "test": ["2022-01-01", "2022-12-31"]})

    metrics_path = directory / "test/metrics.json"
    original_metrics_text = metrics_path.read_text()
    tampered = json.loads(original_metrics_text)
    tampered["rnic"]["mean"] = 0.99
    metrics_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="metrics changed"):
        finalizer.finalize_cell(directory, tmp_path, bootstrap_samples=40, protocol="recent_alpha_v1", data_config=data)
    metrics_path.write_text(original_metrics_text)
    pool_path = directory / "final_pool.json"
    pool = json.loads(pool_path.read_text())
    pool["expressions"] = ["$close"]
    pool_path.write_text(json.dumps(pool))
    with pytest.raises(RuntimeError, match="input changed"):
        finalizer.finalize_cell(directory, tmp_path, bootstrap_samples=40, protocol="recent_alpha_v1", data_config=data)


def test_recent_finalization_rejects_nonterminal_or_empty_pool(tmp_path):
    path = tmp_path / "final_pool.json"
    path.write_text(json.dumps({"expressions": ["$close"]}))
    with pytest.raises(ValueError, match="terminal pool"):
        finalizer.finalize_cell(tmp_path, tmp_path, protocol="recent_alpha_v1", data_config={})
    path.write_text(json.dumps({"expressions": [], "selection_rule": "fixed_budget_terminal_pool"}))
    with pytest.raises(ValueError, match="empty"):
        finalizer.finalize_cell(tmp_path, tmp_path, protocol="recent_alpha_v1", data_config={})


@pytest.mark.parametrize("tamper", ["metrics_modified", "metrics_deleted", "marker_missing", "marker_started", "summary_changed"])
def test_cached_experiment_verifies_cell_artifacts_without_refitting(tmp_path, monkeypatch, tamper):
    experiment = {"methods": ["random"], "rewards": ["r1_oof"], "seeds": [0], "search_steps": 100}
    raw = {"experiment": experiment, "protocol": "recent_alpha_v1"}
    root = tmp_path / "runs" / "demo"
    cell = root / "random/r1_oof/seed_0"
    pool = cell / "final_pool.json"
    write_json(pool, {"expressions": ["$close"], "selection_rule": "fixed_budget_terminal_pool"})
    paths = SimpleNamespace(runs_root=tmp_path / "runs", processed_root=tmp_path / "processed", code_root=tmp_path)
    evaluation = EvaluationConfig().model_dump()
    data = {"validation": ["2020-07-01", "2020-12-31"], "test": ["2021-01-01", "2021-12-31"]}
    monkeypatch.setattr(finalizer, "load_yaml", lambda path: raw)
    monkeypatch.setattr(finalizer, "load_paths", lambda path: paths)
    monkeypatch.setattr("rlalpha.config.resolve_data_evaluation", lambda raw, root: (data, evaluation))
    monkeypatch.setattr(finalizer, "_assert_experiment_frozen", lambda *args: [pool])
    monkeypatch.setattr(finalizer, "_panel_fingerprints", lambda root: [])
    monkeypatch.setattr(finalizer, "_evaluation_code_fingerprints", lambda: [])
    fits = []

    def fit_once(run_dir, processed_root, **options):
        fits.append(run_dir)
        input_hash = finalizer.cell_input_hash(run_dir, processed_root, options["evaluation_config"], options["data_config"], options["protocol"], options["finalization_scope_hash"])
        metrics = {"input_hash": input_hash, "rnic": {"mean": 0.01}}
        write_json(run_dir / "test/metrics.json", metrics)
        write_json(run_dir / "test/finalization.json", {"status": "complete", "input_hash": input_hash, "metrics_hash": stable_hash(metrics)})
        return metrics

    monkeypatch.setattr(finalizer, "finalize_cell", fit_once)
    first = finalizer.finalize_experiment("demo", tmp_path / "config.yaml")
    assert finalizer.finalize_experiment("demo", tmp_path / "config.yaml") == first
    assert len(fits) == 1
    if tamper == "metrics_modified":
        metrics = json.loads((cell / "test/metrics.json").read_text())
        metrics["rnic"]["mean"] = 0.99
        write_json(cell / "test/metrics.json", metrics)
    elif tamper == "metrics_deleted":
        (cell / "test/metrics.json").unlink()
    elif tamper == "marker_missing":
        (cell / "test/finalization.json").unlink()
    elif tamper == "marker_started":
        marker = json.loads((cell / "test/finalization.json").read_text())
        write_json(cell / "test/finalization.json", {**marker, "status": "started"})
    else:
        summary = json.loads((root / "evaluation_summary.json").read_text())
        summary["random/r1_oof/seed_0"]["metrics"]["rnic"]["mean"] = 0.99
        write_json(root / "evaluation_summary.json", summary)
    with pytest.raises(RuntimeError, match="finalized|cached evaluation"):
        finalizer.finalize_experiment("demo", tmp_path / "config.yaml")
    assert len(fits) == 1
