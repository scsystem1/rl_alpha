from __future__ import annotations

import numpy as np
import pytest
import yaml

from rlalpha.evaluation.portfolio import PortfolioBacktester, PortfolioResult, portfolio_metrics, project_fully_neutral
from rlalpha.evaluation.finalize import _assert_experiment_frozen, _average_pair_correlation, _max_abs_exposure
from rlalpha.evaluation.statistics import benjamini_hochberg, factor_significance, paired_summary
from rlalpha.config import PathsConfig
from rlalpha.reporting.build import _paired_sharpe


def test_fully_neutral_qp_constraints() -> None:
    n = 200
    score = np.linspace(-1, 1, n)
    eligible = np.ones(n, dtype=bool)
    exposures = np.zeros((n, 22))
    exposures[:, 0] = 1.0
    target = np.zeros(n)
    target[:40] = -0.5 / 40
    target[-40:] = 0.5 / 40
    projected, audit = project_fully_neutral(target, score, exposures, eligible)
    assert projected is not None, audit
    assert abs(projected.sum()) < 1e-8
    assert abs(np.abs(projected).sum() - 1) < 1e-6
    assert np.abs(projected).max() <= 0.020001
    assert audit["max_risk_exposure"] < 1e-8


def test_fully_neutral_solution_is_rejected_when_postsolve_tolerance_fails() -> None:
    n = 200
    score = np.linspace(-1, 1, n)
    exposures = np.zeros((n, 22)); exposures[:, 0] = 1.0
    target = np.zeros(n); target[:40] = -0.5 / 40; target[-40:] = 0.5 / 40
    projected, audit = project_fully_neutral(target, score, exposures, np.ones(n, bool), net_tolerance=1e-30, gross_tolerance=1e-30, weight_tolerance=1e-30)
    assert projected is None
    assert not audit["accepted"]
    assert audit["constraint_violations"]


def test_four_sleeve_execution_delay_and_daily_returns() -> None:
    days, assets = 30, 20
    scores = np.tile(np.arange(assets), (days, 1)).astype(float)
    returns = np.zeros_like(scores)
    returns[:, -4:] = 0.01
    returns[:, :4] = -0.01
    result = PortfolioBacktester(5, 20).run(scores, returns, np.ones_like(scores, dtype=bool))
    assert np.allclose(result.weights[:2], 0)
    assert np.isclose(result.turnover[1], 0.25)
    assert np.isclose(result.turnover[2], 0.0)
    assert np.isclose(np.abs(result.weights[2]).sum(), 0.25)
    assert np.isclose(result.gross_returns[2], 0.0025)
    assert np.isclose(np.abs(result.weights[17]).sum(), 1.0)


def test_transaction_cost_and_paired_statistics() -> None:
    result = PortfolioResult(np.zeros((2, 1)), np.array([0.01, 0.0]), np.array([1.0, 0.5]), np.zeros(2, int), np.zeros(2, bool), [])
    metrics = portfolio_metrics(result, 10)
    expected = np.array([0.009, -0.0005])
    assert np.isclose(metrics["annual_return"], expected.mean() * 252)
    paired = paired_summary(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5]), bootstrap_samples=100)
    assert np.isclose(paired["mean"], 0.5)


def test_max_drawdown_includes_initial_wealth():
    empty = np.zeros((1, 1))
    first_loss = PortfolioResult(empty, np.array([-0.10]), np.zeros(1), np.zeros(1, int), np.zeros(1, bool), [])
    assert np.isclose(portfolio_metrics(first_loss, 0)["max_drawdown"], -0.10)
    path = PortfolioResult(np.zeros((3, 1)), np.array([0.10, -0.20, -0.10]), np.zeros(3), np.zeros(3, int), np.zeros(3, bool), [])
    # 1 -> 1.1 -> 0.88 -> 0.792, a 28% drawdown from the peak.
    assert np.isclose(portfolio_metrics(path, 0)["max_drawdown"], -0.28)


def test_missing_held_return_invalidates_pnl_and_performance_path():
    scores = np.tile(np.arange(20), (4, 1)).astype(float)
    returns = np.zeros_like(scores)
    returns[2, -1] = np.nan
    result = PortfolioBacktester(1, 1).run(scores, returns, np.ones_like(scores, dtype=bool))
    assert result.missing_held_returns[2] == 1
    assert np.isnan(result.gross_returns[2])
    metrics = portfolio_metrics(result, 0)
    assert metrics["invalid_return_path"]
    assert np.isnan(metrics["annual_return"])


def test_factor_pair_correlation_and_all_exposure_columns_are_reported() -> None:
    left = np.arange(12, dtype=float).reshape(3, 4)
    right = -left
    assert np.isclose(_average_pair_correlation([left, right]), 1.0)
    realized = np.zeros((3, 22))
    realized[:, 0] = 0.4
    realized[:, 1] = 0.1
    assert np.isclose(_max_abs_exposure(realized), 0.4)


def test_factor_significance_and_bh_fdr_boundaries() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0.03, 0.02, 100)
    summary = factor_significance(values, hac_lag=5, bootstrap_samples=100, seed=9)
    assert summary["status"] == "ok"
    assert summary["n_days"] == 100
    assert 0 <= summary["p_value"] <= 1
    insufficient = factor_significance(np.ones(10), min_days=30)
    assert insufficient["status"] == "insufficient_data"
    assert np.isnan(insufficient["p_value"])
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_test_opening_refuses_an_incomplete_configured_matrix(tmp_path) -> None:
    config = tmp_path / "experiment.yaml"
    experiment = {"methods": ["random"], "rewards": ["r0"], "seeds": [0], "valid_unique_budget": 8}
    config.write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")
    paths = PathsConfig(code_root=tmp_path, runs_root=tmp_path / "runs", processed_root=tmp_path / "processed")
    with pytest.raises(RuntimeError, match="missing_cells"):
        _assert_experiment_frozen(config, paths, tmp_path / "runs/incomplete", experiment)


def test_paired_sharpe_refuses_a_missing_return_path() -> None:
    delta, interval = _paired_sharpe(np.array([0.01, np.nan]), np.array([0.0, 0.01]), samples=10)
    assert np.isnan(delta)
    assert np.isnan(interval).all()
