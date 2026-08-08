from __future__ import annotations

import numpy as np

from rlalpha.evaluation.portfolio import PortfolioBacktester, PortfolioResult, portfolio_metrics, project_fully_neutral
from rlalpha.evaluation.finalize import _average_pair_correlation, _max_abs_exposure
from rlalpha.evaluation.statistics import paired_summary


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


def test_factor_pair_correlation_and_all_exposure_columns_are_reported() -> None:
    left = np.arange(12, dtype=float).reshape(3, 4)
    right = -left
    assert np.isclose(_average_pair_correlation([left, right]), 1.0)
    realized = np.zeros((3, 22))
    realized[:, 0] = 0.4
    realized[:, 1] = 0.1
    assert np.isclose(_max_abs_exposure(realized), 0.4)
