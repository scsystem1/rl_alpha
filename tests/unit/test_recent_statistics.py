from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlalpha.evaluation.statistics import (
    bootstrap_date_indices,
    factor_significance,
    moving_block_bootstrap,
    paired_summary,
    series_summary,
)
from rlalpha.rewards.statistics import gap_aware_mean_se, newey_west_mean_se


def test_hac_keeps_missing_trading_days_and_reports_two_sided_p():
    values = np.array([0.1, 0.2, 0.3, np.nan, np.nan, -0.2, -0.1, 0.4])
    summary = series_summary(values, hac_lag=2, bootstrap_samples=50)
    assert summary["n"] == 6
    assert summary["hac_se"] == pytest.approx(gap_aware_mean_se(values, 2))
    assert summary["hac_se"] != pytest.approx(newey_west_mean_se(values, 2))
    assert 0 <= summary["p_value"] <= 1
    factor = factor_significance(values, hac_lag=2, min_days=2, bootstrap_samples=50)
    assert factor["hac_se"] == summary["hac_se"]


def test_shared_bootstrap_respects_years_and_preserves_missing_positions():
    dates = pd.bdate_range("2020-12-14", "2021-01-18")
    indices = bootstrap_date_indices(dates, block=4, samples=100, seed=12)
    assert np.array_equal(indices, bootstrap_date_indices(dates, block=4, samples=100, seed=12))
    assert np.all(dates.year.to_numpy()[indices] == dates.year.to_numpy()[None, :])
    values = np.arange(len(dates), dtype=float)
    values[4:10] = np.nan
    draw = values[indices]
    annual_draws = []
    for year in dates.year.unique():
        mask = dates.year == year
        annual_draws.append(np.nanmean(draw[:, mask], axis=1) * np.isfinite(values[mask]).sum() / np.isfinite(values).sum())
    expected = np.nanquantile(np.sum(annual_draws, axis=0), [0.025, 0.975])
    assert np.allclose(moving_block_bootstrap(values, 4, 100, 12, dates=dates, bootstrap_indices=indices), expected)
    shifted = moving_block_bootstrap(values + 2, 4, 100, 12, dates=dates, bootstrap_indices=indices)
    assert np.allclose(shifted, expected + 2)
    pair = paired_summary(values + 2, values, dates=dates, bootstrap_block=4, bootstrap_samples=100, bootstrap_indices=indices)
    assert pair["mean"] == 2
    assert pair["bootstrap_95_ci"] == [2, 2]


def test_year_bootstrap_weights_use_original_effective_days():
    dates = pd.to_datetime(["2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"])
    values = np.array([1, 1, np.nan, np.nan, 3, 3, 3, 3], dtype=float)
    # The draws retain different counts in the first year, but the mean of
    # that year remains 1 and its original effective-day weight remains 2/6.
    indices = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 2, 4, 4, 5, 6]])
    interval = moving_block_bootstrap(values, samples=2, dates=dates, bootstrap_indices=indices)
    assert interval == pytest.approx((7 / 3, 7 / 3))


def test_method_mean_does_not_multiply_date_count_by_seed_count():
    dates = pd.bdate_range("2021-01-01", periods=80)
    values = np.random.default_rng(20).normal(0.01, 0.02, len(dates))
    values[-21:] = np.nan
    replicated_seed_mean = np.stack([values, values, values], axis=1).mean(axis=1)
    options = dict(dates=dates, bootstrap_samples=50)
    replicated, original = series_summary(replicated_seed_mean, **options), series_summary(values, **options)
    for key in original:
        assert replicated[key] == pytest.approx(original[key])
    assert series_summary(values, **options)["n"] == 59


def test_bootstrap_grid_validation_and_empty_scores():
    with pytest.raises(ValueError, match="unique and increasing"):
        bootstrap_date_indices(pd.to_datetime(["2021-01-01", "2021-01-01"]))
    with pytest.raises(ValueError, match="matching trading dates"):
        series_summary(np.ones(2), dates=pd.bdate_range("2021-01-01", periods=3))
    summary = series_summary(np.full(5, np.nan), bootstrap_samples=20)
    assert summary["n"] == 0
    assert np.isnan(summary["hac_se"])
    assert np.isnan(summary["p_value"])
