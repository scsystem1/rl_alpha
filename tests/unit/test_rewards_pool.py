from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

from rlalpha.factors.combiner import RidgeCombiner
from rlalpha.factors.transform import FactorTransformPipeline, TransformConfig
from rlalpha.factors.calculator import FactorCalculator, daily_corr
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry, PoolScore
from rlalpha.rewards.r0 import R0Objective
from rlalpha.rewards.r2_lcb import R2LCBObjective
from rlalpha.rewards.statistics import lcb_score, newey_west_mean_se


def test_ridge_combiner_matches_explicit_linear_system():
    rng = np.random.default_rng(0)
    signals = [rng.normal(size=(30, 50)) for _ in range(3)]
    label = 0.3 * signals[0] - 0.1 * signals[1] + rng.normal(size=(30, 50))
    mask = np.ones_like(label, dtype=bool)
    combiner = RidgeCombiner(1e-3)
    weights = combiner.fit(signals, label, mask)
    assert weights.shape == (3,)
    assert np.isfinite(weights).all()
    score = R0Objective(label, mask).score_pool(signals)
    assert np.isclose(score.objective, np.nanmean(score.daily_ic))


def test_ridge_transform_pipeline_is_identical_after_serialization():
    rng = np.random.default_rng(44)
    days, assets = 8, 50
    exposures = np.stack([np.column_stack([np.ones(assets), np.linspace(-1, 1, assets)]) for _ in range(days)])
    signals = [rng.normal(size=(days, assets)), rng.normal(size=(days, assets))]
    label = 0.2 * signals[0] - 0.1 * signals[1] + exposures[:, :, 1] + rng.normal(size=(days, assets))
    mask = np.ones_like(label, dtype=bool)
    combiner = RidgeCombiner(1e-3, FactorTransformPipeline(TransformConfig(neutralize=True)))
    weights = combiner.fit(signals, label, mask, exposures)
    expected, expected_mask, _ = combiner.transform(signals, mask, exposures)
    restored = RidgeCombiner.from_dict(combiner.to_dict())
    actual, actual_mask, _ = restored.transform(signals, mask, exposures)
    assert np.allclose(restored.weights_, weights)
    assert np.array_equal(actual_mask, expected_mask)
    assert np.allclose(actual, expected, equal_nan=True)


def test_pool_scoring_matches_explicit_complete_case_combination():
    rng = np.random.default_rng(19)
    label = rng.normal(size=(40, 60))
    mask = rng.random(label.shape) > 0.05
    signals = [rng.normal(size=label.shape) for _ in range(4)]
    signals[1][5:10, :7] = np.nan
    objective = R0Objective(label, mask)
    score = objective.score_pool(signals)
    common = mask & np.isfinite(label)
    for signal in signals:
        common &= np.isfinite(signal)
    calculator = FactorCalculator(label, common)
    prepared = [calculator.standardize(signal) for signal in signals]
    combined = sum(weight * signal for signal, weight in zip(prepared, score.weights, strict=True))
    expected = daily_corr(combined, label, common)
    assert np.allclose(score.daily_ic, expected, equal_nan=True, atol=1e-12)


def test_neutralized_signal_and_label_use_identical_projection_sample():
    rng = np.random.default_rng(123)
    days, assets = 3, 40
    x = np.linspace(-2, 2, assets)
    exposures = np.stack([np.column_stack([np.ones(assets), x]) for _ in range(days)])
    signal = np.tile(2.0 * x, (days, 1)) + rng.normal(scale=0.01, size=(days, assets))
    label = np.tile(-3.0 * x, (days, 1)) + rng.normal(scale=0.01, size=(days, assets))
    label[:, -5:] = np.nan
    objective = R2LCBObjective(label, np.ones_like(label, dtype=bool), exposures, hac_lag=1)
    residual_signals, residual_label = objective._neutralized_inputs([signal])
    for day in range(days):
        common = np.isfinite(residual_label[day]) & np.isfinite(residual_signals[0][day])
        assert np.array_equal(common, np.arange(assets) < assets - 5)
        assert np.max(np.abs(exposures[day, common].T @ residual_signals[0][day, common])) < 1e-8
        assert np.max(np.abs(exposures[day, common].T @ residual_label[day, common])) < 1e-8


def test_zero_variance_standardized_day_is_removed_before_neutralization():
    rng = np.random.default_rng(8)
    days, assets = 3, 40
    exposures = np.stack([np.column_stack([np.ones(assets), np.linspace(-1, 1, assets)]) for _ in range(days)])
    signal = rng.normal(size=(days, assets))
    signal[1] = 5.0
    label = rng.normal(size=(days, assets))
    pipeline = FactorTransformPipeline(TransformConfig(neutralize=True))
    transformed = pipeline.fit_transform([signal], label, np.ones_like(label, bool), exposures)
    assert not transformed.mask[1].any()
    objective = R2LCBObjective(label, np.ones_like(label, bool), exposures, hac_lag=1)
    residual_signals, residual_label = objective._neutralized_inputs([signal])
    assert np.isnan(residual_signals[0][1]).all()
    assert np.isnan(residual_label[1]).all()


class _SumObjective:
    def score_pool(self, signals):
        values = [float(signal) for signal in signals]
        score = sum(values)
        return PoolScore(score, score, tuple(), tuple(values))


def test_pool_exact_replacement_and_one_admission_per_group():
    pool = PoolManager(_SumObjective(), capacity=2, min_delta=1e-5)
    pool.consider_group([PoolEntry("a", "a", 1.0), PoolEntry("b", "b", 2.0)])
    assert [entry.expr_hash for entry in pool.entries] == ["b"]
    pool.consider_group([PoolEntry("c", "c", 1.0)])
    assert len(pool.entries) == 2
    admission = pool.consider_group([PoolEntry("d", "d", 5.0), PoolEntry("e", "e", 4.0)])
    assert admission.admitted and admission.candidate_hash == "d"
    assert sorted(float(entry.signal) for entry in pool.entries) == [2.0, 5.0]


class _NonFiniteObjective:
    def score_pool(self, signals):
        objective = float("nan") if signals else 0.0
        return PoolScore(objective, objective, tuple(), tuple())


def test_non_finite_candidate_objective_gets_explicit_invalid_penalty():
    pool = PoolManager(_NonFiniteObjective())
    score = pool.score_candidates([PoolEntry("x", "x", 1.0)])[0]
    assert not score.valid
    assert score.reason == "non_finite_objective"
    assert score.shaped_reward == -1.0
    admission = pool.consider_group([PoolEntry("x", "x", 1.0)], [score])
    assert not admission.admitted
    assert pool.version == 0


def test_newey_west_lcb_is_mean_minus_standard_error_not_std():
    values = np.array([0.01, 0.02, -0.01, 0.03, 0.00] * 20)
    se = newey_west_mean_se(values, lag=20)
    objective, mean, reported = lcb_score(values, lag=20)
    assert np.isclose(reported, se)
    assert np.isclose(objective, mean - 1.645 * se)
    assert not np.isclose(objective, mean - 1.645 * values.std())


def test_newey_west_matches_statsmodels_without_small_sample_correction():
    values = np.array([0.01, 0.02, -0.01, 0.03, 0.00] * 20)
    fit = sm.OLS(values, np.ones((len(values), 1))).fit()
    reference = float(np.sqrt(cov_hac(fit, nlags=20, use_correction=False)[0, 0]))
    assert np.isclose(newey_west_mean_se(values, lag=20), reference, atol=1e-14)


def test_newey_west_clips_lag_and_bartlett_weights_together():
    import statsmodels.api as sm

    values = np.array([0.1, -0.2, 0.3, 0.4])
    reference = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False}).bse[0]
    assert np.isclose(newey_west_mean_se(values, lag=20), reference, atol=1e-14)


def test_r2_uses_neutralized_daily_ic_lcb():
    rng = np.random.default_rng(4)
    days, assets = 30, 80
    exposure = np.stack([np.column_stack([np.ones(assets), rng.normal(size=assets)]) for _ in range(days)])
    residual = rng.normal(size=(days, assets))
    label = 0.1 * residual + exposure[:, :, 1] + rng.normal(size=(days, assets))
    score = R2LCBObjective(label, np.ones_like(label, dtype=bool), exposure, hac_lag=5).score_pool([residual])
    assert np.isfinite(score.objective)
    assert np.isclose(score.objective, score.mean_ic - 1.645 * score.standard_error)
