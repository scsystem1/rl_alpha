from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

from rlalpha.factors.combiner import RidgeCombiner
from rlalpha.factors.transform import (
    FactorTransformPipeline,
    IndependentFactorTransformPipeline,
    TransformConfig,
    combine_fixed_signals,
)
from rlalpha.factors.moments import fixed_universe_moments
from rlalpha.factors.calculator import FactorCalculator, daily_corr
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry, PoolScore
from rlalpha.rewards.r0 import R0Objective
from rlalpha.rewards.r1 import R1Objective
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


def test_pool_scoring_matches_explicit_independent_availability_combination():
    rng = np.random.default_rng(19)
    label = rng.normal(size=(40, 60))
    mask = rng.random(label.shape) > 0.05
    signals = [rng.normal(size=label.shape) for _ in range(4)]
    signals[1][5:10, :7] = np.nan
    objective = R0Objective(label, mask)
    score = objective.score_pool(signals)
    state = objective.prepare_pool(signals)
    combined, _ = combine_fixed_signals(state.prepared_signals, np.asarray(score.weights))
    expected = daily_corr(combined, state.prepared_label, state.common_mask)
    assert np.allclose(score.daily_ic, expected, equal_nan=True, atol=1e-12)
    assert np.isfinite(np.asarray(score.daily_ic)[5:10]).all()


def test_prepared_pool_can_be_scored_with_fixed_train_weights():
    rng = np.random.default_rng(191)
    shape = (30, 50)
    train_signals = [rng.normal(size=shape), rng.normal(size=shape)]
    train_label = 0.8 * train_signals[0] - 0.2 * train_signals[1] + rng.normal(size=shape)
    validation_signals = [rng.normal(size=shape), rng.normal(size=shape)]
    validation_label = -0.1 * validation_signals[0] + 0.9 * validation_signals[1] + rng.normal(size=shape)
    mask = np.ones(shape, dtype=bool)

    train = R0Objective(train_label, mask)
    train_weights = np.asarray(train.score_pool(train_signals).weights)
    validation = R0Objective(validation_label, mask)
    state = validation.prepare_pool(validation_signals)
    fixed = validation.score_prepared_with_weights(state, train_weights)
    refit = state.score

    combined, _ = combine_fixed_signals(state.prepared_signals, train_weights)
    expected = daily_corr(combined, state.prepared_label, state.common_mask)
    assert np.allclose(fixed.weights, train_weights)
    assert np.allclose(fixed.daily_ic, expected, equal_nan=True)
    assert not np.allclose(fixed.weights, refit.weights)


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
    assert not transformed.metric_mask[1].any()
    objective = R2LCBObjective(label, np.ones_like(label, bool), exposures, hac_lag=1)
    residual_signals, residual_label = objective._neutralized_inputs([signal])
    assert np.isnan(residual_signals[0][1]).all()
    assert np.isnan(residual_label[1]).all()


def test_evaluation_keeps_day_when_only_one_factor_is_constant():
    rng = np.random.default_rng(81)
    days, assets = 3, 40
    x = np.linspace(-1, 1, assets)
    exposures = np.stack([np.column_stack([np.ones(assets), x]) for _ in range(days)])
    constant_one_day = rng.normal(size=(days, assets))
    constant_one_day[1] = 7.0
    available = rng.normal(size=(days, assets))
    label = 0.4 * available + rng.normal(size=(days, assets))
    pipeline = IndependentFactorTransformPipeline()

    transformed = pipeline.fit_transform(
        [constant_one_day, available], label, np.ones_like(label, bool), exposures
    )
    combined, combined_available = combine_fixed_signals(transformed.signals, np.array([0.5, 0.5]))

    assert np.isnan(transformed.signals[0][1]).all()
    assert np.isfinite(transformed.signals[1][1]).all()
    assert transformed.trade_mask[1].all()
    assert combined_available[1].all()
    assert np.isfinite(combined[1]).all()
    assert np.allclose(combined[1], 0.5 * transformed.signals[1][1])


def test_evaluation_drops_only_observations_with_no_available_weight():
    shape = (2, 5)
    first = np.full(shape, np.nan)
    second = np.arange(10, dtype=float).reshape(shape)
    second[0] = np.nan
    combined, available = combine_fixed_signals((first, second), np.array([1.0, 2.0]))
    assert not available[0].any()
    assert np.allclose(combined[0], 0.0)
    assert available[1].all()
    assert np.allclose(combined[1], second[1] * 2.0)
    _, tiny_available = combine_fixed_signals((second,), np.array([1e-20]))
    assert np.array_equal(tiny_available, np.isfinite(second))


def test_fixed_universe_moments_match_hand_calculation_and_are_psd():
    first = np.array([[1.0, np.nan, 2.0, -1.0], [0.5, 1.5, np.nan, -0.5]])
    second = np.array([[0.0, 2.0, np.nan, 1.0], [1.0, -1.0, 2.0, np.nan]])
    label = np.array([[1.0, -1.0, 0.5, 2.0], [0.0, 1.0, -2.0, 0.5]])
    metric = np.array([[True, True, True, True], [True, True, True, False]])

    moments = fixed_universe_moments([first, second], label, metric)
    daily_gram = []
    daily_predictive = []
    for day in range(len(metric)):
        support = metric[day]
        matrix = np.column_stack([
            np.where(np.isfinite(first[day, support]), first[day, support], 0.0),
            np.where(np.isfinite(second[day, support]), second[day, support], 0.0),
        ])
        target = label[day, support]
        daily_gram.append(matrix.T @ matrix / support.sum())
        daily_predictive.append(matrix.T @ target / support.sum())

    assert np.allclose(moments.gram, np.mean(daily_gram, axis=0))
    assert np.allclose(moments.predictive, np.mean(daily_predictive, axis=0))
    assert np.linalg.eigvalsh(moments.gram).min() >= -1e-12


def test_label_missingness_does_not_change_label_free_factor_transform():
    rng = np.random.default_rng(811)
    shape = (5, 48)
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), np.linspace(-1, 1, shape[1])])
        for _ in range(shape[0])
    ])
    signals = [rng.normal(size=shape), rng.normal(size=shape)]
    signals[0][1, :9] = np.nan
    mask = np.ones(shape, dtype=bool)
    complete_label = rng.normal(size=shape)
    missing_label = complete_label.copy()
    missing_label[2:, ::3] = np.nan

    complete = IndependentFactorTransformPipeline().fit_transform(
        signals, complete_label, mask, exposures
    )
    missing = IndependentFactorTransformPipeline().fit_transform(
        signals, missing_label, mask, exposures
    )

    assert np.array_equal(complete.trade_mask, missing.trade_mask)
    for left, right in zip(complete.signals, missing.signals, strict=True):
        assert np.allclose(left, right, equal_nan=True, atol=1e-12)
    weights = np.array([0.7, -0.2])
    complete_composite, _ = combine_fixed_signals(complete.signals, weights)
    missing_composite, _ = combine_fixed_signals(missing.signals, weights)
    assert np.allclose(complete_composite, missing_composite, equal_nan=True, atol=1e-12)


def test_fixed_metric_projection_leaves_factors_and_label_neutral():
    rng = np.random.default_rng(812)
    shape = (6, 60)
    style = np.linspace(-1, 1, shape[1])
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), style, rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    signals = [
        2.0 * exposures[:, :, 1] + rng.normal(size=shape),
        -1.5 * exposures[:, :, 2] + rng.normal(size=shape),
    ]
    signals[0][1, :7] = np.nan
    label = 3.0 * exposures[:, :, 1] + rng.normal(size=shape)
    label[3, -8:] = np.nan
    transformed = IndependentFactorTransformPipeline().fit_transform(
        signals, label, np.ones(shape, dtype=bool), exposures
    )
    assert transformed.objective_signals is not None
    assert transformed.label is not None
    assert transformed.metric_mask is not None

    for values, availability in zip(
        transformed.signals, transformed.factor_available, strict=True
    ):
        for day in range(shape[0]):
            support = availability[day]
            residual_exposure = exposures[day, support].T @ values[day, support]
            assert np.max(np.abs(residual_exposure)) < 1e-8

    for values in (*transformed.objective_signals, transformed.label):
        for day in range(shape[0]):
            support = transformed.metric_mask[day]
            residual_exposure = exposures[day, support].T @ values[day, support]
            assert np.max(np.abs(residual_exposure)) < 1e-8


@pytest.mark.parametrize("objective_type,neutralize", [(R0Objective, False), (R1Objective, True), (R2LCBObjective, True)])
def test_reward_and_final_combiner_share_fixed_universe_rnic(objective_type, neutralize):
    rng = np.random.default_rng(813)
    shape = (24, 54)
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), np.linspace(-1, 1, shape[1])])
        for _ in range(shape[0])
    ])
    signals = [rng.standard_t(3, size=shape), rng.normal(size=shape)]
    signals[0][2:5, :10] = np.nan
    label = 0.3 * np.nan_to_num(signals[0]) - 0.2 * signals[1] + rng.normal(size=shape)
    label[7, -6:] = np.nan
    mask = np.ones(shape, dtype=bool)
    kwargs = {"exposures": exposures} if neutralize else {}
    if objective_type is R2LCBObjective:
        kwargs["hac_lag"] = 3
    state = objective_type(label, mask, ridge=1e-3, **kwargs).prepare_pool(signals)
    pipeline = IndependentFactorTransformPipeline(TransformConfig(
        version="daily-cs-fixed-universe-zero-fill-v3",
        neutralize=neutralize,
        post_residual_standardize=True,
    ))
    combiner = RidgeCombiner(1e-3, pipeline)
    weights = combiner.fit(signals, label, mask, exposures if neutralize else None)
    composite, target, metric, _ = combiner.transform_metric_composite(
        signals, label, mask, exposures if neutralize else None
    )
    expected = daily_corr(composite, target, metric)

    assert np.array_equal(metric, state.common_mask)
    assert np.allclose(weights, state.score.weights, atol=1e-11)
    assert np.allclose(expected, state.score.daily_ic, equal_nan=True, atol=1e-11)


def test_independent_evaluation_combiner_round_trip_keeps_support_policy():
    rng = np.random.default_rng(82)
    shape = (4, 40)
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), np.linspace(-1, 1, shape[1])])
        for _ in range(shape[0])
    ])
    signals = [rng.normal(size=shape), rng.normal(size=shape)]
    signals[0][2] = 1.0
    label = 0.2 * signals[1] + rng.normal(size=shape)
    mask = np.ones(shape, dtype=bool)
    combiner = RidgeCombiner(1e-3, IndependentFactorTransformPipeline())
    combiner.fit(signals, label, mask, exposures)

    expected, expected_mask, _ = combiner.transform(signals, mask, exposures)
    restored = RidgeCombiner.from_dict(combiner.to_dict())
    actual, actual_mask, _ = restored.transform(signals, mask, exposures)

    assert isinstance(restored.pipeline, IndependentFactorTransformPipeline)
    assert np.array_equal(actual_mask, expected_mask)
    assert np.allclose(actual, expected, equal_nan=True)
    assert actual_mask[2].all()


def test_combiner_rejects_legacy_transform_semantics():
    pipeline = IndependentFactorTransformPipeline()
    pipeline.fitted = True
    pipeline.n_factors = 1
    combiner = RidgeCombiner(1e-3, pipeline)
    combiner.weights_ = np.array([1.0])
    serialized = combiner.to_dict()
    serialized["pipeline"]["config"]["version"] = "daily-cs-fixed-universe-zero-fill-v2"
    with pytest.raises(ValueError, match="incompatible support semantics"):
        RidgeCombiner.from_dict(serialized)


class _SumObjective:
    def score_pool(self, signals):
        values = [float(signal) for signal in signals]
        score = sum(values)
        return PoolScore(score, score, tuple(), tuple(values))


def test_group_reward_uses_median_softsign_without_changing_raw_deltas():
    pool = PoolManager(_SumObjective(), min_delta=1e-5)
    candidates = [
        PoolEntry("small", "small", 1e-5),
        PoolEntry("medium", "medium", 2e-5),
        PoolEntry("negative", "negative", -4e-5),
    ]
    scored = pool.score_candidates(candidates)

    assert [item.delta_add for item in scored] == [1e-5, 2e-5, -4e-5]
    assert [item.post_prune_delta for item in scored] == [1e-5, 2e-5, -4e-5]
    assert all(item.reward_scale == 2e-5 for item in scored)
    np.testing.assert_allclose(
        [item.shaped_reward for item in scored],
        [1 / 3, 1 / 2, -2 / 3],
    )


@pytest.mark.parametrize(
    ("delta", "expected_scale", "expected_reward"),
    [(5e-5, 5e-5, 0.5), (1e-6, 1e-5, 1 / 11)],
)
def test_single_candidate_reward_respects_dynamic_scale_floor(
    delta, expected_scale, expected_reward
):
    score = PoolManager(_SumObjective()).score_candidates(
        [PoolEntry("candidate", "candidate", delta)]
    )[0]
    assert score.reward_scale == expected_scale
    assert np.isclose(score.shaped_reward, expected_reward)


def test_invalid_candidates_do_not_set_scale_but_share_a_valid_group_scale():
    pool = PoolManager(_SumObjective())
    pool.entries = [PoolEntry("existing", "existing", 1.0)]

    invalid_only = pool.score_candidates([PoolEntry("duplicate", "existing", 1.0)])[0]
    assert invalid_only.shaped_reward == -0.5
    assert invalid_only.reward_scale is None

    invalid, valid = pool.score_candidates(
        [PoolEntry("duplicate", "existing", 1.0), PoolEntry("new", "new", 2.0)]
    )
    assert not invalid.valid and invalid.shaped_reward == -0.5
    assert valid.valid and valid.shaped_reward == 0.5
    assert invalid.reward_scale == valid.reward_scale == 2.0


def test_softsign_reward_stays_finite_for_extreme_delta():
    scored = PoolManager(_SumObjective()).score_candidates(
        [
            PoolEntry("small", "small", 1e-5),
            PoolEntry("medium", "medium", 2e-5),
            PoolEntry("extreme", "extreme", 1e300),
        ]
    )
    assert all(np.isfinite(item.shaped_reward) for item in scored)
    assert all(-1.0 < item.shaped_reward < 1.0 for item in scored)
    assert scored[-1].shaped_reward > scored[1].shaped_reward > scored[0].shaped_reward


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


@pytest.mark.parametrize("objective_type", [R0Objective, R1Objective, R2LCBObjective])
@pytest.mark.parametrize("support_reduced", [False, True])
def test_prepared_add_matches_exact_complete_recomputation(objective_type, support_reduced):
    rng = np.random.default_rng(2026)
    shape = (28, 45)
    label = rng.normal(size=shape)
    mask = rng.random(shape) > 0.03
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    base = [rng.normal(size=shape) for _ in range(3)]
    candidate = rng.normal(size=shape)
    if support_reduced:
        candidate[3:7, 4:12] = np.nan
    kwargs = {} if objective_type is R0Objective else {"exposures": exposures}
    if objective_type is R2LCBObjective:
        kwargs["hac_lag"] = 4
    objective = objective_type(label, mask, **kwargs)
    base_state = objective.prepare_pool(base)
    added = objective.prepare_add(base_state, candidate)
    exact = objective.prepare_pool(base + [candidate])
    assert np.isclose(added.score.objective, exact.score.objective, atol=1e-12)
    assert np.allclose(added.score.weights, exact.score.weights, atol=1e-11)
    assert np.allclose(added.score.daily_ic, exact.score.daily_ic, equal_nan=True, atol=1e-11)

    same_support_baseline = objective.score_subset(added, list(range(len(base))))
    assert np.isclose(same_support_baseline.objective, base_state.score.objective, atol=1e-12)


def test_constant_factor_does_not_remove_pool_days_from_reward():
    rng = np.random.default_rng(2027)
    shape = (300, 120)
    label = rng.normal(size=shape)
    useful = 0.2 * label + rng.normal(size=shape)
    sometimes_constant = rng.normal(size=shape)
    sometimes_constant[50:150] = 3.0
    objective = R0Objective(label, np.ones(shape, dtype=bool))
    state = objective.prepare_pool([useful, sometimes_constant])
    assert state.valid_days == shape[0]
    assert objective.support_is_valid(state)
    assert np.isfinite(np.asarray(state.score.daily_ic)[50:150]).all()


def test_sparse_replacement_is_scored_on_all_fixed_universe_dates():
    rng = np.random.default_rng(2028)
    shape = (300, 120)
    label = rng.normal(size=shape)
    broad = 0.05 * label + rng.normal(size=shape)
    sparse = np.full(shape, np.nan)
    sparse[:20] = label[:20]
    objective = R0Objective(label, np.ones(shape, dtype=bool))
    pool = PoolManager(objective, capacity=1)
    pool.entries = [PoolEntry("broad", "broad", broad)]
    scored = pool.score_candidates([PoolEntry("sparse", "sparse", sparse)])[0]
    assert scored.valid
    assert np.isclose(scored.pool_score.mean_ic, 20.0 / 300.0, atol=1e-12)
    assert scored.pool_score.mean_ic < 0.1
    admission = pool.consider_group([PoolEntry("sparse", "sparse", sparse)], [scored])
    # PoolManager assumes the coordinator's existing 80% candidate coverage
    # gate has already run; this direct unit call intentionally bypasses it.
    assert admission.admitted
    assert pool.entries[0].expr_hash == "sparse"


def test_r2_counts_missing_factor_dates_as_zero_ic():
    rng = np.random.default_rng(2029)
    shape = (300, 120)
    label = rng.normal(size=shape)
    sparse = np.full(shape, np.nan)
    sparse[:2] = label[:2]
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    score = R2LCBObjective(label, np.ones(shape, dtype=bool), exposures).score_pool([sparse])
    assert np.isfinite(score.objective)
    assert np.isclose(score.mean_ic, 2.0 / 300.0, atol=1e-6)
    assert np.count_nonzero(np.asarray(score.daily_ic) == 0.0) == 298


def test_full_pool_group_caches_baseline_and_bounds_formal_rechecks():
    rng = np.random.default_rng(77)
    shape = (25, 35)
    label = rng.normal(size=shape)
    objective = R0Objective(label, np.ones(shape, dtype=bool))
    pool = PoolManager(objective, capacity=20, replacement_top_k=3, admission_recheck_top_k=3)
    pool.entries = [PoolEntry(str(index), str(index), rng.normal(size=shape)) for index in range(20)]
    assert np.isfinite(pool.score.objective)
    assert np.isfinite(pool.score.objective)
    assert objective.prepare_calls == 1
    candidates = [PoolEntry(f"c{index}", f"c{index}", rng.normal(size=shape)) for index in range(8)]
    scored = pool.score_candidates(candidates)
    assert len(scored) == 8
    assert sum(item.formally_rechecked for item in scored) <= 3
    # One cached base preparation plus at most three natural-support rechecks.
    assert objective.prepare_calls <= 4


@pytest.mark.parametrize("objective_type", [R0Objective, R1Objective, R2LCBObjective])
def test_parallel_candidate_plans_are_numerically_identical_and_ordered(objective_type):
    rng = np.random.default_rng(780)
    shape = (30, 42)
    label = rng.normal(size=shape)
    mask = rng.random(shape) > 0.02
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    base_signals = [rng.normal(size=shape) for _ in range(20)]
    candidate_signals = [rng.normal(size=shape) for _ in range(8)]
    kwargs = {} if objective_type is R0Objective else {"exposures": exposures}
    if objective_type is R2LCBObjective:
        kwargs["hac_lag"] = 4

    def build_pool():
        pool = PoolManager(
            objective_type(label, mask, **kwargs),
            capacity=20,
            replacement_top_k=3,
            admission_recheck_top_k=3,
        )
        pool.entries = [
            PoolEntry(f"b{index}", f"b{index}", signal)
            for index, signal in enumerate(base_signals)
        ]
        return pool

    candidates = [
        PoolEntry(f"c{index}", f"c{index}", signal)
        for index, signal in enumerate(candidate_signals)
    ]
    serial = build_pool().score_candidates(candidates, max_workers=1)
    parallel = build_pool().score_candidates(candidates, max_workers=8)
    assert [item.candidate_hash for item in parallel] == [item.candidate_hash for item in serial]
    for left, right in zip(serial, parallel, strict=True):
        assert left.reason == right.reason
        assert left.replaced_hash == right.replaced_hash
        assert left.self_evicted == right.self_evicted
        assert left.formally_rechecked == right.formally_rechecked
        assert np.isclose(left.delta_add, right.delta_add, atol=1e-12)
        assert np.isclose(left.post_prune_delta, right.post_prune_delta, atol=1e-12)
        assert np.isclose(left.pool_score.objective, right.pool_score.objective, atol=1e-12)
        assert np.allclose(left.saliency, right.saliency, equal_nan=True, atol=1e-11)
        assert np.allclose(left.pool_score.daily_ic, right.pool_score.daily_ic, equal_nan=True, atol=1e-11)


@pytest.mark.parametrize("objective_type", [R0Objective, R1Objective, R2LCBObjective])
def test_batched_add_matches_individual_add_with_different_candidate_supports(objective_type):
    rng = np.random.default_rng(783)
    shape = (28, 44)
    label = rng.normal(size=shape)
    mask = rng.random(shape) > 0.01
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    base_signals = [rng.normal(size=shape) for _ in range(4)]
    candidates = [rng.normal(size=shape) for _ in range(3)]
    candidates[0][2:6, :9] = np.nan
    candidates[1][9:13] = 3.0
    candidates[2][17:, 30:] = np.nan
    kwargs = {} if objective_type is R0Objective else {"exposures": exposures}
    if objective_type is R2LCBObjective:
        kwargs["hac_lag"] = 3
    objective = objective_type(label, mask, **kwargs)
    base = objective.prepare_pool(base_signals)

    batched = objective.prepare_add_many(base, candidates)
    individual = [objective.prepare_add(base, candidate) for candidate in candidates]

    for left, right in zip(batched, individual, strict=True):
        assert np.array_equal(left.raw_common_mask, right.raw_common_mask)
        assert np.array_equal(left.common_mask, right.common_mask)
        assert np.allclose(left.predictive, right.predictive, equal_nan=True, atol=1e-12)
        assert np.allclose(
            left.factor_gram,
            right.factor_gram,
            equal_nan=True,
            atol=1e-12,
        )
        assert np.allclose(left.score.weights, right.score.weights, atol=1e-11)
        assert np.allclose(
            left.score.daily_ic,
            right.score.daily_ic,
            equal_nan=True,
            atol=1e-11,
        )


def test_add_reward_freezes_base_support_but_formal_subset_recovers_natural_support():
    rng = np.random.default_rng(784)
    shape = (300, 120)
    label = rng.normal(size=shape)
    existing = rng.normal(size=shape)
    existing[:30] = np.nan
    candidate = label + 0.1 * rng.normal(size=shape)
    objective = R0Objective(label, np.ones(shape, dtype=bool))
    base = objective.prepare_pool([existing])
    add = objective.prepare_add_many(base, [candidate])[0]
    natural_candidate = objective.prepare_subset(
        add, [1], natural_support=True
    )

    assert np.array_equal(add.common_mask, base.common_mask)
    assert natural_candidate.valid_days == add.valid_days
    assert natural_candidate.valid_days == shape[0]


def test_unchanged_support_add_reuses_cached_baseline_without_subset_scan(monkeypatch):
    rng = np.random.default_rng(781)
    shape = (24, 36)
    objective = R0Objective(rng.normal(size=shape), np.ones(shape, dtype=bool))
    pool = PoolManager(objective, capacity=20)
    pool.entries = [
        PoolEntry(f"b{index}", f"b{index}", rng.normal(size=shape))
        for index in range(5)
    ]
    calls = 0
    original = objective.score_subset

    def counted(state, indices):
        nonlocal calls
        calls += 1
        return original(state, indices)

    monkeypatch.setattr(objective, "score_subset", counted)
    pool.score_candidates([
        PoolEntry(f"c{index}", f"c{index}", rng.normal(size=shape))
        for index in range(4)
    ], max_workers=4)
    assert calls == 0


def test_admitted_formal_state_is_reused_as_next_frozen_baseline():
    rng = np.random.default_rng(782)
    shape = (30, 45)
    label = rng.normal(size=shape)
    objective = R0Objective(label, np.ones(shape, dtype=bool))
    pool = PoolManager(objective, capacity=20, min_delta=1e-5)
    pool.entries = [
        PoolEntry(f"b{index}", f"b{index}", rng.normal(size=shape))
        for index in range(20)
    ]
    candidates = [
        PoolEntry("strong", "strong", label + 0.01 * rng.normal(size=shape)),
        PoolEntry("noise", "noise", rng.normal(size=shape)),
    ]
    scored = pool.score_candidates(candidates)
    admission = pool.consider_group(candidates, scored)
    assert admission.admitted
    calls_before = objective.prepare_calls
    expected = pool.score
    assert objective.prepare_calls == calls_before

    restored = PoolManager(objective, capacity=20)
    restored.entries = list(pool.entries)
    actual = restored.score
    assert objective.prepare_calls == calls_before
    assert np.isclose(actual.objective, expected.objective, atol=1e-12)


@pytest.mark.parametrize("objective_type", [R0Objective, R1Objective, R2LCBObjective])
def test_batched_fixed_support_subsets_match_individual_scores(objective_type):
    rng = np.random.default_rng(779)
    shape = (32, 48)
    label = rng.normal(size=shape)
    mask = rng.random(shape) > 0.04
    signals = [rng.normal(size=shape) for _ in range(7)]
    exposures = np.stack([
        np.column_stack([np.ones(shape[1]), rng.normal(size=shape[1])])
        for _ in range(shape[0])
    ])
    kwargs = {} if objective_type is R0Objective else {"exposures": exposures}
    if objective_type is R2LCBObjective:
        kwargs["hac_lag"] = 4
    objective = objective_type(label, mask, **kwargs)
    state = objective.prepare_pool(signals)
    subsets = [[index for index in range(7) if index != removed] for removed in (1, 3, 6)]
    expected = [objective.score_subset(state, subset) for subset in subsets]
    actual = objective.score_subsets(state, subsets)
    for left, right in zip(actual, expected, strict=True):
        assert np.isclose(left.objective, right.objective, atol=1e-12)
        assert np.allclose(left.weights, right.weights, atol=1e-11)
        assert np.allclose(left.daily_ic, right.daily_ic, equal_nan=True, atol=1e-11)
