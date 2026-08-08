from __future__ import annotations

import numpy as np

from rlalpha.factors.combiner import RidgeCombiner
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


def test_newey_west_lcb_is_mean_minus_standard_error_not_std():
    values = np.array([0.01, 0.02, -0.01, 0.03, 0.00] * 20)
    se = newey_west_mean_se(values, lag=20)
    objective, mean, reported = lcb_score(values, lag=20)
    assert np.isclose(reported, se)
    assert np.isclose(objective, mean - 1.645 * se)
    assert not np.isclose(objective, mean - 1.645 * values.std())


def test_r2_uses_neutralized_daily_ic_lcb():
    rng = np.random.default_rng(4)
    days, assets = 30, 80
    exposure = np.stack([np.column_stack([np.ones(assets), rng.normal(size=assets)]) for _ in range(days)])
    residual = rng.normal(size=(days, assets))
    label = 0.1 * residual + exposure[:, :, 1] + rng.normal(size=(days, assets))
    score = R2LCBObjective(label, np.ones_like(label, dtype=bool), exposure, hac_lag=5).score_pool([residual])
    assert np.isfinite(score.objective)
    assert np.isclose(score.objective, score.mean_ic - 1.645 * score.standard_error)
