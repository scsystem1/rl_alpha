from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rlalpha.factors.pool import PoolManager
from rlalpha.rewards.factory import objective_for, prompt_objective_for
from rlalpha.search.prompt_diagnostics import PoolPromptDiagnostics


def _sample():
    dates = pd.bdate_range("2018-07-01", "2020-06-30")
    rng = np.random.default_rng(101)
    shape = (len(dates), 40)
    signals = [rng.normal(size=shape) for _ in range(3)]
    risk = np.stack([np.ones(shape), rng.normal(size=shape)], axis=-1)
    panel = SimpleNamespace(target_dates=dates, label=.3 * signals[0] + .15 * signals[1] + rng.normal(size=shape),
        common_mask=np.ones(shape, bool), exposures=risk, target=lambda value: value)
    config = {"ridge": .01, "hac_lag": 20, "critical_value": .5, "min_pool_valid_days": 80,
        "min_pool_valid_day_rate": .8, "min_pool_observation_rate": .8,
        "time_folds": [
            {"fit": ["2018-07-01", "2018-12-31"], "score": ["2019-01-01", "2019-06-30"]},
            {"fit": ["2019-01-01", "2019-06-30"], "score": ["2019-07-01", "2019-12-31"]},
            {"fit": ["2019-07-01", "2019-12-31"], "score": ["2020-01-01", "2020-06-30"]}]}
    return panel, signals, config


def test_recent_three_fold_support_and_next_close_label_purge():
    panel, signals, config = _sample()
    objective = objective_for("r1_oof", panel, config)
    state = objective.prepare_pool(signals)
    assert objective.support_diagnostics(state)["valid"]
    assert objective.snapshot_diagnostics(state)["estimator"] == "rolling_oof"
    for fold, fit, score in zip(objective.time_folds, objective.fit_rows, objective.score_rows):
        for name, rows in (("fit", fit), ("score", score)):
            positions = np.flatnonzero((panel.target_dates >= fold[name][0]) & (panel.target_dates <= fold[name][1]))
            assert rows.sum() == len(positions) - 21
            assert np.flatnonzero(rows)[-1] == positions[-22]
            assert rows.sum() >= 80
    with pytest.raises(ValueError, match="insufficient OOF"):
        objective_for("r1_oof", panel, {**config, "min_pool_valid_days": 252}).prepare_pool(signals)


def test_paired_lcb_changes_only_increment_penalty():
    panel, signals, config = _sample()
    r1, r2 = [objective_for(name, panel, config) for name in ("r1_oof", "r2_paired_oof")]
    old1, old2 = [obj.prepare_pool(signals[:1]) for obj in (r1, r2)]
    new1, new2 = [obj.prepare_pool(signals[:2]) for obj in (r1, r2)]
    for left, right in ((old1, old2), (new1, new2)):
        np.testing.assert_allclose(left.score.daily_ic, right.score.daily_ic, equal_nan=True)
        np.testing.assert_allclose(left.score.weights, right.score.weights)
        for a, b in zip(left.fold_fits, right.fold_fits):
            np.testing.assert_allclose(a.weights, b.weights)
    a, b = r1.compare_scores(old1.score, new1.score), r2.compare_scores(old2.score, new2.score)
    assert a.mean_delta == pytest.approx(b.mean_delta)
    assert a.standard_error == pytest.approx(b.standard_error)
    assert a.reward - b.reward == pytest.approx(.5 * a.standard_error)
    zero = objective_for("r2_paired_oof", panel, {**config, "critical_value": 0})
    assert zero.compare_scores(old1.score, new1.score).reward == a.reward


def test_prompt_evidence_preserves_custom_recent_folds():
    panel, _, config = _sample()
    objective = objective_for("r2_paired_oof", panel, config)
    assert prompt_objective_for(panel, config).time_folds == objective.time_folds
    pool = PoolManager(objective)
    diagnostic = PoolPromptDiagnostics(panel, pool, config)
    assert diagnostic.objective is objective
    assert diagnostic().oof_mean_rnic == 0.0
