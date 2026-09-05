from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import PreparedPoolState, RewardObjective, _fixed_universe_daily_corr
from .statistics import gap_aware_mean_se
from ..factors.moments import daily_fixed_universe_moments, solve_psd_ridge
from ..factors.records import PoolIncrement, PoolScore
from ..factors.transform import combine_fixed_signals, prepare_fixed_universe_inputs


DEFAULT_TIME_FOLDS = (
    {"fit": ("2010-01-01", "2012-12-31"), "score": ("2013-01-01", "2014-12-31")},
    {"fit": ("2012-01-01", "2014-12-31"), "score": ("2015-01-01", "2016-12-31")},
    {"fit": ("2014-01-01", "2016-12-31"), "score": ("2017-01-01", "2018-12-31")},
)


@dataclass(frozen=True)
class FoldFit:
    gram: np.ndarray
    predictive: np.ndarray
    inverse: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PreparedOOFPoolState(PreparedPoolState):
    fold_fits: tuple[FoldFit, ...]


class WalkForwardObjective(RewardObjective):
    """Train-only rolling fits with paired, date-aligned subsequent scores.

    Cross-sectional transforms are shared by all folds. Only small sufficient
    statistics and ridge fits differ. score.weights always means FULL train
    weights for outer validation, never an average of the fold weights.
    """

    def __init__(self, *args, dates, time_folds=DEFAULT_TIME_FOLDS,
                 horizon_trading_days=20, hac_lag=20, critical_value=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.exposures is None:
            raise ValueError("OOF RNIC requires risk exposures")
        self.dates = np.asarray(dates, dtype="datetime64[D]")
        if (self.dates.shape != (len(self.label),) or np.isnat(self.dates).any()
                or np.any(self.dates[1:] <= self.dates[:-1])):
            raise ValueError("OOF dates must be a strictly increasing trading-day axis")
        if horizon_trading_days < 1 or hac_lag < 0 or not np.isfinite(critical_value) or critical_value < 0:
            raise ValueError("invalid OOF horizon or uncertainty settings")
        self.hac_lag, self.critical_value = int(hac_lag), float(critical_value)
        self.horizon_trading_days = int(horizon_trading_days)
        offset = self.horizon_trading_days + 1  # next-close entry, t+h+1 exit
        exits = np.full(self.dates.shape, np.datetime64("NaT", "D"))
        if len(exits) > offset:
            exits[:-offset] = self.dates[offset:]
        self.time_folds = tuple({part: tuple(str(v) for v in fold[part]) for part in ("fit", "score")} for fold in time_folds)
        if not self.time_folds:
            raise ValueError("OOF requires explicit nonempty time folds")
        self.fit_rows, self.score_rows = [], []
        previous_end = np.datetime64("1678-01-01")
        for fold in self.time_folds:
            a, b = (np.asarray(fold[part], dtype="datetime64[D]") for part in ("fit", "score"))
            if (a.shape != (2,) or b.shape != (2,) or np.isnat(a).any() or np.isnat(b).any()
                    or not (a[0] <= a[1] < b[0] <= b[1]) or b[0] <= previous_end):
                raise ValueError("OOF needs chronological fits and disjoint ordered score windows")
            previous_end = b[1]
            for bounds, destination in ((a, self.fit_rows), (b, self.score_rows)):
                # Allow the holiday/weekend at a calendar boundary, not a truncated panel.
                if (self.dates[0] > bounds[0] + np.timedelta64(7, "D")
                        or self.dates[-1] < bounds[1] - np.timedelta64(7, "D")):
                    raise ValueError("panel does not cover the configured OOF folds")
                destination.append((self.dates >= bounds[0]) & (self.dates <= bounds[1]) & (exits <= bounds[1]))
        self.scoring_rows = np.logical_or.reduce(self.score_rows)
        self._coverage = None
        self._zero_daily = None

    def objective_name(self):
        return "r2_paired_oof" if self.critical_value else "r1_oof"

    def _check_coverage(self, raw_common, common, label):
        if self._coverage is not None:
            return
        zero = _fixed_universe_daily_corr(np.zeros_like(label), label, common)
        records = []
        for k, (fit, score) in enumerate(zip(self.fit_rows, self.score_rows, strict=True)):
            record = {"fold": k, **self.time_folds[k]}
            for name, rows in (("fit", fit), ("score", score)):
                expected_days = int((rows & (raw_common.sum(axis=1) >= 3)).sum())
                valid_days = int((rows & np.isfinite(zero)).sum())
                required = max(self.min_pool_valid_days, int(np.ceil(self.min_pool_valid_day_rate * expected_days)))
                observations = int(common[rows].sum())
                expected_obs = int(raw_common[rows].sum())
                rate = observations / max(1, expected_obs)
                record[name + "_support"] = {"valid_days": valid_days, "required_valid_days": required,
                    "valid_observations": observations, "base_observations": expected_obs,
                    "valid_day_rate": valid_days / max(1, expected_days), "observation_rate": rate}
                if valid_days < required or rate < self.min_pool_observation_rate:
                    raise ValueError(f"insufficient OOF {name} support in fold {k}: {record[name + '_support']}")
            records.append(record)
        self._zero_daily = np.where(self.scoring_rows, zero, np.nan)
        self._coverage = records

    def support_diagnostics(self, state):
        self._check_coverage(state.raw_common_mask, state.common_mask, state.prepared_label)
        days = int(np.isfinite(state.score.daily_ic).sum())
        expected_days = int(np.isfinite(self._zero_daily).sum())
        observations = int(state.common_mask[self.scoring_rows].sum())
        base_obs = int(state.raw_common_mask[self.scoring_rows].sum())
        return {"valid": days == expected_days, "valid_days": days,
            "required_valid_days": expected_days, "valid_observations": observations,
            "base_observations": base_obs, "valid_day_rate": days / max(1, expected_days),
            "observation_rate": observations / max(1, base_obs), "folds": self._coverage}

    def prepare_pool(self, signals):
        if signals:
            return super().prepare_pool(signals)
        empty = super().prepare_pool([])
        return self._from_prepared([], empty.raw_common_mask, empty.common_mask, [], empty.prepared_label)[0]

    def _build_independent_state(self, raw_signals, reference_mask, prepared_signals=None):
        deployment = prepared_signals if prepared_signals is not None else self._independent_signals(raw_signals, reference_mask)
        prepared, label, common, _ = prepare_fixed_universe_inputs(
            deployment, self.label, reference_mask, self.exposures, neutralize=True)
        return self._from_prepared(raw_signals, reference_mask, common, list(prepared), label)[0]

    def _from_prepared(self, raw, raw_common, common, prepared, label, subsets=None):
        self._check_coverage(raw_common, common, label)
        dg, dp, valid = daily_fixed_universe_moments(prepared, label, common)
        moments = [(dg[valid].mean(axis=0), dp[valid].mean(axis=0))]
        moments.extend((dg[rows & valid].mean(axis=0), dp[rows & valid].mean(axis=0)) for rows in self.fit_rows)
        subsets = [list(range(len(raw)))] if subsets is None else subsets
        states = []
        for indices in subsets:
            idx = np.asarray(indices, dtype=int)
            selected = [(g[np.ix_(idx, idx)], p[idx]) for g, p in moments]
            states.append(self._assemble([raw[i] for i in idx], raw_common, common,
                [prepared[i] for i in idx], label, selected[0], selected[1:]))
        return states

    def _assemble(self, raw, raw_common, common, prepared, label, full_moments, fold_moments):
        def fit(g, p):
            w, inverse = solve_psd_ridge(g, p, self.ridge) if len(p) else (np.empty(0), np.empty((0, 0)))
            return FoldFit(g, p, inverse, w)
        full = fit(*full_moments)
        folds = tuple(fit(*moments) for moments in fold_moments)
        daily = self._zero_daily.copy()
        if prepared:
            for rows, fitted in zip(self.score_rows, folds, strict=True):
                combined, _ = combine_fixed_signals([value[rows] for value in prepared], fitted.weights)
                daily[rows] = _fixed_universe_daily_corr(combined, label[rows], common[rows])
        if not np.array_equal(np.isfinite(daily), np.isfinite(self._zero_daily)):
            raise ValueError("OOF prediction changed the fixed scoring dates")
        mean = float(np.nanmean(daily))
        score = PoolScore(mean, mean, tuple(map(float, daily)), tuple(map(float, full.weights)), gap_aware_mean_se(daily, self.hac_lag))
        return PreparedOOFPoolState(tuple(raw), raw_common, common, tuple(prepared), label,
            full.gram, full.predictive, full.inverse, score, folds)

    def prepare_add_many(self, base, candidates):
        arrays = [np.asarray(value, dtype=float) for value in candidates]
        if not arrays:
            return []
        return self._prepare_add_many_from_prepared_candidates(base, arrays, self._independent_signals(arrays, base.raw_common_mask))

    def _prepare_add_many_from_prepared_candidates(self, base, candidates, prepared_candidates):
        prepared, label, common, _ = prepare_fixed_universe_inputs(
            prepared_candidates, self.label, base.raw_common_mask, self.exposures, neutralize=True)
        if not np.array_equal(common, base.common_mask) or not np.allclose(label, base.prepared_label, equal_nan=True):
            raise RuntimeError("fixed metric universe changed while adding OOF candidates")
        n = len(base.raw_signals)
        return self._from_prepared(list(base.raw_signals) + candidates, base.raw_common_mask, common,
            list(base.prepared_signals) + list(prepared), base.prepared_label,
            [list(range(n)) + [n + i] for i in range(len(candidates))])

    def prepare_subset(self, state, indices, *, natural_support=False):
        idx = np.asarray(indices, dtype=int)
        subset = lambda g, p: (g[np.ix_(idx, idx)], p[idx])
        return self._assemble([state.raw_signals[i] for i in idx], state.raw_common_mask, state.common_mask,
            [state.prepared_signals[i] for i in idx], state.prepared_label,
            subset(state.factor_gram, state.predictive), [subset(f.gram, f.predictive) for f in state.fold_fits])

    def score_subsets(self, state, subsets):
        return [self.prepare_subset(state, indices).score for indices in subsets]

    def compare_scores(self, old, new):
        before, after = np.asarray(old.daily_ic), np.asarray(new.daily_ic)
        if (before.shape != self.dates.shape or after.shape != before.shape
                or not np.array_equal(np.isfinite(before), np.isfinite(after))):
            raise ValueError("OOF comparison requires the same scoring dates")
        delta = after - before
        n = int(np.isfinite(delta).sum())
        if n < 2:
            raise ValueError("OOF comparison has insufficient paired dates")
        mean, se = float(np.nanmean(delta)), gap_aware_mean_se(delta, self.hac_lag)
        penalty = self.critical_value * se
        return PoolIncrement(mean, se, penalty, mean - penalty, n,
            tuple(float(np.nanmean(delta[rows])) for rows in self.score_rows))

    def saliency(self, state):
        return np.mean([np.square(f.weights) / np.diag(f.inverse) for f in state.fold_fits], axis=0)

    def prompt_summary(self, state):
        weights = [f.weights / max(float(np.abs(f.weights).sum()), 1e-30) for f in state.fold_fits]
        return {"oof_mean_rnic": state.score.mean_ic,
            "normalized_fold_weights": tuple(map(float, np.mean(weights, axis=0)))}

    def snapshot_diagnostics(self, state):
        return {"estimator": "rolling_oof", "time_folds": self.time_folds,
            "weights_source": "full_train", "fold_weights": [f.weights.tolist() for f in state.fold_fits],
            "fold_mean_rnic": [float(np.nanmean(np.asarray(state.score.daily_ic)[r])) for r in self.score_rows],
            "hac_lag": self.hac_lag, "critical_value": self.critical_value,
            "label_exit_offset": self.horizon_trading_days + 1}
