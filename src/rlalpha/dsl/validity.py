from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.numerics import finite_corr


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    reason: str
    coverage: float
    valid_days: int
    variable_day_rate: float
    max_pool_correlation: float


def validate_signal(signal: np.ndarray, membership: np.ndarray, pool_signals: list[np.ndarray] | None = None, min_coverage: float = 0.80, min_days: int = 252, min_assets: int = 100, min_variable_day_rate: float = 0.80, max_pool_corr: float = 0.95) -> ValidityResult:
    signal = np.asarray(signal, dtype=float)
    membership = np.asarray(membership, dtype=bool)
    if signal.shape != membership.shape:
        raise ValueError("signal and membership shapes differ")
    finite = np.isfinite(signal) & membership
    coverage = float(finite.sum() / max(1, membership.sum()))
    valid_day_mask = finite.sum(axis=1) >= min_assets
    variable = np.zeros(signal.shape[0], dtype=bool)
    for index in np.flatnonzero(valid_day_mask):
        variable[index] = np.nanstd(np.where(finite[index], signal[index], np.nan)) > 1e-6
    valid_days = int(valid_day_mask.sum())
    variable_rate = float(variable.sum() / max(1, valid_days))
    correlations = []
    for pool in pool_signals or []:
        daily = [finite_corr(signal[t][membership[t]], np.asarray(pool)[t][membership[t]]) for t in np.flatnonzero(valid_day_mask)]
        finite_daily = np.asarray(daily, dtype=float)
        correlations.append(abs(float(np.nanmean(finite_daily))) if np.isfinite(finite_daily).any() else 0.0)
    maximum = max(correlations, default=0.0)
    reason = "ok"
    if coverage < min_coverage: reason = "coverage_failure"
    elif valid_days < min_days: reason = "insufficient_valid_days"
    elif variable_rate < min_variable_day_rate: reason = "near_constant"
    elif maximum > max_pool_corr: reason = "near_duplicate_signal"
    return ValidityResult(reason == "ok", reason, coverage, valid_days, variable_rate, maximum)
