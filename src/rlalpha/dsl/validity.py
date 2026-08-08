from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..factors.calculator import daily_corr


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
    count = finite.sum(axis=1)
    valid_day_mask = count >= min_assets
    values = np.where(finite, signal, 0.0)
    with np.errstate(all="ignore"):
        variance = np.square(values).sum(axis=1) / count - np.square(values.sum(axis=1) / count)
    variable = valid_day_mask & (variance > 1e-12)
    valid_days = int(valid_day_mask.sum())
    variable_rate = float(variable.sum() / max(1, valid_days))
    correlations = []
    for pool in pool_signals or []:
        finite_daily = daily_corr(signal, np.asarray(pool), membership)
        finite_daily = finite_daily[valid_day_mask]
        correlations.append(abs(float(np.nanmean(finite_daily))) if np.isfinite(finite_daily).any() else 0.0)
    maximum = max(correlations, default=0.0)
    reason = "ok"
    if coverage < min_coverage: reason = "coverage_failure"
    elif valid_days < min_days: reason = "insufficient_valid_days"
    elif variable_rate < min_variable_day_rate: reason = "near_constant"
    elif maximum > max_pool_corr: reason = "near_duplicate_signal"
    return ValidityResult(reason == "ok", reason, coverage, valid_days, variable_rate, maximum)
