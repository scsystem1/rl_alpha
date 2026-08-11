from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from ..factors.calculator import daily_corr
from ..utils.numerics import finite_corr


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    reason: str
    coverage: float
    valid_days: int
    variable_day_rate: float
    max_pool_correlation: float
    mean_abs_daily_corr: float
    pooled_correlation: float
    mean_abs_daily_rank_corr: float
    correlation_coverage: float


def validate_signal(signal: np.ndarray, membership: np.ndarray, pool_signals: list[np.ndarray] | None = None, min_coverage: float = 0.80, min_days: int = 252, min_assets: int = 100, min_variable_day_rate: float = 0.80, max_pool_corr: float = 0.95) -> ValidityResult:
    signal = np.asarray(signal, dtype=float)
    membership = np.asarray(membership, dtype=bool)
    if signal.shape != membership.shape:
        raise ValueError("signal and membership shapes differ")
    finite = np.isfinite(signal) & membership
    signal_coverage = float(finite.sum() / max(1, membership.sum()))
    count = finite.sum(axis=1)
    valid_day_mask = count >= min_assets
    values = np.where(finite, signal, 0.0)
    with np.errstate(all="ignore"):
        variance = np.square(values).sum(axis=1) / count - np.square(values.sum(axis=1) / count)
    variable = valid_day_mask & (variance > 1e-12)
    valid_days = int(valid_day_mask.sum())
    variable_rate = float(variable.sum() / max(1, valid_days))
    diagnostics: list[tuple[float, float, float, float]] = []
    for pool in pool_signals or []:
        pool = np.asarray(pool, dtype=float)
        if pool.shape != signal.shape:
            raise ValueError("pool signal shape differs from candidate")
        pearson_daily = daily_corr(signal, pool, membership)[valid_day_mask]
        finite_pearson = pearson_daily[np.isfinite(pearson_daily)]
        mean_abs = float(np.mean(np.abs(finite_pearson))) if len(finite_pearson) else 0.0
        common = membership & np.isfinite(signal) & np.isfinite(pool)
        pooled = finite_corr(signal[common], pool[common])
        pooled = float(pooled) if np.isfinite(pooled) else 0.0
        rank_values = []
        for day in np.flatnonzero(valid_day_mask):
            day_common = common[day]
            if day_common.sum() < 3:
                continue
            rank_values.append(finite_corr(rankdata(signal[day, day_common]), rankdata(pool[day, day_common])))
        finite_rank = np.asarray(rank_values, dtype=float)
        finite_rank = finite_rank[np.isfinite(finite_rank)]
        mean_abs_rank = float(np.mean(np.abs(finite_rank))) if len(finite_rank) else 0.0
        correlation_coverage = float(len(finite_pearson) / max(1, int(valid_day_mask.sum())))
        diagnostics.append((mean_abs, pooled, mean_abs_rank, correlation_coverage))
    if diagnostics:
        redundancy = [max(item[0], abs(item[1]), item[2]) for item in diagnostics]
        strongest = int(np.argmax(redundancy))
        mean_abs, pooled, mean_abs_rank, corr_coverage = diagnostics[strongest]
        maximum = redundancy[strongest]
    else:
        maximum = mean_abs = pooled = mean_abs_rank = corr_coverage = 0.0
    reason = "ok"
    if signal_coverage < min_coverage: reason = "coverage_failure"
    elif valid_days < min_days: reason = "insufficient_valid_days"
    elif variable_rate < min_variable_day_rate: reason = "near_constant"
    elif maximum > max_pool_corr: reason = "near_duplicate_signal"
    return ValidityResult(
        reason == "ok",
        reason,
        signal_coverage,
        valid_days,
        variable_rate,
        maximum,
        mean_abs,
        pooled,
        mean_abs_rank,
        corr_coverage,
    )
