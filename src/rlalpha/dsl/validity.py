from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

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


def _result(
    reason: str,
    coverage: float,
    valid_days: int,
    variable_rate: float,
    maximum: float = 0.0,
    mean_abs: float = 0.0,
    pooled: float = 0.0,
    mean_abs_rank: float = 0.0,
    correlation_coverage: float = 0.0,
) -> ValidityResult:
    return ValidityResult(
        reason == "ok",
        reason,
        coverage,
        valid_days,
        variable_rate,
        maximum,
        mean_abs,
        pooled,
        mean_abs_rank,
        correlation_coverage,
    )


@njit(cache=True)
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Exact scipy.stats.rankdata(method='average') equivalent for finite data."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        value = values[order[start]]
        while end < len(values) and values[order[end]] == value:
            end += 1
        average = 0.5 * (start + 1 + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


@njit(cache=True, parallel=True)
def _daily_rank_corr_exact(
    signal: np.ndarray,
    pool: np.ndarray,
    membership: np.ndarray,
    valid_day_mask: np.ndarray,
) -> np.ndarray:
    """Compute exact common-support daily rank correlations outside Python."""
    days, assets = signal.shape
    output = np.full(days, np.nan, dtype=np.float64)
    for day in prange(days):
        if not valid_day_mask[day]:
            continue
        count = 0
        left = np.empty(assets, dtype=np.float64)
        right = np.empty(assets, dtype=np.float64)
        for asset in range(assets):
            if membership[day, asset] and np.isfinite(signal[day, asset]) and np.isfinite(pool[day, asset]):
                left[count] = signal[day, asset]
                right[count] = pool[day, asset]
                count += 1
        if count < 3:
            continue
        left_rank = _average_ranks(left[:count])
        right_rank = _average_ranks(right[:count])
        left_mean = np.mean(left_rank)
        right_mean = np.mean(right_rank)
        covariance = 0.0
        left_variance = 0.0
        right_variance = 0.0
        for index in range(count):
            centered_left = left_rank[index] - left_mean
            centered_right = right_rank[index] - right_mean
            covariance += centered_left * centered_right
            left_variance += centered_left * centered_left
            right_variance += centered_right * centered_right
        denominator = np.sqrt(left_variance * right_variance)
        if denominator > 0.0:
            output[day] = covariance / denominator
    return output


@njit(cache=True, nogil=True)
def _daily_rank_corr_exact_serial(
    signal: np.ndarray,
    pool: np.ndarray,
    membership: np.ndarray,
    valid_day_mask: np.ndarray,
) -> np.ndarray:
    """Thread-safe inner kernel for outer candidate-level parallelism.

    Numba's workqueue backend cannot safely run several parallel regions from
    independent Python threads.  GRPO therefore distributes candidates across
    threads and uses this single-threaded, GIL-free kernel inside each one.
    """
    days, assets = signal.shape
    output = np.full(days, np.nan, dtype=np.float64)
    for day in range(days):
        if not valid_day_mask[day]:
            continue
        count = 0
        left = np.empty(assets, dtype=np.float64)
        right = np.empty(assets, dtype=np.float64)
        for asset in range(assets):
            if membership[day, asset] and np.isfinite(signal[day, asset]) and np.isfinite(pool[day, asset]):
                left[count] = signal[day, asset]
                right[count] = pool[day, asset]
                count += 1
        if count < 3:
            continue
        left_rank = _average_ranks(left[:count])
        right_rank = _average_ranks(right[:count])
        left_mean = np.mean(left_rank)
        right_mean = np.mean(right_rank)
        covariance = 0.0
        left_variance = 0.0
        right_variance = 0.0
        for index in range(count):
            centered_left = left_rank[index] - left_mean
            centered_right = right_rank[index] - right_mean
            covariance += centered_left * centered_right
            left_variance += centered_left * centered_left
            right_variance += centered_right * centered_right
        denominator = np.sqrt(left_variance * right_variance)
        if denominator > 0.0:
            output[day] = covariance / denominator
    return output


def validate_signal(
    signal: np.ndarray,
    membership: np.ndarray,
    pool_signals: list[np.ndarray] | None = None,
    min_coverage: float = 0.80,
    min_days: int = 252,
    min_assets: int = 100,
    min_variable_day_rate: float = 0.80,
    max_pool_corr: float = 0.95,
    *,
    parallel_rank: bool = True,
) -> ValidityResult:
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
    # Cheap intrinsic gates must run before any O(pool_size * panel) redundancy
    # work.  Previously even obviously sparse/constant candidates paid for
    # thousands of daily rank correlations against every pool member.
    if signal_coverage < min_coverage:
        return _result("coverage_failure", signal_coverage, valid_days, variable_rate)
    if valid_days < min_days:
        return _result("insufficient_valid_days", signal_coverage, valid_days, variable_rate)
    if variable_rate < min_variable_day_rate:
        return _result("near_constant", signal_coverage, valid_days, variable_rate)
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
        # The exact common-support calculation is unchanged, but the day loop,
        # tie-aware ranking and correlations run in parallel Numba code rather
        # than tens of thousands of Python/scipy calls per candidate.
        rank_kernel = _daily_rank_corr_exact if parallel_rank else _daily_rank_corr_exact_serial
        finite_rank = rank_kernel(signal, pool, membership, valid_day_mask)
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
    reason = "near_duplicate_signal" if maximum > max_pool_corr else "ok"
    return _result(
        reason, signal_coverage, valid_days, variable_rate, maximum,
        mean_abs, pooled, mean_abs_rank, corr_coverage,
    )
