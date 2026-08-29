from __future__ import annotations

from collections import defaultdict
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


@njit(cache=True, parallel=True)
def _daily_ranks_exact(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Average ranks on an already exact, pair-specific daily support.

    The support is deliberately supplied by the caller.  Ranking a signal on
    its own finite observations and intersecting afterwards is not equivalent
    when its partner has missing values.
    """
    days, assets = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for day in prange(days):
        count = 0
        compact = np.empty(assets, dtype=np.float64)
        positions = np.empty(assets, dtype=np.int64)
        for asset in range(assets):
            if support[day, asset]:
                compact[count] = values[day, asset]
                positions[count] = asset
                count += 1
        if count < 3:
            continue
        ranks = _average_ranks(compact[:count])
        for index in range(count):
            output[day, positions[index]] = ranks[index]
    return output


@njit(cache=True, parallel=True)
def _daily_corr_from_exact_ranks(
    left_ranks: np.ndarray,
    right_ranks: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Correlate cached ranks using the legacy kernel's arithmetic order."""
    days, assets = left_ranks.shape
    output = np.full(days, np.nan, dtype=np.float64)
    for day in prange(days):
        count = 0
        left_sum = 0.0
        right_sum = 0.0
        for asset in range(assets):
            if support[day, asset]:
                count += 1
                left_sum += left_ranks[day, asset]
                right_sum += right_ranks[day, asset]
        if count < 3:
            continue
        left_mean = left_sum / count
        right_mean = right_sum / count
        covariance = 0.0
        left_variance = 0.0
        right_variance = 0.0
        for asset in range(assets):
            if support[day, asset]:
                centered_left = left_ranks[day, asset] - left_mean
                centered_right = right_ranks[day, asset] - right_mean
                covariance += centered_left * centered_right
                left_variance += centered_left * centered_left
                right_variance += centered_right * centered_right
        denominator = np.sqrt(left_variance * right_variance)
        if denominator > 0.0:
            output[day] = covariance / denominator
    return output


def _batch_rank_correlations_exact(
    signals: list[np.ndarray],
    pools: list[np.ndarray],
    membership: np.ndarray,
    valid_day_masks: list[np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute exact pairwise daily rank correlations with bounded reuse.

    Pairs are grouped only when their complete pair-specific support masks are
    byte-for-byte identical.  Within such a group, ranks from the smaller side
    of the candidate/pool bipartite graph are retained and the larger side is
    streamed one array at a time.  With the formal 8-by-20 protocol this caps
    the rank cache at nine panel-shaped float64 arrays.
    """
    if not signals or not pools:
        return {}
    membership = np.asarray(membership, dtype=bool)
    groups: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    for signal_index, (signal, valid_days) in enumerate(zip(signals, valid_day_masks, strict=True)):
        signal = np.asarray(signal, dtype=float)
        for pool_index, pool in enumerate(pools):
            pool = np.asarray(pool, dtype=float)
            support = (
                membership
                & np.asarray(valid_days, dtype=bool)[:, None]
                & np.isfinite(signal)
                & np.isfinite(pool)
            )
            # The packed bytes are the exact key, not merely a digest.  This
            # makes support reuse collision-free and therefore semantic, while
            # remaining temporary and bounded to one rollout batch.
            groups[np.packbits(support, axis=None).tobytes()].append((signal_index, pool_index))

    results: dict[tuple[int, int], np.ndarray] = {}
    for pairs in groups.values():
        candidate_counts: dict[int, int] = defaultdict(int)
        pool_counts: dict[int, int] = defaultdict(int)
        for signal_index, pool_index in pairs:
            candidate_counts[signal_index] += 1
            pool_counts[pool_index] += 1
        reusable = max((*candidate_counts.values(), *pool_counts.values()), default=0) > 1
        if not reusable:
            for signal_index, pool_index in pairs:
                results[(signal_index, pool_index)] = _daily_rank_corr_exact(
                    signals[signal_index], pools[pool_index], membership, valid_day_masks[signal_index]
                )
            continue

        first_signal, first_pool = pairs[0]
        support = (
            membership
            & np.asarray(valid_day_masks[first_signal], dtype=bool)[:, None]
            & np.isfinite(signals[first_signal])
            & np.isfinite(pools[first_pool])
        )
        pair_set = set(pairs)
        candidate_indices = sorted(candidate_counts)
        pool_indices = sorted(pool_counts)
        if len(candidate_indices) <= len(pool_indices):
            cached = {
                index: _daily_ranks_exact(np.asarray(signals[index], dtype=float), support)
                for index in candidate_indices
            }
            for pool_index in pool_indices:
                pool_ranks = _daily_ranks_exact(np.asarray(pools[pool_index], dtype=float), support)
                for signal_index in candidate_indices:
                    if (signal_index, pool_index) in pair_set:
                        results[(signal_index, pool_index)] = _daily_corr_from_exact_ranks(
                            cached[signal_index], pool_ranks, support
                        )
        else:
            cached = {
                index: _daily_ranks_exact(np.asarray(pools[index], dtype=float), support)
                for index in pool_indices
            }
            for signal_index in candidate_indices:
                signal_ranks = _daily_ranks_exact(np.asarray(signals[signal_index], dtype=float), support)
                for pool_index in pool_indices:
                    if (signal_index, pool_index) in pair_set:
                        results[(signal_index, pool_index)] = _daily_corr_from_exact_ranks(
                            signal_ranks, cached[pool_index], support
                        )
    return results


def _intrinsic_validity(
    signal: np.ndarray,
    membership: np.ndarray,
    min_coverage: float,
    min_days: int,
    min_assets: int,
    min_variable_day_rate: float,
) -> tuple[np.ndarray, float, np.ndarray, int, float, ValidityResult | None]:
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
    if signal_coverage < min_coverage:
        early = _result("coverage_failure", signal_coverage, valid_days, variable_rate)
    elif valid_days < min_days:
        early = _result("insufficient_valid_days", signal_coverage, valid_days, variable_rate)
    elif variable_rate < min_variable_day_rate:
        early = _result("near_constant", signal_coverage, valid_days, variable_rate)
    else:
        early = None
    return signal, signal_coverage, valid_day_mask, valid_days, variable_rate, early


def _finish_validity(
    signal: np.ndarray,
    membership: np.ndarray,
    pools: list[np.ndarray],
    signal_coverage: float,
    valid_day_mask: np.ndarray,
    valid_days: int,
    variable_rate: float,
    max_pool_corr: float,
    rank_results: dict[int, np.ndarray] | None = None,
    *,
    parallel_rank: bool = True,
) -> ValidityResult:
    diagnostics: list[tuple[float, float, float, float]] = []
    for pool_index, pool in enumerate(pools):
        pool = np.asarray(pool, dtype=float)
        if pool.shape != signal.shape:
            raise ValueError("pool signal shape differs from candidate")
        pearson_daily = daily_corr(signal, pool, membership)[valid_day_mask]
        finite_pearson = pearson_daily[np.isfinite(pearson_daily)]
        mean_abs = float(np.mean(np.abs(finite_pearson))) if len(finite_pearson) else 0.0
        common = membership & np.isfinite(signal) & np.isfinite(pool)
        pooled = finite_corr(signal[common], pool[common])
        pooled = float(pooled) if np.isfinite(pooled) else 0.0
        if rank_results is None:
            rank_kernel = _daily_rank_corr_exact if parallel_rank else _daily_rank_corr_exact_serial
            finite_rank = rank_kernel(signal, pool, membership, valid_day_mask)
        else:
            finite_rank = rank_results[pool_index]
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
    signal, signal_coverage, valid_day_mask, valid_days, variable_rate, early = _intrinsic_validity(
        signal, membership, min_coverage, min_days, min_assets, min_variable_day_rate
    )
    # Cheap intrinsic gates must run before any O(pool_size * panel) redundancy
    # work.  Previously even obviously sparse/constant candidates paid for
    # thousands of daily rank correlations against every pool member.
    if early is not None:
        return early
    return _finish_validity(
        signal,
        np.asarray(membership, dtype=bool),
        list(pool_signals or []),
        signal_coverage,
        valid_day_mask,
        valid_days,
        variable_rate,
        max_pool_corr,
        parallel_rank=parallel_rank,
    )


def validate_signals(
    signals: list[np.ndarray],
    membership: np.ndarray,
    pool_signals: list[np.ndarray] | None = None,
    min_coverage: float = 0.80,
    min_days: int = 252,
    min_assets: int = 100,
    min_variable_day_rate: float = 0.80,
    max_pool_corr: float = 0.95,
) -> list[ValidityResult]:
    """Validate one frozen candidate group with exact, bounded rank reuse.

    All scalar gates, Pearson diagnostics, tie handling and strongest-pool
    selection are shared with :func:`validate_signal`.  Only daily rank arrays
    on byte-identical pair supports are reused.
    """
    membership = np.asarray(membership, dtype=bool)
    pools = [np.asarray(pool, dtype=float) for pool in (pool_signals or [])]
    prepared = [
        _intrinsic_validity(
            signal, membership, min_coverage, min_days, min_assets, min_variable_day_rate
        )
        for signal in signals
    ]
    eligible_indices = [index for index, item in enumerate(prepared) if item[-1] is None]
    eligible_signals = [prepared[index][0] for index in eligible_indices]
    eligible_masks = [prepared[index][2] for index in eligible_indices]
    batched = _batch_rank_correlations_exact(eligible_signals, pools, membership, eligible_masks)
    eligible_position = {source: target for target, source in enumerate(eligible_indices)}
    results: list[ValidityResult] = []
    for index, item in enumerate(prepared):
        signal, coverage, valid_mask, valid_days, variable_rate, early = item
        if early is not None:
            results.append(early)
            continue
        position = eligible_position[index]
        rank_results = {
            pool_index: batched[(position, pool_index)] for pool_index in range(len(pools))
        }
        results.append(
            _finish_validity(
                signal,
                membership,
                pools,
                coverage,
                valid_mask,
                valid_days,
                variable_rate,
                max_pool_corr,
                rank_results,
            )
        )
    return results
