from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..rewards.statistics import gap_aware_mean_se


def bootstrap_date_indices(
    dates: pd.DatetimeIndex | np.ndarray,
    block: int = 20,
    samples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Shared moving-block draws, stratified by calendar year.

    Dates must contain the complete trading-day grid, including dates whose
    score is missing. Draws depend only on that grid, never on a method's
    scores or finite-value mask, so paired comparisons use identical draws.
    """
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("bootstrap dates must be unique and increasing")
    if block < 1 or samples < 1:
        raise ValueError("bootstrap block and samples must be positive")
    rng = np.random.default_rng(seed)
    draws = []
    for year in dates.year.unique():
        positions = np.flatnonzero(dates.year == year)
        width = min(block, len(positions))
        count = (len(positions) + width - 1) // width
        starts = rng.integers(0, len(positions) - width + 1, size=(samples, count))
        local = (starts[..., None] + np.arange(width)).reshape(samples, -1)[:, :len(positions)]
        draws.append(positions[local])
    return np.concatenate(draws, axis=1) if draws else np.empty((samples, 0), dtype=int)


def moving_block_bootstrap(
    values: np.ndarray, block: int = 20, samples: int = 2000, seed: int = 0,
    *, dates: pd.DatetimeIndex | np.ndarray | None = None,
    bootstrap_indices: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bootstrap values must be a vector")
    if dates is not None and len(dates) != len(values):
        raise ValueError("bootstrap dates and values differ in length")
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    if bootstrap_indices is None:
        if dates is None:
            # No supplied calendar means one stratum on the original grid.
            if block < 1 or samples < 1:
                raise ValueError("bootstrap block and samples must be positive")
            width = min(block, len(values))
            count = (len(values) + width - 1) // width
            starts = np.random.default_rng(seed).integers(0, len(values) - width + 1, size=(samples, count))
            bootstrap_indices = (starts[..., None] + np.arange(width)).reshape(samples, -1)[:, :len(values)]
        else:
            bootstrap_indices = bootstrap_date_indices(dates, block, samples, seed)
    indices = np.asarray(bootstrap_indices)
    if indices.shape != (samples, len(values)) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("bootstrap indices must have shape (samples, trading_days)")
    if np.any((indices < 0) | (indices >= len(values))):
        raise ValueError("bootstrap indices are out of bounds")
    draw = values[indices]
    if dates is None:
        counts = np.isfinite(draw).sum(axis=1)
        means = np.divide(np.where(np.isfinite(draw), draw, 0.0).sum(axis=1), counts,
                          out=np.full(samples, np.nan), where=counts > 0)
    else:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        means = np.zeros(samples)
        total = int(np.isfinite(values).sum())
        for year in np.unique(years):
            columns = years == year
            original_count = int(np.isfinite(values[columns]).sum())
            if not original_count:
                continue
            local = draw[:, columns]
            counts = np.isfinite(local).sum(axis=1)
            annual_mean = np.divide(np.where(np.isfinite(local), local, 0.0).sum(axis=1), counts,
                                    out=np.full(samples, np.nan), where=counts > 0)
            # Keep the original effective-day mixture of years fixed. A draw
            # with no observations in a positive-weight year stays invalid.
            means += annual_mean * (original_count / total)
    finite_means = means[np.isfinite(means)]
    if not len(finite_means):
        return float("nan"), float("nan")
    low, high = np.quantile(finite_means, [0.025, 0.975])
    return float(low), float(high)


def series_summary(
    values: np.ndarray, hac_lag: int = 20, bootstrap_samples: int = 2000, seed: int = 0,
    *, dates: pd.DatetimeIndex | np.ndarray | None = None,
    bootstrap_block: int = 20, bootstrap_indices: np.ndarray | None = None,
) -> dict[str, float | list[float] | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or (dates is not None and len(dates) != len(values)):
        raise ValueError("summary requires a vector and matching trading dates")
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "hac_se": float("nan"), "hac_t": float("nan"), "p_value": float("nan"), "bootstrap_95_ci": [float("nan"), float("nan")]}
    mean = float(finite.mean())
    se = gap_aware_mean_se(values, hac_lag)
    statistic = mean / se if np.isfinite(se) and se > 0 else float("nan")
    return {
        "n": int(len(finite)),
        "mean": mean,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "hac_se": se,
        "hac_t": statistic,
        "p_value": float(2.0 * norm.sf(abs(statistic))),
        "bootstrap_95_ci": list(moving_block_bootstrap(values, bootstrap_block, bootstrap_samples, seed, dates=dates, bootstrap_indices=bootstrap_indices)),
    }


def paired_summary(first: np.ndarray, second: np.ndarray, **kwargs: object) -> dict[str, float | list[float] | int]:
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape:
        raise ValueError("paired series shapes differ")
    difference = np.where(np.isfinite(first) & np.isfinite(second), first - second, np.nan)
    return series_summary(difference, **kwargs)


def factor_significance(
    values: np.ndarray,
    *,
    hac_lag: int = 20,
    bootstrap_block: int = 20,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    min_days: int = 30,
    dates: pd.DatetimeIndex | np.ndarray | None = None,
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or (dates is not None and len(dates) != len(values)):
        raise ValueError("significance requires a vector and matching trading dates")
    finite = values[np.isfinite(values)]
    base: dict[str, object] = {
        "n_days": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else float("nan"),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "positive_day_rate": float(np.mean(finite > 0)) if len(finite) else float("nan"),
        "hac_lag": int(hac_lag),
        "bootstrap_block_length": int(bootstrap_block),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }
    if len(finite) < min_days or len(finite) < 2 or np.std(finite, ddof=1) <= 1e-15:
        return {**base, "status": "insufficient_data", "hac_se": float("nan"), "hac_t": float("nan"), "p_value": float("nan"), "bootstrap_95_ci": [float("nan"), float("nan")]}
    se = gap_aware_mean_se(values, hac_lag)
    if not np.isfinite(se) or se <= 0:
        return {**base, "status": "insufficient_data", "hac_se": se, "hac_t": float("nan"), "p_value": float("nan"), "bootstrap_95_ci": [float("nan"), float("nan")]}
    statistic = float(np.mean(finite) / se)
    return {
        **base,
        "status": "ok",
        "hac_se": float(se),
        "hac_t": statistic,
        "p_value": float(2.0 * norm.sf(abs(statistic))),
        "bootstrap_95_ci": list(moving_block_bootstrap(values, bootstrap_block, bootstrap_samples, seed, dates=dates, bootstrap_indices=bootstrap_indices)),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving missing entries."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return result
    finite = np.clip(values[finite_indices], 0.0, 1.0)
    order = np.argsort(finite, kind="stable")
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    local = np.empty_like(adjusted)
    local[order] = np.minimum(adjusted, 1.0)
    result[finite_indices] = local
    return result
