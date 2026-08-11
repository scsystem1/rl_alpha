from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..rewards.statistics import newey_west_mean_se


def moving_block_bootstrap(values: np.ndarray, block: int = 20, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    block = min(max(1, block), len(values))
    starts = np.arange(len(values) - block + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    n_blocks = int(np.ceil(len(values) / block))
    for index in range(samples):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        draw = np.concatenate([values[start : start + block] for start in chosen])[: len(values)]
        means[index] = draw.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def series_summary(values: np.ndarray, hac_lag: int = 20, bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, float | list[float] | int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "hac_se": float("nan"), "hac_t": float("nan"), "bootstrap_95_ci": [float("nan"), float("nan")]}
    mean = float(finite.mean())
    se = newey_west_mean_se(finite, min(hac_lag, max(0, len(finite) - 1)))
    return {
        "n": int(len(finite)),
        "mean": mean,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "hac_se": se,
        "hac_t": mean / se if np.isfinite(se) and se > 0 else float("nan"),
        "bootstrap_95_ci": list(moving_block_bootstrap(finite, 20, bootstrap_samples, seed)),
    }


def paired_summary(first: np.ndarray, second: np.ndarray, **kwargs: int) -> dict[str, float | list[float] | int]:
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
) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
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
    se = newey_west_mean_se(finite, min(hac_lag, len(finite) - 1))
    if not np.isfinite(se) or se <= 0:
        return {**base, "status": "insufficient_data", "hac_se": se, "hac_t": float("nan"), "p_value": float("nan"), "bootstrap_95_ci": [float("nan"), float("nan")]}
    statistic = float(np.mean(finite) / se)
    return {
        **base,
        "status": "ok",
        "hac_se": float(se),
        "hac_t": statistic,
        "p_value": float(2.0 * norm.sf(abs(statistic))),
        "bootstrap_95_ci": list(moving_block_bootstrap(finite, bootstrap_block, bootstrap_samples, seed)),
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
