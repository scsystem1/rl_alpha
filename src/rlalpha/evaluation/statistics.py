from __future__ import annotations

import numpy as np

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
