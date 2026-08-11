from __future__ import annotations

import math

import numpy as np


def newey_west_mean_se(values: np.ndarray, lag: int = 20) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < 2:
        return float("nan")
    centered = values - values.mean()
    max_lag = min(lag, count - 1)
    long_run_variance = float(centered @ centered / count)
    for offset in range(1, max_lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / count)
        weight = 1.0 - offset / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    return math.sqrt(max(0.0, long_run_variance) / count)


def lcb_score(values: np.ndarray, lag: int = 20, critical_value: float = 1.645) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    standard_error = newey_west_mean_se(finite, lag=lag)
    return mean - critical_value * standard_error, mean, standard_error
