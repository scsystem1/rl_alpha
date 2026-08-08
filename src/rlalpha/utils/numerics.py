from __future__ import annotations

import numpy as np


def finite_corr(a: np.ndarray, b: np.ndarray, min_count: int = 3) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_count:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def winsorize_zscore(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return result
    low, high = np.quantile(values[mask], [lower, upper])
    clipped = np.clip(values[mask], low, high)
    std = clipped.std(ddof=0)
    if std <= 1e-12:
        return result
    result[mask] = (clipped - clipped.mean()) / std
    return result

