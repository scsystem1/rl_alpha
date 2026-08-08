from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from ..utils.numerics import finite_corr, winsorize_zscore


@dataclass(frozen=True)
class FactorSignal:
    values: np.ndarray
    expr_hash: str = ""
    expression: str = ""


class FactorCalculator:
    def __init__(self, label: np.ndarray, mask: np.ndarray):
        self.label = np.asarray(label, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        if self.label.shape != self.mask.shape:
            raise ValueError("label and mask shapes differ")

    def standardize(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        if signal.shape != self.mask.shape:
            raise ValueError("signal shape differs from calculator panel")
        values = np.where(self.mask & np.isfinite(signal), signal, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nanquantile(values, 0.01, axis=1)
            high = np.nanquantile(values, 0.99, axis=1)
            clipped = np.clip(values, low[:, None], high[:, None])
            mean = np.nanmean(clipped, axis=1)
            std = np.nanstd(clipped, axis=1)
            result = (clipped - mean[:, None]) / std[:, None]
        result[~self.mask | ~np.isfinite(result) | (std <= 1e-12)[:, None]] = np.nan
        return result

    def ic_series(self, signal: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
        standardized = self.standardize(signal)
        target = self.label if label is None else np.asarray(label, dtype=float)
        return daily_corr(standardized, target, self.mask)

    def single_ic(self, signal: np.ndarray, label: np.ndarray | None = None) -> float:
        values = self.ic_series(signal, label)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def mutual_ic(self, first: np.ndarray, second: np.ndarray) -> float:
        left, right = self.standardize(first), self.standardize(second)
        values = [finite_corr(left[day][self.mask[day]], right[day][self.mask[day]]) for day in range(len(left))]
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def pool_ic(self, signals: list[np.ndarray], weights: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
        if not signals:
            return np.zeros(self.label.shape[0])
        standardized = np.stack([self.standardize(signal) for signal in signals], axis=-1)
        combined = np.nansum(standardized * np.asarray(weights)[None, None, :], axis=-1)
        all_missing = ~np.isfinite(standardized).any(axis=-1)
        combined[all_missing] = np.nan
        return self.ic_series(combined, label)

    def pool_ic_prestandardized(self, signals: list[np.ndarray], weights: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
        if not signals:
            return np.zeros(self.label.shape[0])
        combined = np.zeros(self.label.shape, dtype=float)
        available = np.zeros(self.label.shape, dtype=bool)
        for signal, weight in zip(signals, weights, strict=True):
            finite = np.isfinite(signal)
            combined += np.where(finite, signal, 0.0) * weight
            available |= finite
        combined[~available] = np.nan
        return self.ic_series(combined, label)


def daily_corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    left, right, mask = np.asarray(left, float), np.asarray(right, float), np.asarray(mask, bool)
    common = mask & np.isfinite(left) & np.isfinite(right)
    count = common.sum(axis=1)
    x = np.where(common, left, 0.0)
    y = np.where(common, right, 0.0)
    with np.errstate(all="ignore"):
        sx, sy = x.sum(axis=1), y.sum(axis=1)
        covariance = (x * y).sum(axis=1) - sx * sy / count
        variance_x = (x * x).sum(axis=1) - sx * sx / count
        variance_y = (y * y).sum(axis=1) - sy * sy / count
        result = covariance / np.sqrt(variance_x * variance_y)
    result[(count < 3) | (variance_x <= 1e-24) | (variance_y <= 1e-24)] = np.nan
    return result
