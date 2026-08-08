from __future__ import annotations

from dataclasses import dataclass

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
        result = np.full_like(signal, np.nan)
        for day in range(signal.shape[0]):
            eligible = self.mask[day]
            standardized = winsorize_zscore(np.where(eligible, signal[day], np.nan))
            result[day, eligible] = standardized[eligible]
        return result

    def ic_series(self, signal: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
        standardized = self.standardize(signal)
        target = self.label if label is None else np.asarray(label, dtype=float)
        return np.asarray([finite_corr(standardized[day][self.mask[day]], target[day][self.mask[day]]) for day in range(len(standardized))])

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

