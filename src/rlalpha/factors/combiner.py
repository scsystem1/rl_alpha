from __future__ import annotations

import numpy as np

from .calculator import FactorCalculator, daily_corr
from ..utils.numerics import finite_corr


class RidgeCombiner:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not signals:
            return np.empty(0)
        calculator = FactorCalculator(label, mask)
        standardized = [calculator.standardize(signal) for signal in signals]
        return self.fit_prestandardized(standardized, label, mask)

    def fit_prestandardized(self, standardized: list[np.ndarray], label: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not standardized:
            return np.empty(0)
        n_factors = len(standardized)
        correlation = np.eye(n_factors)
        for left in range(n_factors):
            for right in range(left + 1, n_factors):
                daily = daily_corr(standardized[left], standardized[right], mask)
                value = float(np.nanmean(daily)) if np.isfinite(daily).any() else 0.0
                correlation[left, right] = correlation[right, left] = value
        predictive = np.asarray([np.nanmean(values) if np.isfinite(values).any() else 0.0 for values in (daily_corr(signal, label, mask) for signal in standardized)])
        predictive = np.nan_to_num(predictive)
        return np.linalg.solve(correlation + self.ridge * np.eye(n_factors), predictive)
