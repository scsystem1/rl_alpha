from __future__ import annotations

import numpy as np

from .calculator import FactorCalculator
from ..utils.numerics import finite_corr


class RidgeCombiner:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not signals:
            return np.empty(0)
        calculator = FactorCalculator(label, mask)
        standardized = [calculator.standardize(signal) for signal in signals]
        n_factors = len(signals)
        correlation = np.eye(n_factors)
        for left in range(n_factors):
            for right in range(left + 1, n_factors):
                daily = [finite_corr(standardized[left][day][mask[day]], standardized[right][day][mask[day]]) for day in range(mask.shape[0])]
                value = float(np.nanmean(daily)) if np.isfinite(daily).any() else 0.0
                correlation[left, right] = correlation[right, left] = value
        predictive = np.asarray([calculator.single_ic(signal, label) for signal in signals])
        predictive = np.nan_to_num(predictive)
        return np.linalg.solve(correlation + self.ridge * np.eye(n_factors), predictive)

