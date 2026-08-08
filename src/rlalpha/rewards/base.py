from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..factors.calculator import FactorCalculator
from ..factors.combiner import RidgeCombiner
from ..factors.records import PoolScore
from ..risk.neutralize import RiskNeutralizer


class RewardObjective(ABC):
    def __init__(self, label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None, ridge: float = 1e-3):
        self.label = np.asarray(label, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.exposures = None if exposures is None else np.asarray(exposures, dtype=float)
        self.combiner = RidgeCombiner(ridge)
        if self.label.shape != self.mask.shape:
            raise ValueError("label and mask shapes differ")

    def _neutralized_inputs(self, signals: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        if self.exposures.shape[:2] != self.label.shape:
            raise ValueError("exposure panel shape mismatch")
        calculator = FactorCalculator(self.label, self.mask)
        standardized = [calculator.standardize(signal) for signal in signals]
        neutralizer = RiskNeutralizer()
        residual_signals = [np.full_like(self.label, np.nan) for _ in signals]
        residual_label = np.full_like(self.label, np.nan)
        for day in range(self.label.shape[0]):
            day_mask = self.mask[day] & np.isfinite(self.exposures[day]).all(axis=1)
            values = np.column_stack([self.label[day], *[signal[day] for signal in standardized]])
            common = day_mask & np.isfinite(values).all(axis=1)
            if common.sum() <= self.exposures.shape[2]:
                continue
            residuals, _ = neutralizer.residualize_matrix(day, values, self.exposures[day], common)
            residual_label[day] = residuals[:, 0]
            for index, signal in enumerate(standardized):
                residual_signals[index][day] = residuals[:, index + 1]
        return residual_signals, residual_label

    def _daily_ic(self, signals: list[np.ndarray], label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not signals:
            return np.full(self.label.shape[0], np.nan), np.empty(0)
        weights = self.combiner.fit(signals, label, self.mask)
        daily = FactorCalculator(label, self.mask).pool_ic(signals, weights, label)
        return daily, weights

    @abstractmethod
    def score_pool(self, signals: list[np.ndarray]) -> PoolScore: ...
