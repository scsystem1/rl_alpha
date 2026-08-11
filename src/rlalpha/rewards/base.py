from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..factors.calculator import FactorCalculator, daily_corr
from ..factors.records import PoolScore
from ..risk.neutralize import RiskNeutralizer


class RewardObjective(ABC):
    """Train-only pool objective with explicit complete-case semantics.

    Every pool score uses one date-by-asset complete-case mask for every factor
    in that pool and the label.  Risk-neutral objectives additionally fit the
    signal and label projections on that exact same sample.  This is slower
    than treating missing factor values as zeros, but it preserves the
    experiment's statistical meaning.
    """

    def __init__(self, label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None, ridge: float = 1e-3):
        self.label = np.asarray(label, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.exposures = None if exposures is None else np.asarray(exposures, dtype=float)
        self.ridge = float(ridge)
        self.last_neutralization_diagnostics: list[dict[str, object]] = []
        if self.label.shape != self.mask.shape:
            raise ValueError("label and mask shapes differ")
        if self.exposures is not None and self.exposures.shape[:2] != self.label.shape:
            raise ValueError("exposure panel shape mismatch")

    def _complete_case_mask(self, signals: list[np.ndarray], label: np.ndarray) -> np.ndarray:
        common = self.mask & np.isfinite(label)
        for signal in signals:
            values = np.asarray(signal, dtype=float)
            if values.shape != self.label.shape:
                raise ValueError("signal panel shape mismatch")
            common &= np.isfinite(values)
        return common

    def _neutralized_inputs(self, signals: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        common = self._complete_case_mask(signals, self.label) & np.isfinite(self.exposures).all(axis=2)
        calculator = FactorCalculator(self.label, common)
        standardized = [calculator.standardize(signal) for signal in signals]
        for signal in standardized:
            common &= np.isfinite(signal)
        residual_signals = [np.full_like(self.label, np.nan, dtype=float) for _ in signals]
        residual_label = np.full_like(self.label, np.nan, dtype=float)
        neutralizer = RiskNeutralizer()
        diagnostics: list[dict[str, object]] = []
        for day in range(self.label.shape[0]):
            day_mask = common[day]
            if day_mask.sum() <= self.exposures.shape[2]:
                diagnostics.append({
                    "date": str(day),
                    "status": "insufficient_observations",
                    "n_observations": int(day_mask.sum()),
                    "n_columns": int(self.exposures.shape[2]),
                })
                continue
            matrix = np.column_stack([signal[day] for signal in standardized] + [self.label[day]])
            residual, day_diagnostics = neutralizer.residualize_matrix(day, matrix, self.exposures[day], day_mask)
            for index in range(len(signals)):
                residual_signals[index][day] = residual[:, index]
            residual_label[day] = residual[:, -1]
            diagnostics.append({**day_diagnostics[-1], "status": "ok", "joint_columns": len(signals) + 1})
        self.last_neutralization_diagnostics = diagnostics
        return residual_signals, residual_label

    def _daily_ic(self, signals: list[np.ndarray], label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not signals:
            return np.full(self.label.shape[0], np.nan), np.empty(0)
        label = np.asarray(label, dtype=float)
        common = self._complete_case_mask(signals, label)
        calculator = FactorCalculator(label, common)
        prepared = [calculator.standardize(signal) for signal in signals]
        count = len(prepared)
        factor_correlation = np.eye(count)
        predictive = np.zeros(count)
        for left in range(count):
            predictive_daily = daily_corr(prepared[left], label, common)
            if np.isfinite(predictive_daily).any():
                predictive[left] = float(np.nanmean(predictive_daily))
            for right in range(left + 1, count):
                pair_daily = daily_corr(prepared[left], prepared[right], common)
                value = float(np.nanmean(pair_daily)) if np.isfinite(pair_daily).any() else 0.0
                factor_correlation[left, right] = factor_correlation[right, left] = value
        weights = np.linalg.solve(factor_correlation + self.ridge * np.eye(count), predictive)
        combined = np.sum(np.stack(prepared, axis=-1) * weights[None, None, :], axis=-1)
        combined[~common] = np.nan
        return daily_corr(combined, label, common), weights

    @abstractmethod
    def score_pool(self, signals: list[np.ndarray]) -> PoolScore: ...
