from __future__ import annotations

from abc import ABC, abstractmethod
import weakref

import numpy as np

from ..factors.calculator import FactorCalculator
from ..factors.records import PoolScore
from ..risk.neutralize import RiskNeutralizer


class RewardObjective(ABC):
    def __init__(self, label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None, ridge: float = 1e-3):
        self.label = np.asarray(label, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.exposures = None if exposures is None else np.asarray(exposures, dtype=float)
        self.ridge = float(ridge)
        self._standardized_cache: dict[int, tuple[weakref.ReferenceType[np.ndarray], np.ndarray]] = {}
        self._residual_cache: dict[int, tuple[weakref.ReferenceType[np.ndarray], np.ndarray]] = {}
        self._residual_label_cache: np.ndarray | None = None
        self._prepared_cache: dict[int, tuple[weakref.ReferenceType[np.ndarray], np.ndarray]] = {}
        self._moment_cache: dict[int, tuple[weakref.ReferenceType[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._cross_cache: dict[tuple[int, int], tuple[weakref.ReferenceType[np.ndarray], weakref.ReferenceType[np.ndarray], np.ndarray]] = {}
        if self.label.shape != self.mask.shape:
            raise ValueError("label and mask shapes differ")
        self._moment_label: np.ndarray | None = None
        self._set_moment_label(self.label)

    def _set_moment_label(self, label: np.ndarray) -> None:
        if self._moment_label is label:
            return
        self._moment_label = label
        self._moment_cache.clear()
        self._cross_cache.clear()
        self._fixed_common = self.mask & np.isfinite(label)
        self._fixed_count = self._fixed_common.sum(axis=1).astype(float)
        fixed_label = np.where(self._fixed_common, label, 0.0)
        self._fixed_label_sum = fixed_label.sum(axis=1)
        self._fixed_label_square = np.square(fixed_label).sum(axis=1)

    @staticmethod
    def _store_weak(cache: dict, key: object, source: np.ndarray, value: object) -> tuple[weakref.ReferenceType[np.ndarray], object]:
        reference = weakref.ref(source, lambda _: cache.pop(key, None))
        record = (reference, value)
        cache[key] = record
        return record

    def _neutralized_inputs(self, signals: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        if self.exposures.shape[:2] != self.label.shape:
            raise ValueError("exposure panel shape mismatch")
        calculator = FactorCalculator(self.label, self.mask)
        standardized = []
        for signal in signals:
            key = id(signal)
            cached = self._standardized_cache.get(key)
            if cached is None or cached[0]() is not signal:
                cached = self._store_weak(self._standardized_cache, key, signal, calculator.standardize(signal))
            standardized.append(cached[1])
        neutralizer = RiskNeutralizer()
        if self._residual_label_cache is None:
            residual_label = np.full_like(self.label, np.nan)
            for day in range(self.label.shape[0]):
                common = self.mask[day] & np.isfinite(self.label[day]) & np.isfinite(self.exposures[day]).all(axis=1)
                if common.sum() > self.exposures.shape[2]:
                    residual_label[day], _ = neutralizer.residualize_vector(day, self.label[day], self.exposures[day], common)
            self._residual_label_cache = residual_label
        residual_signals = []
        for source, signal in zip(signals, standardized, strict=True):
            key = id(source)
            cached = self._residual_cache.get(key)
            if cached is None or cached[0]() is not source:
                residual = np.full_like(self.label, np.nan)
                for day in range(self.label.shape[0]):
                    common = self.mask[day] & np.isfinite(signal[day]) & np.isfinite(self.exposures[day]).all(axis=1)
                    if common.sum() > self.exposures.shape[2]:
                        residual[day], _ = neutralizer.residualize_vector(day, signal[day], self.exposures[day], common)
                cached = self._store_weak(self._residual_cache, key, source, residual)
            residual_signals.append(cached[1])
        return residual_signals, self._residual_label_cache

    def _moments(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = id(signal)
        cached = self._moment_cache.get(key)
        if cached is None or cached[0]() is not signal:
            filled = np.where(self._fixed_common & np.isfinite(signal), signal, 0.0)
            value = (filled.sum(axis=1), np.square(filled).sum(axis=1), (filled * np.where(self._fixed_common, self._moment_label, 0.0)).sum(axis=1), filled)
            cached = self._store_weak(self._moment_cache, key, signal, value)
        return cached[1]

    def _cross(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left is right:
            return self._moments(left)[1]
        key = tuple(sorted((id(left), id(right))))
        cached = self._cross_cache.get(key)
        first_source, second_source = (None, None) if cached is None else (cached[0](), cached[1]())
        if cached is None or not ((first_source is left and second_source is right) or (first_source is right and second_source is left)):
            value = np.einsum("ij,ij->i", self._moments(left)[3], self._moments(right)[3], optimize=True)
            first = weakref.ref(left, lambda _: self._cross_cache.pop(key, None))
            second = weakref.ref(right, lambda _: self._cross_cache.pop(key, None))
            cached = (first, second, value)
            self._cross_cache[key] = cached
        return cached[2]

    def _correlation(self, sum_x: np.ndarray, square_x: np.ndarray, cross_xy: np.ndarray) -> np.ndarray:
        count, sum_y, square_y = self._fixed_count, self._fixed_label_sum, self._fixed_label_square
        with np.errstate(all="ignore"):
            covariance = cross_xy - sum_x * sum_y / count
            variance_x = square_x - np.square(sum_x) / count
            variance_y = square_y - np.square(sum_y) / count
            result = covariance / np.sqrt(variance_x * variance_y)
        result[(count < 3) | (variance_x <= 1e-24) | (variance_y <= 1e-24)] = np.nan
        return result

    def _daily_ic(self, signals: list[np.ndarray], label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not signals:
            return np.full(self.label.shape[0], np.nan), np.empty(0)
        self._set_moment_label(label)
        calculator = FactorCalculator(label, self.mask)
        prepared = []
        for signal in signals:
            key = id(signal)
            cached = self._prepared_cache.get(key)
            if cached is None or cached[0]() is not signal:
                cached = self._store_weak(self._prepared_cache, key, signal, calculator.standardize(signal))
            prepared.append(cached[1])
        count = len(prepared)
        factor_correlation = np.eye(count)
        predictive = np.empty(count)
        moments = [self._moments(signal) for signal in prepared]
        for left in range(count):
            predictive_daily = self._correlation(moments[left][0], moments[left][1], moments[left][2])
            predictive[left] = float(np.nanmean(predictive_daily)) if np.isfinite(predictive_daily).any() else 0.0
            for right in range(left + 1, count):
                cross = self._cross(prepared[left], prepared[right])
                sx, sx2 = moments[left][0], moments[left][1]
                sy, sy2 = moments[right][0], moments[right][1]
                with np.errstate(all="ignore"):
                    covariance = cross - sx * sy / self._fixed_count
                    variance_left = sx2 - np.square(sx) / self._fixed_count
                    variance_right = sy2 - np.square(sy) / self._fixed_count
                    pair_daily = covariance / np.sqrt(variance_left * variance_right)
                pair_daily[(self._fixed_count < 3) | (variance_left <= 1e-24) | (variance_right <= 1e-24)] = np.nan
                value = float(np.nanmean(pair_daily)) if np.isfinite(pair_daily).any() else 0.0
                factor_correlation[left, right] = factor_correlation[right, left] = value
        weights = np.linalg.solve(factor_correlation + self.ridge * np.eye(count), np.nan_to_num(predictive))
        sums = np.stack([item[0] for item in moments], axis=1)
        label_cross = np.stack([item[2] for item in moments], axis=1)
        gram = np.empty((self.label.shape[0], count, count))
        for left in range(count):
            for right in range(left, count):
                gram[:, left, right] = gram[:, right, left] = self._cross(prepared[left], prepared[right])
        combined_sum = sums @ weights
        combined_square = np.einsum("i,tij,j->t", weights, gram, weights, optimize=True)
        combined_label_cross = label_cross @ weights
        daily = self._correlation(combined_sum, combined_square, combined_label_cross)
        return daily, weights

    @abstractmethod
    def score_pool(self, signals: list[np.ndarray]) -> PoolScore: ...
