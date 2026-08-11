from __future__ import annotations

import numpy as np

from .calculator import FactorCalculator, daily_corr
from .transform import FactorTransformPipeline, TransformConfig


class RidgeCombiner:
    def __init__(self, ridge: float = 1e-3, pipeline: FactorTransformPipeline | None = None):
        self.ridge = ridge
        self.pipeline = pipeline or FactorTransformPipeline(TransformConfig(neutralize=False))
        self.weights_: np.ndarray | None = None
        self.last_fit_result = None

    def fit(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None) -> np.ndarray:
        if not signals:
            return np.empty(0)
        transformed = self.pipeline.fit_transform(signals, label, mask, exposures)
        self.last_fit_result = transformed
        assert transformed.label is not None
        self.weights_ = self.fit_prestandardized(list(transformed.signals), transformed.label, transformed.mask)
        return self.weights_.copy()

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

    def transform(self, signals: list[np.ndarray], mask: np.ndarray, exposures: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        if self.weights_ is None:
            raise RuntimeError("combiner must be fitted before transform")
        transformed = self.pipeline.transform_portfolio(signals, mask, exposures)
        stacked = np.stack(transformed.signals, axis=-1)
        combined = np.sum(stacked * self.weights_[None, None, :], axis=-1)
        combined[~transformed.mask] = np.nan
        return combined, transformed.mask, transformed.diagnostics

    def to_dict(self) -> dict[str, object]:
        if self.weights_ is None:
            raise RuntimeError("cannot serialize an unfitted combiner")
        return {"ridge": self.ridge, "weights": self.weights_.tolist(), "pipeline": self.pipeline.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RidgeCombiner":
        pipeline = FactorTransformPipeline.from_dict(value["pipeline"])  # type: ignore[arg-type]
        combiner = cls(float(value["ridge"]), pipeline)
        combiner.weights_ = np.asarray(value["weights"], dtype=float)
        return combiner
