from __future__ import annotations

import numpy as np

from .moments import fixed_universe_moments, solve_psd_ridge
from .transform import (
    FactorTransformPipeline,
    IndependentFactorTransformPipeline,
    TransformConfig,
    combine_fixed_signals,
    prepare_fixed_universe_inputs,
)


class RidgeCombiner:
    def __init__(self, ridge: float = 1e-3, pipeline: FactorTransformPipeline | None = None):
        self.ridge = ridge
        self.pipeline = pipeline or IndependentFactorTransformPipeline(
            TransformConfig(
                version="daily-cs-fixed-universe-zero-fill-v2",
                neutralize=False,
            )
        )
        self.weights_: np.ndarray | None = None
        self.last_fit_result = None
        self.moment_diagnostics_: dict[str, float | int] | None = None

    def fit(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None) -> np.ndarray:
        if not signals:
            return np.empty(0)
        transformed = self.pipeline.fit_transform(signals, label, mask, exposures)
        self.last_fit_result = transformed
        if transformed.objective_signals is None or transformed.label is None or transformed.metric_mask is None:
            raise RuntimeError("fit transform did not produce fixed-universe objective inputs")
        self.weights_ = self.fit_prestandardized(
            list(transformed.objective_signals), transformed.label, transformed.metric_mask
        )
        return self.weights_.copy()

    def fit_prestandardized(self, standardized: list[np.ndarray], label: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not standardized:
            return np.empty(0)
        moments = fixed_universe_moments(standardized, label, mask)
        weights, _ = solve_psd_ridge(moments.gram, moments.predictive, self.ridge)
        self.moment_diagnostics_ = {
            "valid_days": moments.valid_days,
            "min_eigenvalue": moments.min_eigenvalue,
            "condition_number": moments.condition_number,
        }
        return weights

    def transform(self, signals: list[np.ndarray], mask: np.ndarray, exposures: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        if self.weights_ is None:
            raise RuntimeError("combiner must be fitted before transform")
        transformed = self.pipeline.transform_portfolio(signals, mask, exposures)
        combined, available = combine_fixed_signals(transformed.signals, self.weights_)
        result_mask = transformed.trade_mask & available
        combined[~result_mask] = np.nan
        return combined, result_mask, transformed.diagnostics

    def transform_metric_composite(
        self,
        signals: list[np.ndarray],
        label: np.ndarray,
        mask: np.ndarray,
        exposures: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        """Create label-free composite first, then project it with the label."""

        if self.weights_ is None:
            raise RuntimeError("combiner must be fitted before transform")
        deployment = self.pipeline.transform_portfolio(signals, mask, exposures)
        composite, _ = combine_fixed_signals(deployment.signals, self.weights_)
        objective, target, metric, diagnostics = prepare_fixed_universe_inputs(
            (composite,), label, deployment.trade_mask, exposures,
            neutralize=self.pipeline.config.neutralize,
        )
        return objective[0], target, metric, deployment.diagnostics + diagnostics

    def to_dict(self) -> dict[str, object]:
        if self.weights_ is None:
            raise RuntimeError("cannot serialize an unfitted combiner")
        return {
            "ridge": self.ridge,
            "weights": self.weights_.tolist(),
            "pipeline": self.pipeline.to_dict(),
            "moment_diagnostics": self.moment_diagnostics_,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RidgeCombiner":
        pipeline = FactorTransformPipeline.from_dict(value["pipeline"])  # type: ignore[arg-type]
        combiner = cls(float(value["ridge"]), pipeline)
        combiner.weights_ = np.asarray(value["weights"], dtype=float)
        diagnostics = value.get("moment_diagnostics")
        combiner.moment_diagnostics_ = diagnostics if isinstance(diagnostics, dict) else None
        return combiner
