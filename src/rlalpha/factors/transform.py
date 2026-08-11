from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .calculator import FactorCalculator
from ..risk.neutralize import RiskNeutralizer


@dataclass(frozen=True)
class TransformConfig:
    version: str = "daily-cs-joint-mask-v1"
    neutralize: bool = True
    post_residual_standardize: bool = True


@dataclass(frozen=True)
class TransformResult:
    signals: tuple[np.ndarray, ...]
    label: np.ndarray | None
    mask: np.ndarray
    diagnostics: tuple[dict[str, Any], ...]


class FactorTransformPipeline:
    """Serializable, identical fit/out-of-sample factor transformations.

    Winsorization and z-scoring are explicitly daily cross-sectional
    operations, so they intentionally use each date's eligible cross-section;
    there are no global moments to re-estimate on validation or test.  All
    learned choices are represented by :class:`TransformConfig` and frozen at
    fit time.
    """

    def __init__(self, config: TransformConfig | None = None):
        self.config = config or TransformConfig()
        self.fitted = False
        self.n_factors: int | None = None

    @staticmethod
    def _validate(signals: list[np.ndarray], mask: np.ndarray, label: np.ndarray | None, exposures: np.ndarray | None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None, np.ndarray | None]:
        mask = np.asarray(mask, dtype=bool)
        values = [np.asarray(signal, dtype=float) for signal in signals]
        if not values:
            raise ValueError("at least one factor signal is required")
        if any(signal.shape != mask.shape for signal in values):
            raise ValueError("factor signal and mask shapes differ")
        target = None if label is None else np.asarray(label, dtype=float)
        if target is not None and target.shape != mask.shape:
            raise ValueError("label and mask shapes differ")
        risk = None if exposures is None else np.asarray(exposures, dtype=float)
        if risk is not None and risk.shape[:2] != mask.shape:
            raise ValueError("exposure panel shape differs")
        return values, mask, target, risk

    def _apply(self, signals: list[np.ndarray], mask: np.ndarray, label: np.ndarray | None, exposures: np.ndarray | None) -> TransformResult:
        signals, mask, label, exposures = self._validate(signals, mask, label, exposures)
        common = mask.copy()
        for signal in signals:
            common &= np.isfinite(signal)
        if label is not None:
            common &= np.isfinite(label)
        if self.config.neutralize:
            if exposures is None:
                raise ValueError("neutralized transform requires exposures")
            common &= np.isfinite(exposures).all(axis=2)
        dummy_label = np.zeros(common.shape) if label is None else label
        calculator = FactorCalculator(dummy_label, common)
        prepared = [calculator.standardize(signal) for signal in signals]
        # Winsorization/z-scoring can invalidate an otherwise complete day
        # (for example a constant cross-section).  Projection must use the
        # transformed complete-case mask, not the pre-transform raw mask.
        for signal in prepared:
            common &= np.isfinite(signal)
        target = None if label is None else label.copy()
        diagnostics: list[dict[str, Any]] = []
        if self.config.neutralize:
            assert exposures is not None
            neutralizer = RiskNeutralizer()
            residual_signals = [np.full(common.shape, np.nan) for _ in signals]
            residual_target = None if label is None else np.full(common.shape, np.nan)
            for day in range(common.shape[0]):
                day_mask = common[day]
                if day_mask.sum() <= exposures.shape[2]:
                    diagnostics.append({"date": str(day), "status": "insufficient_observations", "n_observations": int(day_mask.sum()), "n_columns": int(exposures.shape[2])})
                    common[day] = False
                    continue
                columns = [signal[day] for signal in prepared]
                if label is not None:
                    columns.append(label[day])
                residual, day_diagnostics = neutralizer.residualize_matrix(day, np.column_stack(columns), exposures[day], day_mask)
                for factor_index in range(len(signals)):
                    residual_signals[factor_index][day] = residual[:, factor_index]
                if residual_target is not None:
                    residual_target[day] = residual[:, -1]
                diagnostics.append({**day_diagnostics[-1], "status": "ok", "joint_columns": len(columns)})
            prepared = residual_signals
            target = residual_target
        if self.config.post_residual_standardize:
            post_label = np.zeros(common.shape) if target is None else target
            post = FactorCalculator(post_label, common)
            prepared = [post.standardize(signal) for signal in prepared]
            if target is not None:
                target = post.standardize(target)
            for signal in prepared:
                common &= np.isfinite(signal)
            if target is not None:
                common &= np.isfinite(target)
        for signal in prepared:
            signal[~common] = np.nan
        if target is not None:
            target[~common] = np.nan
        return TransformResult(tuple(prepared), target, common, tuple(diagnostics))

    def fit_transform(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None) -> TransformResult:
        self.fitted = True
        self.n_factors = len(signals)
        return self._apply(signals, mask, label, exposures)

    def transform_ic(self, signals: list[np.ndarray], label: np.ndarray, mask: np.ndarray, exposures: np.ndarray | None = None) -> TransformResult:
        self._assert_fitted(signals)
        return self._apply(signals, mask, label, exposures)

    def transform_portfolio(self, signals: list[np.ndarray], mask: np.ndarray, exposures: np.ndarray | None = None) -> TransformResult:
        self._assert_fitted(signals)
        return self._apply(signals, mask, None, exposures)

    def _assert_fitted(self, signals: list[np.ndarray]) -> None:
        if not self.fitted or self.n_factors is None:
            raise RuntimeError("transform pipeline must be fitted before out-of-sample use")
        if len(signals) != self.n_factors:
            raise ValueError(f"expected {self.n_factors} factors, received {len(signals)}")

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted transform pipeline")
        return {"config": asdict(self.config), "fitted": True, "n_factors": self.n_factors}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactorTransformPipeline":
        pipeline = cls(TransformConfig(**value["config"]))
        pipeline.fitted = bool(value.get("fitted"))
        pipeline.n_factors = int(value["n_factors"])
        return pipeline
