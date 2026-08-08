from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NeutralizationDiagnostics:
    n_observations: int
    n_columns: int
    rank: int
    condition_number: float
    ridge: float
    max_residual_exposure: float


class RiskNeutralizer:
    def __init__(self, condition_threshold: float = 1e12, ridge: float = 1e-10):
        self.condition_threshold = condition_threshold
        self.ridge = ridge

    def _solve(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        q, r = np.linalg.qr(x, mode="reduced")
        rank = int(np.linalg.matrix_rank(r))
        condition = float(np.linalg.cond(r))
        ridge_used = 0.0
        if rank < x.shape[1] or condition > self.condition_threshold:
            ridge_used = self.ridge
            coefficient = np.linalg.solve(x.T @ x + ridge_used * np.eye(x.shape[1]), x.T @ y)
        else:
            coefficient = np.linalg.solve(r, q.T @ y)
        return coefficient, rank, condition, ridge_used

    def residualize_vector(self, date: object, values: np.ndarray, exposures: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        values = np.asarray(values, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        mask = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(exposures).all(axis=1)
        result = np.full_like(values, np.nan)
        x, y = exposures[mask], values[mask]
        if len(y) <= x.shape[1]:
            raise ValueError(f"insufficient observations on {date}: {len(y)} for {x.shape[1]} exposures")
        coefficient, rank, condition, ridge_used = self._solve(x, y)
        residual = y - x @ coefficient
        result[mask] = residual
        max_exposure = float(np.max(np.abs(x.T @ residual / len(residual))))
        diagnostics = NeutralizationDiagnostics(len(y), x.shape[1], rank, condition, ridge_used, max_exposure)
        return result, {"date": str(date), **asdict(diagnostics)}

    def residualize_matrix(self, date: object, factor_matrix: np.ndarray, exposures: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
        factors = np.asarray(factor_matrix, dtype=float)
        if factors.ndim == 1:
            residual, diagnostics = self.residualize_vector(date, factors, exposures, mask)
            return residual[:, None], [diagnostics]
        exposures = np.asarray(exposures, dtype=float)
        common = np.asarray(mask, dtype=bool) & np.isfinite(factors).all(axis=1) & np.isfinite(exposures).all(axis=1)
        result = np.full_like(factors, np.nan)
        x, y = exposures[common], factors[common]
        if len(y) <= x.shape[1]:
            raise ValueError(f"insufficient observations on {date}: {len(y)} for {x.shape[1]} exposures")
        coefficient, rank, condition, ridge_used = self._solve(x, y)
        residual = y - x @ coefficient
        result[common] = residual
        diagnostics = []
        for column in range(factors.shape[1]):
            max_exposure = float(np.max(np.abs(x.T @ residual[:, column] / len(residual))))
            record = NeutralizationDiagnostics(len(y), x.shape[1], rank, condition, ridge_used, max_exposure)
            diagnostics.append({"date": str(date), **asdict(record)})
        return result, diagnostics
