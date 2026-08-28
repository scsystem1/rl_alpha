from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedUniverseMoments:
    gram: np.ndarray
    predictive: np.ndarray
    valid_days: int
    min_eigenvalue: float
    condition_number: float


def fixed_universe_moments(
    signals: tuple[np.ndarray, ...] | list[np.ndarray],
    label: np.ndarray,
    metric_mask: np.ndarray,
) -> FixedUniverseMoments:
    """Estimate day-equal factor Gram and factor/label cross moments.

    Inputs must already represent the objective space.  Missing factor values
    are zero opinions on the fixed metric universe, never pairwise deletions.
    """

    count = len(signals)
    target = np.asarray(label, dtype=float)
    mask = np.asarray(metric_mask, dtype=bool) & np.isfinite(target)
    if target.shape != mask.shape:
        raise ValueError("label and metric mask shapes differ")
    if not count:
        empty = np.empty((0, 0), dtype=float)
        return FixedUniverseMoments(empty, np.empty(0), 0, float("nan"), float("nan"))
    matrix = np.stack([np.asarray(signal, dtype=float) for signal in signals], axis=-1)
    if matrix.shape[:2] != mask.shape:
        raise ValueError("signal and metric mask shapes differ")
    matrix = np.where(mask[..., None], np.where(np.isfinite(matrix), matrix, 0.0), 0.0)
    y = np.where(mask, target, 0.0)
    observations = mask.sum(axis=1).astype(float)
    valid_days = observations >= 3
    if not valid_days.any():
        raise ValueError("fixed-universe moments have no valid dates")
    scale = np.zeros_like(observations)
    scale[valid_days] = 1.0 / observations[valid_days]
    daily_gram = np.einsum("daf,dag,d->dfg", matrix, matrix, scale, optimize=True)
    daily_predictive = np.einsum("daf,da,d->df", matrix, y, scale, optimize=True)
    gram = np.mean(daily_gram[valid_days], axis=0)
    # Eliminate asymmetric floating-point accumulation before eigensolving.
    gram = 0.5 * (gram + gram.T)
    predictive = np.mean(daily_predictive[valid_days], axis=0)
    if not np.isfinite(gram).all() or not np.isfinite(predictive).all():
        raise ValueError("fixed-universe moments are non-finite")
    eigenvalues = np.linalg.eigvalsh(gram)
    scale_bound = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -1e-10 * scale_bound:
        raise ValueError("fixed-universe Gram matrix is not positive semidefinite")
    positive = eigenvalues[eigenvalues > 1e-12 * scale_bound]
    condition = (
        float(positive[-1] / positive[0]) if len(positive) else float("inf")
    )
    return FixedUniverseMoments(
        gram,
        predictive,
        int(valid_days.sum()),
        float(eigenvalues[0]),
        condition,
    )


def solve_psd_ridge(
    gram: np.ndarray,
    predictive: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a finite PSD Gram ridge system and retain its inverse for saliency."""

    gram = np.asarray(gram, dtype=float)
    predictive = np.asarray(predictive, dtype=float)
    if gram.shape != (len(predictive), len(predictive)):
        raise ValueError("Gram and predictive dimensions differ")
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    system = 0.5 * (gram + gram.T) + ridge * np.eye(len(predictive))
    if not np.isfinite(system).all() or not np.isfinite(predictive).all():
        raise ValueError("ridge system is non-finite")
    try:
        np.linalg.cholesky(system)
        weights = np.linalg.solve(system, predictive)
        inverse = np.linalg.solve(system, np.eye(len(predictive)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("PSD ridge system is not positive definite") from exc
    return weights, inverse
