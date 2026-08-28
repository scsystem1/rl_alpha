from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import threading

import numpy as np
from numba import njit, prange

from ..factors.calculator import FactorCalculator, daily_corr
from ..factors.records import PoolScore
from ..factors.transform import combine_available_signals
from ..risk.neutralize import PreparedRiskSolve, RiskNeutralizer


def _daily_corr_columns(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Daily correlations for two column batches on one exact support."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 2:
        left = left[..., None]
    if right.ndim == 2:
        right = right[..., None]
    mask = np.asarray(mask, dtype=bool)
    if left.shape[:2] != mask.shape or right.shape[:2] != mask.shape:
        raise ValueError("batched correlation panel shape mismatch")
    left_valid = mask[..., None] & np.isfinite(left)
    right_valid = mask[..., None] & np.isfinite(right)
    left_presence = left_valid.astype(float)
    right_presence = right_valid.astype(float)
    x = np.where(left_valid, left, 0.0)
    y = np.where(right_valid, right, 0.0)
    with np.errstate(all="ignore"):
        count = np.einsum("dal,dar->dlr", left_presence, right_presence, optimize=True)
        sx = np.einsum("dal,dar->dlr", x, right_presence, optimize=True)
        sy = np.einsum("dal,dar->dlr", left_presence, y, optimize=True)
        cross = np.einsum("dal,dar->dlr", x, y, optimize=True)
        covariance = cross - sx * sy / count
        variance_x = np.einsum("dal,dar->dlr", x * x, right_presence, optimize=True) - sx * sx / count
        variance_y = np.einsum("dal,dar->dlr", left_presence, y * y, optimize=True) - sy * sy / count
        result = covariance / np.sqrt(variance_x * variance_y)
    result[(count < 3) | (variance_x <= 1e-24) | (variance_y <= 1e-24)] = np.nan
    return result


@njit(cache=True, inline="always")
def _masked_corr_one_day(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float:
    count = 0
    sx = 0.0
    sy = 0.0
    cross = 0.0
    square_x = 0.0
    square_y = 0.0
    for asset in range(len(mask)):
        x = left[asset]
        y = right[asset]
        if mask[asset] and np.isfinite(x) and np.isfinite(y):
            count += 1
            sx += x
            sy += y
            cross += x * y
            square_x += x * x
            square_y += y * y
    if count < 3:
        return np.nan
    covariance = cross - sx * sy / count
    variance_x = square_x - sx * sx / count
    variance_y = square_y - sy * sy / count
    if variance_x <= 1e-24 or variance_y <= 1e-24:
        return np.nan
    return covariance / np.sqrt(variance_x * variance_y)


@njit(cache=True, parallel=True, nogil=True)
def _daily_factor_moments_kernel(
    matrix: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    days, _, factors = matrix.shape
    predictive = np.full((days, factors), np.nan, dtype=np.float64)
    pairwise = np.full((days, factors, factors), np.nan, dtype=np.float64)
    for day in prange(days):
        for left in range(factors):
            predictive[day, left] = _masked_corr_one_day(
                matrix[day, :, left], label[day], mask[day]
            )
            pairwise[day, left, left] = 1.0
            for right in range(left + 1, factors):
                value = _masked_corr_one_day(
                    matrix[day, :, left], matrix[day, :, right], mask[day]
                )
                pairwise[day, left, right] = value
                pairwise[day, right, left] = value
    return predictive, pairwise


def _factor_moments_batched(
    signals: list[np.ndarray],
    label: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean daily factor/label and factor/factor correlations in one panel pass.

    The old implementation invoked ``daily_corr`` once per factor pair.  For a
    20-factor pool that meant 210 NumPy allocations and complete panel scans.
    This fused Numba kernel walks each day once and parallelizes days without
    launching hundreds of nested parallel regions.
    """
    count = len(signals)
    correlation = np.eye(count, dtype=float)
    predictive = np.zeros(count, dtype=float)
    if not count:
        return correlation, predictive
    label = np.asarray(label, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    matrix = np.ascontiguousarray(np.stack(signals, axis=-1), dtype=np.float64)
    predictive_daily, pair_daily = _daily_factor_moments_kernel(
        matrix, label, mask
    )
    with np.errstate(all="ignore"):
        predictive_means = np.nanmean(predictive_daily, axis=0)
        pair_means = np.nanmean(pair_daily, axis=0)
    predictive[:] = np.where(np.isfinite(predictive_means), predictive_means, 0.0)
    pair_means = np.where(np.isfinite(pair_means), pair_means, 0.0)
    off_diagonal = ~np.eye(count, dtype=bool)
    correlation[off_diagonal] = pair_means[off_diagonal]
    return correlation, predictive


@dataclass(frozen=True)
class PreparedPoolState:
    """Reusable sufficient state for scoring one frozen complete-case pool."""

    raw_signals: tuple[np.ndarray, ...]
    raw_common_mask: np.ndarray
    common_mask: np.ndarray
    prepared_signals: tuple[np.ndarray, ...]
    prepared_label: np.ndarray
    factor_correlation: np.ndarray
    predictive: np.ndarray
    system_inverse: np.ndarray
    score: PoolScore

    @property
    def valid_days(self) -> int:
        return int(np.isfinite(np.asarray(self.score.daily_ic, dtype=float)).sum())

    @property
    def valid_observations(self) -> int:
        return int(self.common_mask.sum())


@dataclass(frozen=True)
class _PreparedNeutralizationSupport:
    common_mask: np.ndarray
    day_solves: tuple[PreparedRiskSolve | None, ...]
    residual_label: np.ndarray
    label_diagnostics: tuple[dict[str, object], ...]


class RewardObjective(ABC):
    """Train-only objective with reusable, exact complete-case score state."""

    def __init__(
        self,
        label: np.ndarray,
        mask: np.ndarray,
        exposures: np.ndarray | None = None,
        ridge: float = 1e-3,
        min_pool_valid_day_rate: float = 0.80,
        min_pool_observation_rate: float = 0.80,
        min_pool_valid_days: int = 252,
    ):
        self.label = np.asarray(label, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.exposures = None if exposures is None else np.asarray(exposures, dtype=float)
        self.ridge = float(ridge)
        self.min_pool_valid_day_rate = float(min_pool_valid_day_rate)
        self.min_pool_observation_rate = float(min_pool_observation_rate)
        self.min_pool_valid_days = int(min_pool_valid_days)
        self.last_neutralization_diagnostics: list[dict[str, object]] = []
        self.prepare_calls = 0
        self.parallel_workers = 1
        self._neutralization_lock = threading.RLock()
        self._neutralization_supports: OrderedDict[
            bytes, _PreparedNeutralizationSupport
        ] = OrderedDict()
        self._prepared_pool_states: OrderedDict[
            tuple[int, ...], PreparedPoolState
        ] = OrderedDict()
        if self.label.shape != self.mask.shape:
            raise ValueError("label and mask shapes differ")
        if self.exposures is not None and self.exposures.shape[:2] != self.label.shape:
            raise ValueError("exposure panel shape mismatch")
        if not 0 < self.min_pool_valid_day_rate <= 1:
            raise ValueError("min_pool_valid_day_rate must be in (0, 1]")
        if not 0 < self.min_pool_observation_rate <= 1:
            raise ValueError("min_pool_observation_rate must be in (0, 1]")
        if self.min_pool_valid_days < 1:
            raise ValueError("min_pool_valid_days must be positive")

    def _base_support(self) -> np.ndarray:
        support = self.mask & np.isfinite(self.label)
        if self.exposures is not None:
            support &= np.isfinite(self.exposures).all(axis=2)
        return support

    def _independent_label(self, support: np.ndarray) -> np.ndarray:
        if self.exposures is None:
            return self.label
        return self._prepared_neutralization_support(support).residual_label

    def _independent_signal(self, signal: np.ndarray, support: np.ndarray) -> np.ndarray:
        return self._independent_signals([signal], support)[0]

    def _independent_signals(
        self,
        signals: list[np.ndarray],
        support: np.ndarray,
    ) -> list[np.ndarray]:
        """Transform factors independently while sharing identical risk solves."""
        arrays = [np.asarray(signal, dtype=float) for signal in signals]
        for values in arrays:
            if values.shape != self.label.shape:
                raise ValueError("signal panel shape mismatch")
        standardized = [
            FactorCalculator(self.label, support & np.isfinite(values)).standardize(values)
            for values in arrays
        ]
        supports = [support & np.isfinite(values) for values in standardized]
        if self.exposures is not None:
            groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
            for index, signal_support in enumerate(supports):
                key = np.packbits(signal_support, axis=None).tobytes()
                group = groups.get(key)
                if group is None:
                    groups[key] = (signal_support, [index])
                elif np.array_equal(group[0], signal_support):
                    group[1].append(index)
                else:  # pragma: no cover - exact fixed-shape packbits key
                    raise RuntimeError("packed independent-factor support collision")
            residuals: list[np.ndarray | None] = [None] * len(standardized)
            for signal_support, indices in groups.values():
                transformed, _ = self._neutralize_prestandardized(
                    [standardized[index] for index in indices], signal_support
                )
                for index, residual in zip(indices, transformed, strict=True):
                    residuals[index] = residual
            standardized = [np.asarray(value) for value in residuals]
            supports = [
                signal_support & np.isfinite(values)
                for signal_support, values in zip(supports, standardized, strict=True)
            ]
            standardized = [
                FactorCalculator(self.label, signal_support).standardize(values)
                for values, signal_support in zip(standardized, supports, strict=True)
            ]
        for values, signal_support in zip(standardized, supports, strict=True):
            values[~signal_support | ~np.isfinite(values)] = np.nan
        return standardized

    def support_diagnostics(self, state: PreparedPoolState) -> dict[str, float | int | bool]:
        base = state.raw_common_mask
        base_days = int((base.sum(axis=1) >= 3).sum())
        valid_days = state.valid_days
        base_observations = int(base.sum())
        valid_observations = state.valid_observations
        required_days = min(
            base_days,
            max(self.min_pool_valid_days, int(np.ceil(self.min_pool_valid_day_rate * base_days))),
        )
        day_rate = float(valid_days / max(1, base_days))
        observation_rate = float(valid_observations / max(1, base_observations))
        valid = bool(
            valid_days >= required_days
            and observation_rate >= self.min_pool_observation_rate
        )
        return {
            "valid": valid,
            "valid_days": valid_days,
            "required_valid_days": required_days,
            "valid_observations": valid_observations,
            "base_observations": base_observations,
            "valid_day_rate": day_rate,
            "observation_rate": observation_rate,
        }

    def required_valid_days(self) -> int:
        base = self._base_support()
        base_days = int((base.sum(axis=1) >= 3).sum())
        return min(
            base_days,
            max(self.min_pool_valid_days, int(np.ceil(self.min_pool_valid_day_rate * base_days))),
        )

    def support_is_valid(self, state: PreparedPoolState) -> bool:
        return bool(self.support_diagnostics(state)["valid"])

    def _complete_case_mask(self, signals: list[np.ndarray], label: np.ndarray) -> np.ndarray:
        common = self.mask & np.isfinite(label)
        for signal in signals:
            values = np.asarray(signal, dtype=float)
            if values.shape != self.label.shape:
                raise ValueError("signal panel shape mismatch")
            common &= np.isfinite(values)
        return common

    def _neutralized_inputs(
        self, signals: list[np.ndarray], common: np.ndarray | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        if common is None:
            common = self._complete_case_mask(signals, self.label)
        common = np.asarray(common, dtype=bool) & np.isfinite(self.exposures).all(axis=2)
        calculator = FactorCalculator(self.label, common)
        standardized = self._standardize_many(calculator, signals)
        for signal in standardized:
            common &= np.isfinite(signal)
        return self._neutralize_prestandardized(standardized, common)

    def _neutralize_prestandardized(
        self,
        standardized: list[np.ndarray],
        common: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Project one or many standardized signals with one cached support."""
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        support = self._prepared_neutralization_support(common)
        residual_signals = [np.full_like(self.label, np.nan, dtype=float) for _ in standardized]
        neutralizer = RiskNeutralizer()
        diagnostics: list[dict[str, object]] = []
        for day in range(self.label.shape[0]):
            day_mask = common[day]
            prepared = support.day_solves[day]
            label_diagnostics = dict(support.label_diagnostics[day])
            if prepared is None:
                diagnostics.append(label_diagnostics)
                continue
            matrix = np.column_stack([signal[day] for signal in standardized])
            residual, _ = neutralizer.residualize_matrix_prepared(
                day,
                matrix,
                self.exposures[day],
                day_mask,
                prepared,
                compute_diagnostics=False,
            )
            for index in range(len(standardized)):
                residual_signals[index][day] = residual[:, index]
            diagnostics.append({
                **label_diagnostics,
                "status": "ok",
                "joint_columns": len(standardized) + 1,
            })
        with self._neutralization_lock:
            self.last_neutralization_diagnostics = diagnostics
        return residual_signals, support.residual_label

    def _prepared_neutralization_support(
        self, common: np.ndarray
    ) -> _PreparedNeutralizationSupport:
        """Cache exposure QR and label residuals for an exact support mask."""
        if self.exposures is None:
            raise ValueError("risk-neutral objective requires exposures")
        packed = np.packbits(np.asarray(common, dtype=bool), axis=None)
        key = hashlib.blake2b(packed, digest_size=16).digest()
        with self._neutralization_lock:
            cached = self._neutralization_supports.get(key)
            if cached is not None and np.array_equal(cached.common_mask, common):
                self._neutralization_supports.move_to_end(key)
                return cached

            neutralizer = RiskNeutralizer()
            solves: list[PreparedRiskSolve | None] = []
            residual_label = np.full_like(self.label, np.nan, dtype=float)
            diagnostics: list[dict[str, object]] = []
            for day in range(self.label.shape[0]):
                day_mask = np.asarray(common[day], dtype=bool)
                if day_mask.sum() <= self.exposures.shape[2]:
                    solves.append(None)
                    diagnostics.append({
                        "date": str(day),
                        "status": "insufficient_observations",
                        "n_observations": int(day_mask.sum()),
                        "n_columns": int(self.exposures.shape[2]),
                    })
                    continue
                prepared = neutralizer.prepare(self.exposures[day], day_mask)
                residual, day_diagnostics = neutralizer.residualize_matrix_prepared(
                    day,
                    self.label[day],
                    self.exposures[day],
                    day_mask,
                    prepared,
                )
                solves.append(prepared)
                residual_label[day] = residual
                diagnostics.append({**day_diagnostics[-1], "status": "ok"})
            state = _PreparedNeutralizationSupport(
                np.asarray(common, dtype=bool).copy(),
                tuple(solves),
                residual_label,
                tuple(diagnostics),
            )
            self._neutralization_supports[key] = state
            self._neutralization_supports.move_to_end(key)
            while len(self._neutralization_supports) > 2:
                self._neutralization_supports.popitem(last=False)
            return state

    def _objective_inputs(
        self, signals: list[np.ndarray], raw_common: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        return signals, self.label

    def _standardize_many(
        self,
        calculator: FactorCalculator,
        signals: list[np.ndarray] | tuple[np.ndarray, ...],
    ) -> list[np.ndarray]:
        """Parallelize independent factor transforms, not full panel jobs."""
        values = list(signals)
        workers = min(max(1, int(self.parallel_workers)), len(values) or 1)
        if workers == 1:
            return [calculator.standardize(signal) for signal in values]
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="factor-standardize",
        ) as executor:
            return list(executor.map(calculator.standardize, values))

    def _score_from_daily(self, daily: np.ndarray, weights: np.ndarray) -> PoolScore:
        mean = float(np.nanmean(daily)) if np.isfinite(daily).any() else float("-inf")
        return PoolScore(mean, mean, tuple(map(float, daily)), tuple(map(float, weights)))

    def _build_state(
        self,
        raw_signals: list[np.ndarray],
        raw_common: np.ndarray,
        objective_signals: list[np.ndarray],
        objective_label: np.ndarray,
    ) -> PreparedPoolState:
        common = self._complete_case_mask(objective_signals, objective_label)
        calculator = FactorCalculator(objective_label, common)
        prepared = self._standardize_many(calculator, objective_signals)
        initial_common = common.copy()
        for signal in prepared:
            common &= np.isfinite(signal)
        if not np.array_equal(common, initial_common):
            calculator = FactorCalculator(objective_label, common)
            prepared = self._standardize_many(calculator, objective_signals)
        correlation, predictive = _factor_moments_batched(
            prepared, objective_label, common
        )
        return self._state_from_moments(
            raw_signals, raw_common, common, prepared, objective_label, correlation, predictive
        )

    def _build_independent_state(
        self,
        raw_signals: list[np.ndarray],
        reference_mask: np.ndarray,
        prepared_signals: list[np.ndarray] | None = None,
    ) -> PreparedPoolState:
        prepared_label = self._independent_label(reference_mask)
        prepared = prepared_signals or self._independent_signals(raw_signals, reference_mask)
        correlation, predictive = _factor_moments_batched(
            prepared, prepared_label, reference_mask
        )
        return self._state_from_moments(
            raw_signals,
            reference_mask,
            reference_mask,
            prepared,
            prepared_label,
            correlation,
            predictive,
        )

    def prepare_pool(self, signals: list[np.ndarray]) -> PreparedPoolState:
        """Prepare a pool with independent per-factor availability."""
        self.prepare_calls += 1
        raw = [np.asarray(signal, dtype=float) for signal in signals]
        raw_common = self._base_support()
        if not raw:
            score = PoolScore(0.0, 0.0, tuple(), tuple(), 0.0)
            empty = np.empty((0, 0), dtype=float)
            return PreparedPoolState(tuple(), raw_common, raw_common, tuple(), self.label, empty, np.empty(0), empty, score)
        return self._build_independent_state(raw, raw_common)

    def prepare_pool_cached(self, signals: list[np.ndarray]) -> PreparedPoolState:
        """Reuse formal recheck states when one becomes the next frozen pool."""
        key = tuple(id(signal) for signal in signals)
        with self._neutralization_lock:
            cached = self._prepared_pool_states.get(key)
            if cached is not None:
                self._prepared_pool_states.move_to_end(key)
                return cached
        state = self.prepare_pool(signals)
        with self._neutralization_lock:
            self._prepared_pool_states[key] = state
            self._prepared_pool_states.move_to_end(key)
            # One frozen baseline plus at most three formal admission rechecks.
            while len(self._prepared_pool_states) > 4:
                self._prepared_pool_states.popitem(last=False)
        return state

    def cache_prepared_pool(self, state: PreparedPoolState) -> None:
        """Register a formally rechecked state as a possible next baseline."""
        key = tuple(id(signal) for signal in state.raw_signals)
        with self._neutralization_lock:
            self._prepared_pool_states[key] = state
            self._prepared_pool_states.move_to_end(key)
            while len(self._prepared_pool_states) > 4:
                self._prepared_pool_states.popitem(last=False)

    def prepare_add(self, base: PreparedPoolState, candidate: np.ndarray) -> PreparedPoolState:
        """Append on the frozen base-pool support for an exact add-only delta."""
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != self.label.shape:
            raise ValueError("signal panel shape mismatch")
        if not base.raw_signals:
            return self.prepare_pool([candidate])
        prepared_candidate = self._independent_signal(candidate, base.raw_common_mask)
        return self._prepare_add_from_prepared_candidate(base, candidate, prepared_candidate)

    def prepare_add_many(
        self,
        base: PreparedPoolState,
        candidates: list[np.ndarray],
    ) -> list[PreparedPoolState]:
        """Batch same-support risk projection across a frozen candidate group."""
        arrays = [np.asarray(candidate, dtype=float) for candidate in candidates]
        if not arrays:
            return []
        if not base.raw_signals:
            return [self.prepare_add(base, candidate) for candidate in arrays]
        for candidate in arrays:
            if candidate.shape != self.label.shape:
                raise ValueError("signal panel shape mismatch")

        prepared = self._independent_signals(arrays, base.raw_common_mask)
        return self._prepare_add_many_from_prepared_candidates(base, arrays, prepared)

    def _prepare_add_many_from_prepared_candidates(
        self,
        base: PreparedPoolState,
        candidates: list[np.ndarray],
        prepared_candidates: list[np.ndarray],
    ) -> list[PreparedPoolState]:
        """Build all add states while loading each frozen base signal once."""
        if len(candidates) != len(prepared_candidates):
            raise ValueError("candidate and prepared-candidate counts differ")
        if not candidates:
            return []
        base_count = len(base.prepared_signals)
        candidate_count = len(candidates)
        base_matrix = np.stack(base.prepared_signals, axis=-1)
        candidate_matrix = np.stack(prepared_candidates, axis=-1)
        label_matrix = np.asarray(base.prepared_label)[..., None]

        predictive_daily = _daily_corr_columns(
            candidate_matrix, label_matrix, base.common_mask
        )[:, :, 0]
        with np.errstate(all="ignore"):
            candidate_predictive = np.nanmean(predictive_daily, axis=0)
        candidate_predictive = np.where(
            np.isfinite(candidate_predictive), candidate_predictive, 0.0
        )
        pair_daily = _daily_corr_columns(
            base_matrix, candidate_matrix, base.common_mask
        )
        with np.errstate(all="ignore"):
            pair_means = np.nanmean(pair_daily, axis=0)
        pair_means = np.where(np.isfinite(pair_means), pair_means, 0.0)

        correlations: list[np.ndarray] = []
        inverses: list[np.ndarray] = []
        weights = np.empty((candidate_count, base_count + 1), dtype=float)
        predictives: list[np.ndarray] = []
        for candidate_index in range(candidate_count):
            correlation = np.eye(base_count + 1, dtype=float)
            correlation[:-1, :-1] = base.factor_correlation
            correlation[:-1, -1] = pair_means[:, candidate_index]
            correlation[-1, :-1] = pair_means[:, candidate_index]
            predictive = np.empty(base_count + 1, dtype=float)
            predictive[:-1] = base.predictive
            predictive[-1] = candidate_predictive[candidate_index]
            system = correlation + self.ridge * np.eye(base_count + 1)
            try:
                inverse = np.linalg.inv(system)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(system)
            correlations.append(correlation)
            predictives.append(predictive)
            inverses.append(inverse)
            weights[candidate_index] = inverse @ predictive

        base_finite = np.isfinite(base_matrix)
        candidate_finite = np.isfinite(candidate_matrix)
        combined = np.einsum(
            "daf,cf->dac",
            np.where(base_finite, base_matrix, 0.0),
            weights[:, :base_count],
            optimize=True,
        )
        combined += np.where(candidate_finite, candidate_matrix, 0.0) * weights[:, base_count][None, None, :]
        active_weight = np.einsum(
            "daf,cf->dac", base_finite.astype(float), np.abs(weights[:, :base_count]), optimize=True
        )
        active_weight += candidate_finite * np.abs(weights[:, base_count])[None, None, :]
        total_weight = np.abs(weights).sum(axis=1)
        available = active_weight > 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            combined *= total_weight[None, None, :] / active_weight
        available &= base.common_mask[..., None]
        combined[~available] = np.nan
        combined_daily = _daily_corr_columns(
            combined, label_matrix, base.common_mask
        )[:, :, 0]

        states = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_weights = weights[candidate_index]
            score = self._score_from_daily(
                combined_daily[:, candidate_index], candidate_weights
            )
            states.append(PreparedPoolState(
                tuple(list(base.raw_signals) + [candidate]),
                base.raw_common_mask,
                available[..., candidate_index],
                tuple(list(base.prepared_signals) + [prepared_candidates[candidate_index]]),
                base.prepared_label,
                correlations[candidate_index],
                predictives[candidate_index],
                inverses[candidate_index],
                score,
            ))
        return states

    def _prepare_add_from_prepared_candidate(
        self,
        base: PreparedPoolState,
        candidate: np.ndarray,
        prepared_candidate: np.ndarray,
    ) -> PreparedPoolState:
        prepared = list(base.prepared_signals) + [prepared_candidate]
        count = len(prepared)
        correlation = np.eye(count, dtype=float)
        correlation[:-1, :-1] = base.factor_correlation
        predictive = np.empty(count, dtype=float)
        predictive[:-1] = base.predictive
        candidate_daily = daily_corr(
            prepared_candidate, base.prepared_label, base.common_mask
        )
        predictive[-1] = (
            float(np.nanmean(candidate_daily))
            if np.isfinite(candidate_daily).any()
            else 0.0
        )
        for index, existing in enumerate(base.prepared_signals):
            values = daily_corr(existing, prepared_candidate, base.common_mask)
            value = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
            correlation[index, -1] = correlation[-1, index] = value
        return self._state_from_moments(
            list(base.raw_signals) + [candidate],
            base.common_mask,
            base.common_mask,
            prepared,
            base.prepared_label,
            correlation,
            predictive,
        )

    def _state_from_moments(
        self,
        raw_signals: list[np.ndarray],
        raw_common: np.ndarray,
        common: np.ndarray,
        prepared: list[np.ndarray],
        label: np.ndarray,
        correlation: np.ndarray,
        predictive: np.ndarray,
    ) -> PreparedPoolState:
        system = correlation + self.ridge * np.eye(len(prepared))
        try:
            inverse = np.linalg.inv(system)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(system)
        weights = inverse @ predictive
        combined, available = combine_available_signals(prepared, weights)
        result_mask = common & available & np.isfinite(label)
        combined[~result_mask] = np.nan
        daily = daily_corr(combined, label, result_mask)
        return PreparedPoolState(
            tuple(raw_signals), raw_common, result_mask, tuple(prepared), label,
            correlation, predictive, inverse, self._score_from_daily(daily, weights),
        )

    def prepare_subset(
        self,
        state: PreparedPoolState,
        indices: list[int] | tuple[int, ...],
        *,
        natural_support: bool = False,
    ) -> PreparedPoolState:
        if not indices:
            empty = np.empty((0, 0), dtype=float)
            score = PoolScore(0.0, 0.0, tuple(), tuple(), 0.0)
            return PreparedPoolState(tuple(), state.raw_common_mask, state.common_mask, tuple(), state.prepared_label, empty, np.empty(0), empty, score)
        selected = np.asarray(indices, dtype=int)
        correlation = state.factor_correlation[np.ix_(selected, selected)]
        predictive = state.predictive[selected]
        prepared = [state.prepared_signals[index] for index in selected]
        return self._state_from_moments(
            [state.raw_signals[index] for index in selected],
            state.raw_common_mask,
            state.raw_common_mask if natural_support else state.common_mask,
            prepared,
            state.prepared_label,
            correlation,
            predictive,
        )

    def score_subset(self, state: PreparedPoolState, indices: list[int] | tuple[int, ...]) -> PoolScore:
        return self.prepare_subset(state, indices).score

    def score_subsets(
        self,
        state: PreparedPoolState,
        subsets: list[list[int]] | tuple[tuple[int, ...], ...],
    ) -> list[PoolScore]:
        """Score several fixed-support subsets in one panel pass.

        All alternatives share the prepared signals, label and support.  The
        old path rebuilt and stacked the complete panel independently for each
        of the three saliency deletions.
        """
        if not subsets:
            return []
        weights: list[np.ndarray] = []
        normalized: list[np.ndarray] = []
        for indices in subsets:
            selected = np.asarray(indices, dtype=int)
            if not len(selected):
                normalized.append(selected)
                weights.append(np.empty(0, dtype=float))
                continue
            correlation = state.factor_correlation[np.ix_(selected, selected)]
            predictive = state.predictive[selected]
            system = correlation + self.ridge * np.eye(len(selected))
            try:
                subset_weights = np.linalg.inv(system) @ predictive
            except np.linalg.LinAlgError:
                subset_weights = np.linalg.pinv(system) @ predictive
            normalized.append(selected)
            weights.append(subset_weights)
        factor_count = len(state.prepared_signals)
        alternative_count = len(normalized)
        full_weights = np.zeros((alternative_count, factor_count), dtype=float)
        for alternative, (selected, subset_weights) in enumerate(
            zip(normalized, weights, strict=True)
        ):
            full_weights[alternative, selected] = subset_weights
        signal_matrix = np.stack(state.prepared_signals, axis=-1)
        finite = np.isfinite(signal_matrix)
        combined = np.einsum(
            "daf,kf->dak",
            np.where(finite, signal_matrix, 0.0),
            full_weights,
            optimize=True,
        )
        active_weight = np.einsum(
            "daf,kf->dak",
            finite.astype(float),
            np.abs(full_weights),
            optimize=True,
        )
        total_weight = np.abs(full_weights).sum(axis=1)
        available = active_weight > 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            combined *= total_weight[None, None, :] / active_weight
        available &= state.common_mask[..., None]
        combined[~available] = np.nan
        daily = _daily_corr_columns(
            combined,
            np.asarray(state.prepared_label)[..., None],
            state.common_mask,
        )[:, :, 0]
        return [
            self._score_from_daily(daily[:, alternative], subset_weights)
            for alternative, subset_weights in enumerate(weights)
        ]

    def compare_prepared_pools(
        self,
        old_state: PreparedPoolState,
        new_state: PreparedPoolState,
    ) -> tuple[PoolScore, PoolScore, PreparedPoolState]:
        """Reuse prepared transforms unless the actual support changed."""
        if np.array_equal(old_state.common_mask, new_state.common_mask):
            return old_state.score, new_state.score, new_state
        return self.compare_pools(
            list(old_state.raw_signals), list(new_state.raw_signals)
        )

    def compare_pools(
        self,
        old_signals: list[np.ndarray],
        new_signals: list[np.ndarray],
    ) -> tuple[PoolScore, PoolScore, PreparedPoolState]:
        """Compare a transition on one exact support and return natural new state."""
        old_natural = self.prepare_pool_cached(old_signals)
        new_natural = self.prepare_pool_cached(new_signals)
        shared = old_natural.common_mask & new_natural.common_mask
        old_shared = self._build_independent_state(old_signals, shared)
        new_shared = self._build_independent_state(new_signals, shared)
        return old_shared.score, new_shared.score, new_natural

    def _daily_ic(self, signals: list[np.ndarray], label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if label is self.label:
            score = self.prepare_pool(signals).score
            return np.asarray(score.daily_ic), np.asarray(score.weights)
        common = self._complete_case_mask(signals, label)
        state = self._build_state(signals, common, signals, label)
        return np.asarray(state.score.daily_ic), np.asarray(state.score.weights)

    def score_pool(self, signals: list[np.ndarray]) -> PoolScore:
        return self.prepare_pool(signals).score

    @abstractmethod
    def objective_name(self) -> str: ...
