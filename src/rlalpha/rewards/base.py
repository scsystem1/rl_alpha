from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import threading

import numpy as np
from ..factors.calculator import FactorCalculator, daily_corr
from ..factors.moments import fixed_universe_moments, solve_psd_ridge
from ..factors.records import PoolScore
from ..factors.transform import combine_fixed_signals, prepare_fixed_universe_inputs
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


def _fixed_universe_daily_corr(
    signal: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Daily Pearson IC where an all-zero opinion has exactly zero IC."""

    result = daily_corr(signal, label, mask)
    for day in np.flatnonzero(~np.isfinite(result)):
        common = mask[day] & np.isfinite(signal[day]) & np.isfinite(label[day])
        if common.sum() < 3:
            continue
        if np.var(label[day, common]) > 1e-24 and np.var(signal[day, common]) <= 1e-24:
            result[day] = 0.0
    return result


def _factor_moments_batched(
    signals: list[np.ndarray],
    label: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Day-equal PSD Gram and factor/label cross moments on one support."""
    moments = fixed_universe_moments(signals, label, mask)
    return moments.gram, moments.predictive


@dataclass(frozen=True)
class PreparedPoolState:
    """Reusable sufficient state for scoring one fixed-universe pool."""

    raw_signals: tuple[np.ndarray, ...]
    raw_common_mask: np.ndarray
    common_mask: np.ndarray
    prepared_signals: tuple[np.ndarray, ...]
    prepared_label: np.ndarray
    factor_gram: np.ndarray
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
    """Train-only objective with reusable, exact fixed-universe score state."""

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
        support = self.mask.copy()
        if self.exposures is not None:
            support &= np.isfinite(self.exposures).all(axis=2)
        return support

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
        standardized = [FactorCalculator(
            np.zeros(self.label.shape, dtype=float), support & np.isfinite(values)
        ).standardize(values) for values in arrays]
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
            neutralizer = RiskNeutralizer()
            for signal_support, indices in groups.values():
                transformed = [np.full_like(self.label, np.nan, dtype=float) for _ in indices]
                for day in range(self.label.shape[0]):
                    day_mask = signal_support[day]
                    if day_mask.sum() <= self.exposures.shape[2]:
                        continue
                    matrix = np.column_stack([standardized[index][day] for index in indices])
                    residual, _ = neutralizer.residualize_matrix(
                        day, matrix, self.exposures[day], day_mask
                    )
                    for column in range(len(indices)):
                        transformed[column][day] = residual[:, column]
                for index, residual in zip(indices, transformed, strict=True):
                    residuals[index] = residual
            standardized = [np.asarray(value) for value in residuals]
            supports = [
                signal_support & np.isfinite(values)
                for signal_support, values in zip(supports, standardized, strict=True)
            ]
            standardized = [
                FactorCalculator(self.label, signal_support).zscore(values)
                for values, signal_support in zip(standardized, supports, strict=True)
            ]
        for values, signal_support in zip(standardized, supports, strict=True):
            values[~signal_support | ~np.isfinite(values)] = np.nan
        return standardized

    def support_diagnostics(self, state: PreparedPoolState) -> dict[str, float | int | bool]:
        base = state.raw_common_mask & np.isfinite(self.label)
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
        base = self._base_support() & np.isfinite(self.label)
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
        gram, predictive = _factor_moments_batched(
            prepared, objective_label, common
        )
        return self._state_from_moments(
            raw_signals, raw_common, common, prepared, objective_label, gram, predictive
        )

    def _build_independent_state(
        self,
        raw_signals: list[np.ndarray],
        reference_mask: np.ndarray,
        prepared_signals: list[np.ndarray] | None = None,
    ) -> PreparedPoolState:
        deployment = prepared_signals or self._independent_signals(raw_signals, reference_mask)
        prepared, prepared_label, metric_mask, _ = prepare_fixed_universe_inputs(
            deployment,
            self.label,
            reference_mask,
            self.exposures,
            neutralize=self.exposures is not None,
        )
        gram, predictive = _factor_moments_batched(
            list(prepared), prepared_label, metric_mask
        )
        return self._state_from_moments(
            raw_signals,
            reference_mask,
            metric_mask,
            list(prepared),
            prepared_label,
            gram,
            predictive,
        )

    def prepare_pool(self, signals: list[np.ndarray]) -> PreparedPoolState:
        """Prepare a pool with independent per-factor availability."""
        self.prepare_calls += 1
        raw = [np.asarray(signal, dtype=float) for signal in signals]
        raw_common = self._base_support()
        if not raw:
            _, prepared_label, metric_mask, _ = prepare_fixed_universe_inputs(
                tuple(), self.label, raw_common, self.exposures,
                neutralize=self.exposures is not None,
            )
            score = PoolScore(0.0, 0.0, tuple(), tuple(), 0.0)
            empty = np.empty((0, 0), dtype=float)
            return PreparedPoolState(
                tuple(), raw_common, metric_mask, tuple(), prepared_label,
                empty, np.empty(0), empty, score,
            )
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
        objective_candidates, prepared_label, metric_mask, _ = prepare_fixed_universe_inputs(
            prepared_candidates,
            self.label,
            base.raw_common_mask,
            self.exposures,
            neutralize=self.exposures is not None,
        )
        if not np.array_equal(metric_mask, base.common_mask):
            raise RuntimeError("fixed metric universe changed while adding candidates")
        if not np.allclose(prepared_label, base.prepared_label, equal_nan=True):
            raise RuntimeError("fixed metric label changed while adding candidates")

        all_prepared = list(base.prepared_signals) + list(objective_candidates)
        all_moments = fixed_universe_moments(
            all_prepared, base.prepared_label, base.common_mask
        )
        base_count = len(base.prepared_signals)
        states: list[PreparedPoolState] = []
        for candidate_index, candidate in enumerate(candidates):
            selected = np.asarray(list(range(base_count)) + [base_count + candidate_index])
            gram = all_moments.gram[np.ix_(selected, selected)]
            predictive = all_moments.predictive[selected]
            prepared = list(base.prepared_signals) + [objective_candidates[candidate_index]]
            states.append(self._state_from_moments(
                list(base.raw_signals) + [candidate],
                base.raw_common_mask,
                base.common_mask,
                prepared,
                base.prepared_label,
                gram,
                predictive,
            ))
        return states

    def _prepare_add_from_prepared_candidate(
        self,
        base: PreparedPoolState,
        candidate: np.ndarray,
        prepared_candidate: np.ndarray,
    ) -> PreparedPoolState:
        return self._prepare_add_many_from_prepared_candidates(
            base, [candidate], [prepared_candidate]
        )[0]

    def _state_from_moments(
        self,
        raw_signals: list[np.ndarray],
        raw_common: np.ndarray,
        common: np.ndarray,
        prepared: list[np.ndarray],
        label: np.ndarray,
        gram: np.ndarray,
        predictive: np.ndarray,
    ) -> PreparedPoolState:
        weights, inverse = solve_psd_ridge(gram, predictive, self.ridge)
        combined, _ = combine_fixed_signals(prepared, weights)
        result_mask = common & np.isfinite(label)
        combined[~result_mask] = np.nan
        daily = _fixed_universe_daily_corr(combined, label, result_mask)
        return PreparedPoolState(
            tuple(raw_signals), raw_common, result_mask, tuple(prepared), label,
            gram, predictive, inverse, self._score_from_daily(daily, weights),
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
        gram = state.factor_gram[np.ix_(selected, selected)]
        predictive = state.predictive[selected]
        prepared = [state.prepared_signals[index] for index in selected]
        return self._state_from_moments(
            [state.raw_signals[index] for index in selected],
            state.raw_common_mask,
            state.common_mask,
            prepared,
            state.prepared_label,
            gram,
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
            gram = state.factor_gram[np.ix_(selected, selected)]
            predictive = state.predictive[selected]
            subset_weights, _ = solve_psd_ridge(gram, predictive, self.ridge)
            normalized.append(selected)
            weights.append(subset_weights)
        factor_count = len(state.prepared_signals)
        alternative_count = len(normalized)
        full_weights = np.zeros((alternative_count, factor_count), dtype=float)
        for alternative, (selected, subset_weights) in enumerate(
            zip(normalized, weights, strict=True)
        ):
            full_weights[alternative, selected] = subset_weights
        combined, _ = combine_fixed_signals(state.prepared_signals, full_weights)
        combined = np.where(state.common_mask[..., None], combined, np.nan)
        daily = _daily_corr_columns(
            combined,
            np.asarray(state.prepared_label)[..., None],
            state.common_mask,
        )[:, :, 0]
        for alternative in range(alternative_count):
            fixed = _fixed_universe_daily_corr(
                combined[:, :, alternative], state.prepared_label, state.common_mask
            )
            daily[:, alternative] = fixed
        return [
            self._score_from_daily(daily[:, alternative], subset_weights)
            for alternative, subset_weights in enumerate(weights)
        ]

    def compare_prepared_pools(
        self,
        old_state: PreparedPoolState,
        new_state: PreparedPoolState,
    ) -> tuple[PoolScore, PoolScore, PreparedPoolState]:
        """Fixed metric support makes natural and shared comparisons identical."""
        if not np.array_equal(old_state.common_mask, new_state.common_mask):
            raise RuntimeError("fixed metric universe changed across prepared pools")
        return old_state.score, new_state.score, new_state

    def compare_pools(
        self,
        old_signals: list[np.ndarray],
        new_signals: list[np.ndarray],
    ) -> tuple[PoolScore, PoolScore, PreparedPoolState]:
        """Compare a transition on the factor-independent metric universe."""
        old_natural = self.prepare_pool_cached(old_signals)
        new_natural = self.prepare_pool_cached(new_signals)
        if not np.array_equal(old_natural.common_mask, new_natural.common_mask):
            raise RuntimeError("fixed metric universe changed across pools")
        return old_natural.score, new_natural.score, new_natural

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
