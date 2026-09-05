from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .records import CandidateScore, PoolEntry, PoolIncrement, PoolScore
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class Admission:
    admitted: bool
    candidate_hash: str | None
    replaced_hash: str | None
    delta: float
    pool_version: int
    reason: str = "ok"
    admission_event_id: str | None = None
    pre_pool_snapshot_hash: str | None = None
    post_pool_snapshot_hash: str | None = None


@dataclass(frozen=True)
class _CandidatePlan:
    candidate: PoolEntry
    score: CandidateScore
    delete_index: int | None
    add_state: object | None = None


class PoolManager:
    """Fixed-capacity pool with add-only rewards and bounded pruning work."""

    def __init__(
        self,
        objective: object,
        capacity: int = 20,
        min_delta: float = 1e-5,
        replacement_top_k: int = 3,
        admission_recheck_top_k: int = 3,
    ):
        self.objective = objective
        self.capacity = capacity
        self.min_delta = min_delta
        self.replacement_top_k = replacement_top_k
        self.admission_recheck_top_k = admission_recheck_top_k
        self.entries: list[PoolEntry] = []
        self.version = 0
        self.history: list[dict[str, object]] = []
        self._prepared_cache: object | None = None
        self._cache_signature: tuple[object, ...] | None = None

    @property
    def hashes(self) -> set[str]:
        return {entry.expr_hash for entry in self.entries}

    def _signature(self) -> tuple[object, ...]:
        return (
            self.version,
            tuple((entry.expr_hash, id(entry.signal)) for entry in self.entries),
        )

    def invalidate_cache(self) -> None:
        self._prepared_cache = None
        self._cache_signature = None

    def prepared_state(self) -> object | None:
        signature = self._signature()
        if self._cache_signature != signature:
            prepare = getattr(self.objective, "prepare_pool", None)
            prepare_cached = getattr(self.objective, "prepare_pool_cached", None)
            self._prepared_cache = (
                prepare_cached([entry.signal for entry in self.entries])
                if callable(prepare_cached)
                else prepare([entry.signal for entry in self.entries])
                if callable(prepare)
                else None
            )
            self._cache_signature = signature
        return self._prepared_cache

    def _score(self, entries: list[PoolEntry]) -> PoolScore:
        if entries is self.entries or (
            len(entries) == len(self.entries)
            and all(left is right for left, right in zip(entries, self.entries, strict=True))
        ):
            prepared = self.prepared_state()
            if prepared is not None:
                return prepared.score
        signals = [entry.signal for entry in entries]
        prepare_cached = getattr(self.objective, "prepare_pool_cached", None)
        if callable(prepare_cached):
            return prepare_cached(signals).score
        return self.objective.score_pool(signals)

    @property
    def score(self) -> PoolScore:
        return self._score(self.entries)

    @staticmethod
    def _finite_delta(value: float) -> bool:
        return bool(np.isfinite(value))

    def _invalid(
        self, candidate: PoolEntry, baseline: PoolScore, reason: str, delta: float = float("-inf")
    ) -> _CandidatePlan:
        shaped = -0.5 if reason == "exact_duplicate" else -1.0
        return _CandidatePlan(
            candidate,
            CandidateScore(
                candidate.expr_hash,
                baseline,
                delta,
                shaped,
                valid=False,
                reason=reason,
                delta_add=delta,
            ),
            None,
        )

    def _fixed_subset_score(self, state: object, indices: list[int]) -> PoolScore:
        return self.objective.score_subset(state, indices)

    def _saliency(self, state: object, score: PoolScore) -> np.ndarray:
        saliency = getattr(self.objective, "saliency", None)
        if callable(saliency):
            return saliency(state)
        weights = np.asarray(score.weights, dtype=float)
        inverse = np.asarray(getattr(state, "system_inverse", np.empty((0, 0))), dtype=float)
        if inverse.shape == (len(weights), len(weights)):
            diagonal = np.diag(inverse)
            with np.errstate(divide="ignore", invalid="ignore"):
                saliency = weights * weights / diagonal
            return np.where(np.isfinite(saliency) & (diagonal > 0), saliency, np.inf)
        return np.where(np.isfinite(weights), weights * weights, np.inf)

    def _increment(self, old: PoolScore, new: PoolScore) -> PoolIncrement:
        compare = getattr(self.objective, "compare_scores", None)
        if callable(compare):
            return compare(old, new)
        mean = float(new.mean_ic - old.mean_ic)
        reward = float(new.objective - old.objective)
        return PoolIncrement(mean, float("nan"), mean - reward, reward)

    def _provisional_candidate_plan(
        self,
        candidate: PoolEntry,
        baseline: PoolScore,
        base_state: object | None,
        existing_hashes: set[str],
        prepared_add_state: object | None = None,
    ) -> _CandidatePlan:
        """Score one candidate against an immutable frozen pool.

        This phase has no group-wide side effects, so candidates may execute
        concurrently.  The bounded natural-support admission rechecks remain
        below in ``score_candidates`` and are selected only after every
        provisional plan is available.
        """
        if candidate.expr_hash in existing_hashes:
            return self._invalid(candidate, baseline, "exact_duplicate", 0.0)
        if not np.isfinite(baseline.objective):
            return self._invalid(candidate, baseline, "non_finite_baseline")

        if base_state is not None:
            add_state = (
                prepared_add_state
                if prepared_add_state is not None
                else self.objective.prepare_add(base_state, candidate.signal)
            )
            add_score = add_state.score
            # On unchanged support the cached frozen baseline is already the
            # exact left side of the add-only delta.  Rebuilding its weighted
            # signal here used to scan the complete panel once per candidate.
            if np.array_equal(add_state.common_mask, base_state.common_mask):
                add_baseline = baseline
            else:
                add_baseline = self._fixed_subset_score(
                    add_state, list(range(len(self.entries)))
                )
        else:
            add_state = None
            add_score = self._score(self.entries + [candidate])
            add_baseline = baseline
        if not np.isfinite(add_score.objective):
            return self._invalid(candidate, add_score, "non_finite_objective")
        support_is_valid = getattr(self.objective, "support_is_valid", None)
        support_diagnostics = getattr(self.objective, "support_diagnostics", None)
        support = support_diagnostics(add_state) if add_state is not None and callable(support_diagnostics) else {}
        if add_state is not None and callable(support_is_valid) and not support_is_valid(add_state):
            invalid = self._invalid(candidate, add_score, "insufficient_pool_support")
            return _CandidatePlan(
                candidate,
                replace(
                    invalid.score,
                    reward_valid_days=int(support.get("valid_days", 0)),
                    reward_valid_observations=int(support.get("valid_observations", 0)),
                    reward_valid_day_rate=float(support.get("valid_day_rate", 0.0)),
                    reward_observation_rate=float(support.get("observation_rate", 0.0)),
                ),
                None,
            )
        add_increment = self._increment(add_baseline, add_score)
        delta_add = add_increment.reward
        if not np.isfinite(delta_add):
            return self._invalid(candidate, add_score, "non_finite_delta")
        # Valid rewards are shaped only after every frozen-pool candidate has
        # an add-only delta, so one robust scale is shared by the whole group.
        shaped = 0.0

        if len(self.entries) < self.capacity:
            # Add reward and admission use the same frozen base support.
            post_delta = delta_add
            return _CandidatePlan(
                candidate,
                CandidateScore(
                    candidate.expr_hash,
                    add_score,
                    delta_add,
                    shaped,
                    delta_add=delta_add,
                    post_prune_delta=post_delta,
                    formally_rechecked=True,
                    reward_valid_days=int(support.get("valid_days", 0)),
                    reward_valid_observations=int(support.get("valid_observations", 0)),
                    reward_valid_day_rate=float(support.get("valid_day_rate", 0.0)),
                    reward_observation_rate=float(support.get("observation_rate", 0.0)),
                    add_increment=add_increment,
                    post_prune_increment=add_increment,
                ),
                None,
            )

        saliency = self._saliency(add_state, add_score)
        count = len(self.entries) + 1
        if len(saliency) != count:
            saliency = np.ones(count, dtype=float)
        delete_indices = np.argsort(saliency, kind="stable")[
            : min(self.replacement_top_k, count)
        ]
        keep_subsets = [
            [index for index in range(count) if index != int(delete_index)]
            for delete_index in delete_indices
        ]
        batch_score = getattr(self.objective, "score_subsets", None)
        batched = (
            batch_score(add_state, keep_subsets)
            if add_state is not None and callable(batch_score)
            else None
        )
        fixed_alternatives: list[tuple[PoolScore, int]] = []
        for alternative, delete_index in enumerate(delete_indices):
            keep = keep_subsets[alternative]
            if batched is not None:
                fixed_score = batched[alternative]
            elif add_state is not None:
                fixed_score = self._fixed_subset_score(add_state, keep)
            else:
                union = self.entries + [candidate]
                fixed_score = self.objective.score_pool(
                    [entry.signal for index, entry in enumerate(union) if index != int(delete_index)]
                )
            fixed_alternatives.append((fixed_score, int(delete_index)))
        finite = [
            item for item in fixed_alternatives if np.isfinite(item[0].objective)
        ]
        if not finite:
            return self._invalid(candidate, add_score, "non_finite_prune_objective")
        fixed_score, delete_index = max(finite, key=lambda item: self._increment(add_baseline, item[0]).reward)
        self_evicted = delete_index == len(self.entries)
        replaced_hash = None if self_evicted else self.entries[delete_index].expr_hash
        labels = [entry.expr_hash for entry in self.entries] + [candidate.expr_hash]
        eviction_candidates = tuple(labels[int(index)] for index in delete_indices)
        post_increment = self._increment(add_baseline, fixed_score)
        provisional_delta = post_increment.reward
        return _CandidatePlan(
            candidate,
            CandidateScore(
                candidate.expr_hash,
                fixed_score,
                delta_add,
                shaped,
                replaced_hash,
                True,
                "self_evicted" if self_evicted else "ok",
                delta_add,
                tuple(map(float, saliency)),
                eviction_candidates,
                provisional_delta,
                self_evicted,
                False,
                bool(delta_add > 0 and self_evicted),
                int(support.get("valid_days", 0)),
                int(support.get("valid_observations", 0)),
                float(support.get("valid_day_rate", 0.0)),
                float(support.get("observation_rate", 0.0)),
                add_increment=add_increment,
                post_prune_increment=post_increment,
            ),
            delete_index,
            add_state,
        )

    def score_candidates(
        self,
        candidates: Iterable[PoolEntry],
        *,
        max_workers: int = 1,
    ) -> list[CandidateScore]:
        """Compute add rewards, then bounded saliency admission plans."""
        candidates = list(candidates)
        baseline = self.score
        base_state = self.prepared_state()
        existing_hashes = self.hashes
        workers = min(max(1, int(max_workers)), len(candidates)) if candidates else 1
        prepare_many = getattr(self.objective, "prepare_add_many", None)
        prepared_add_states = (
            prepare_many(base_state, [candidate.signal for candidate in candidates])
            if base_state is not None and callable(prepare_many)
            else [None] * len(candidates)
        )
        if len(prepared_add_states) != len(candidates):
            raise RuntimeError("batched candidate preparation returned the wrong number of states")
        indexed = list(zip(candidates, prepared_add_states, strict=True))

        def score_one(item: tuple[PoolEntry, object | None]) -> _CandidatePlan:
            candidate, add_state = item
            return self._provisional_candidate_plan(
                candidate, baseline, base_state, existing_hashes, add_state
            )

        if workers == 1:
            plans = [score_one(item) for item in indexed]
        else:
            # executor.map preserves input order, which keeps lineage and all
            # stable tie-breaks identical to the serial implementation.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pool-candidate") as executor:
                plans = list(executor.map(score_one, indexed))

        if len(self.entries) >= self.capacity:
            eligible = [
                (index, plan) for index, plan in enumerate(plans)
                if plan.score.valid and self._finite_delta(plan.score.post_prune_delta)
            ]
            eligible.sort(key=lambda item: item[1].score.post_prune_delta, reverse=True)
            for index, plan in eligible[: self.admission_recheck_top_k]:
                if plan.score.self_evicted:
                    formal_score = baseline
                    formal_delta = 0.0
                    formal_valid = True
                    formal_increment = self._increment(baseline, baseline)
                else:
                    replacement = int(plan.delete_index)
                    formal_entries = (
                        self.entries[:replacement]
                        + [plan.candidate]
                        + self.entries[replacement + 1 :]
                    )
                    # Keep the same order as ``formal_entries`` so its weights
                    # and cached state remain directly reusable after admission.
                    keep = (
                        list(range(replacement))
                        + [len(self.entries)]
                        + list(range(replacement + 1, len(self.entries)))
                    )
                    prepared_new_state = (
                        self.objective.prepare_subset(
                            plan.add_state, keep, natural_support=True
                        )
                        if plan.add_state is not None
                        and callable(getattr(self.objective, "prepare_subset", None))
                        else None
                    )
                    compare_prepared = getattr(
                        self.objective, "compare_prepared_pools", None
                    )
                    compare = getattr(self.objective, "compare_pools", None)
                    natural_new_state = prepared_new_state
                    if (
                        callable(compare_prepared)
                        and base_state is not None
                        and prepared_new_state is not None
                    ):
                        old_score, shared_new_score, natural_new_state = compare_prepared(
                            base_state, prepared_new_state
                        )
                        formal_score = natural_new_state.score
                        shared_delta = self._increment(old_score, shared_new_score).reward
                        formal_increment = self._increment(baseline, formal_score)
                        natural_delta = formal_increment.reward
                        formal_delta = min(shared_delta, natural_delta)
                        support_is_valid = getattr(self.objective, "support_is_valid", None)
                        formal_valid = not callable(support_is_valid) or support_is_valid(natural_new_state)
                    elif callable(compare):
                        old_score, shared_new_score, natural_new_state = compare(
                            [entry.signal for entry in self.entries],
                            [entry.signal for entry in formal_entries],
                        )
                        formal_score = natural_new_state.score
                        # The shared-support delta prevents a candidate from
                        # winning merely by discarding difficult observations.
                        # Requiring the cached natural-support objective to be
                        # monotone as well preserves the pool invariant used by
                        # snapshots and subsequent add-only rewards.
                        shared_delta = self._increment(old_score, shared_new_score).reward
                        formal_increment = self._increment(baseline, formal_score)
                        natural_delta = formal_increment.reward
                        formal_delta = min(shared_delta, natural_delta)
                        support_is_valid = getattr(self.objective, "support_is_valid", None)
                        formal_valid = not callable(support_is_valid) or support_is_valid(natural_new_state)
                    else:
                        formal_score = self._score(formal_entries)
                        formal_increment = self._increment(baseline, formal_score)
                        formal_delta = formal_increment.reward
                        formal_valid = True
                    cache_prepared = getattr(
                        self.objective, "cache_prepared_pool", None
                    )
                    if (
                        formal_valid
                        and natural_new_state is not None
                        and callable(cache_prepared)
                    ):
                        cache_prepared(natural_new_state)
                plans[index] = _CandidatePlan(
                    plan.candidate,
                    replace(
                        plan.score,
                        pool_score=formal_score,
                        post_prune_delta=formal_delta if formal_valid else float("-inf"),
                        post_prune_increment=formal_increment,
                        formally_rechecked=True,
                        valid=bool(plan.score.valid and formal_valid),
                        reason=plan.score.reason if formal_valid else "insufficient_pool_support",
                        shaped_reward=plan.score.shaped_reward if formal_valid else -1.0,
                        positive_not_admitted=bool(
                            plan.score.delta_add > 0 and (not formal_valid or formal_delta <= self.min_delta)
                        ),
                    ),
                    plan.delete_index,
                )
        valid_delta_by_hash: dict[str, float] = {}
        for plan in plans:
            if plan.score.valid and self._finite_delta(plan.score.delta_add):
                valid_delta_by_hash.setdefault(
                    plan.score.candidate_hash, float(plan.score.delta_add)
                )
        valid_deltas = np.asarray(list(valid_delta_by_hash.values()), dtype=float)
        reward_scale = (
            max(float(np.median(np.abs(valid_deltas))), self.min_delta, 1e-5)
            if len(valid_deltas)
            else None
        )
        if reward_scale is not None:
            shaped_plans = []
            for plan in plans:
                score = plan.score
                shaped_reward = score.shaped_reward
                if score.valid and self._finite_delta(score.delta_add):
                    delta_add = float(score.delta_add)
                    shaped_reward = (
                        0.0
                        if delta_add == 0.0
                        else float(
                            np.copysign(
                                min(
                                    1.0 / (1.0 + reward_scale / abs(delta_add)),
                                    float(np.nextafter(1.0, 0.0)),
                                ),
                                delta_add,
                            )
                        )
                    )
                shaped_plans.append(
                    replace(
                        plan,
                        score=replace(
                            score,
                            shaped_reward=float(shaped_reward),
                            reward_scale=float(reward_scale),
                        ),
                    )
                )
            plans = shaped_plans
        return [plan.score for plan in plans]

    def consider_group(
        self,
        candidates: list[PoolEntry],
        precomputed: list[CandidateScore] | None = None,
    ) -> Admission:
        """Admit at most one formally rechecked monotonic pool transition."""
        scored = self.score_candidates(candidates) if precomputed is None else precomputed
        if len(scored) != len(candidates) or {
            item.candidate_hash for item in scored
        } != {item.expr_hash for item in candidates}:
            raise ValueError("precomputed candidate scores do not match admission group")
        pre_hash = stable_hash({
            "pool_version": self.version,
            "factors": [entry.expr_hash for entry in self.entries],
        })
        event_id = f"admission_{stable_hash({'pre_hash': pre_hash, 'candidate_hashes': [item.expr_hash for item in candidates], 'history_index': len(self.history)})[:20]}"
        if not scored:
            return Admission(False, None, None, 0.0, self.version, "empty_group", event_id, pre_hash, pre_hash)
        admissible = [
            item for item in scored
            if item.valid
            and item.formally_rechecked
            and np.isfinite(item.post_prune_delta)
            and not item.self_evicted
        ]
        if not admissible:
            admission = Admission(False, None, None, float("-inf"), self.version, "no_finite_candidate", event_id, pre_hash, pre_hash)
            self.history.append(asdict(admission))
            return admission
        best = max(admissible, key=lambda item: item.post_prune_delta)
        if best.post_prune_delta <= self.min_delta:
            admission = Admission(False, best.candidate_hash, None, best.post_prune_delta, self.version, "delta_below_threshold", event_id, pre_hash, pre_hash)
            self.history.append(asdict(admission))
            return admission
        candidate = next(item for item in candidates if item.expr_hash == best.candidate_hash)
        replaced_hash = None
        if len(self.entries) < self.capacity:
            self.entries.append(candidate)
        else:
            index = next(
                (index for index, entry in enumerate(self.entries) if entry.expr_hash == best.replaced_hash),
                None,
            )
            if index is None:
                raise RuntimeError("precomputed replacement is absent from frozen pool")
            replaced_hash = self.entries[index].expr_hash
            self.entries[index] = candidate
        self.version += 1
        self.invalidate_cache()
        post_hash = stable_hash({
            "pool_version": self.version,
            "factors": [entry.expr_hash for entry in self.entries],
        })
        admission = Admission(
            True,
            candidate.expr_hash,
            replaced_hash,
            best.post_prune_delta,
            self.version,
            "admitted",
            event_id,
            pre_hash,
            post_hash,
        )
        self.history.append(asdict(admission))
        return admission
