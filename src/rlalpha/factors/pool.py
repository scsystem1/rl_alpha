from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .records import CandidateScore, PoolEntry, PoolScore
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


class PoolManager:
    def __init__(self, objective: object, capacity: int = 20, min_delta: float = 1e-5):
        self.objective = objective
        self.capacity = capacity
        self.min_delta = min_delta
        self.entries: list[PoolEntry] = []
        self.version = 0
        self.history: list[dict[str, object]] = []

    @property
    def hashes(self) -> set[str]:
        return {entry.expr_hash for entry in self.entries}

    def _score(self, entries: list[PoolEntry]) -> PoolScore:
        return self.objective.score_pool([entry.signal for entry in entries])

    def score_candidates(self, candidates: Iterable[PoolEntry]) -> list[CandidateScore]:
        baseline = self._score(self.entries)
        outcomes = []
        for candidate in candidates:
            if candidate.expr_hash in self.hashes:
                outcomes.append(CandidateScore(candidate.expr_hash, baseline, 0.0, -0.5, valid=False, reason="exact_duplicate"))
                continue
            if not np.isfinite(baseline.objective):
                outcomes.append(CandidateScore(candidate.expr_hash, baseline, float("-inf"), -1.0, valid=False, reason="non_finite_baseline"))
                continue
            replaced_hash = None
            if len(self.entries) < self.capacity:
                score = self._score(self.entries + [candidate])
            else:
                alternatives = [(self._score(self.entries[:index] + [candidate] + self.entries[index + 1 :]), index) for index in range(len(self.entries))]
                finite_alternatives = [item for item in alternatives if np.isfinite(item[0].objective)]
                if not finite_alternatives:
                    score = alternatives[0][0]
                    outcomes.append(CandidateScore(candidate.expr_hash, score, float("-inf"), -1.0, valid=False, reason="non_finite_objective"))
                    continue
                score, replacement_index = max(finite_alternatives, key=lambda item: item[0].objective)
                replaced_hash = self.entries[replacement_index].expr_hash
            if not np.isfinite(score.objective):
                outcomes.append(CandidateScore(candidate.expr_hash, score, float("-inf"), -1.0, replaced_hash, False, "non_finite_objective"))
                continue
            delta = score.objective - baseline.objective
            if not np.isfinite(delta):
                outcomes.append(CandidateScore(candidate.expr_hash, score, float("-inf"), -1.0, replaced_hash, False, "non_finite_delta"))
                continue
            shaped = float(np.clip(100.0 * delta, -1.0, 1.0))
            outcomes.append(CandidateScore(candidate.expr_hash, score, float(delta), shaped, replaced_hash))
        return outcomes

    def consider_group(self, candidates: list[PoolEntry], precomputed: list[CandidateScore] | None = None) -> Admission:
        """Score against one frozen pool and admit at most one candidate."""
        scored = self.score_candidates(candidates) if precomputed is None else precomputed
        if len(scored) != len(candidates) or {item.candidate_hash for item in scored} != {item.expr_hash for item in candidates}:
            raise ValueError("precomputed candidate scores do not match admission group")
        pre_hash = stable_hash({"pool_version": self.version, "factors": [entry.expr_hash for entry in self.entries]})
        event_id = f"admission_{stable_hash({'pre_hash': pre_hash, 'candidate_hashes': [item.expr_hash for item in candidates], 'history_index': len(self.history)})[:20]}"
        if not scored:
            return Admission(False, None, None, 0.0, self.version, "empty_group", event_id, pre_hash, pre_hash)
        admissible = [item for item in scored if item.valid and np.isfinite(item.delta_objective)]
        if not admissible:
            admission = Admission(False, None, None, float("-inf"), self.version, "no_finite_candidate", event_id, pre_hash, pre_hash)
            self.history.append(asdict(admission))
            return admission
        best = max(admissible, key=lambda item: item.delta_objective)
        if best.delta_objective <= self.min_delta:
            admission = Admission(False, best.candidate_hash, None, best.delta_objective, self.version, "delta_below_threshold", event_id, pre_hash, pre_hash)
            self.history.append(asdict(admission))
            return admission
        candidate = next(item for item in candidates if item.expr_hash == best.candidate_hash)
        replaced = None
        if len(self.entries) < self.capacity:
            self.entries.append(candidate)
        else:
            index = next((index for index, entry in enumerate(self.entries) if entry.expr_hash == best.replaced_hash), None)
            if index is None:
                raise RuntimeError("precomputed replacement is absent from frozen pool")
            replaced = self.entries[index].expr_hash
            self.entries[index] = candidate
        self.version += 1
        post_hash = stable_hash({"pool_version": self.version, "factors": [entry.expr_hash for entry in self.entries]})
        admission = Admission(True, candidate.expr_hash, replaced, best.delta_objective, self.version, "admitted", event_id, pre_hash, post_hash)
        self.history.append(asdict(admission))
        return admission
