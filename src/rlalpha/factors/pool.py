from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .records import CandidateScore, PoolEntry, PoolScore


@dataclass(frozen=True)
class Admission:
    admitted: bool
    candidate_hash: str | None
    replaced_hash: str | None
    delta: float
    pool_version: int


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
                outcomes.append(CandidateScore(candidate.expr_hash, baseline, 0.0, -0.5))
                continue
            if len(self.entries) < self.capacity:
                score = self._score(self.entries + [candidate])
            else:
                alternatives = [self._score(self.entries[:index] + [candidate] + self.entries[index + 1 :]) for index in range(len(self.entries))]
                score = max(alternatives, key=lambda item: item.objective)
            delta = score.objective - baseline.objective
            outcomes.append(CandidateScore(candidate.expr_hash, score, delta, max(-1.0, min(1.0, 100.0 * delta))))
        return outcomes

    def consider_group(self, candidates: list[PoolEntry]) -> Admission:
        """Score against one frozen pool and admit at most one candidate."""
        baseline = self._score(self.entries)
        scored = self.score_candidates(candidates)
        if not scored:
            return Admission(False, None, None, 0.0, self.version)
        best = max(scored, key=lambda item: item.delta_objective)
        if best.delta_objective <= self.min_delta:
            admission = Admission(False, best.candidate_hash, None, best.delta_objective, self.version)
            self.history.append(asdict(admission))
            return admission
        candidate = next(item for item in candidates if item.expr_hash == best.candidate_hash)
        replaced = None
        if len(self.entries) < self.capacity:
            self.entries.append(candidate)
        else:
            alternatives = [(self._score(self.entries[:index] + [candidate] + self.entries[index + 1 :]), index) for index in range(len(self.entries))]
            _, index = max(alternatives, key=lambda item: item[0].objective)
            replaced = self.entries[index].expr_hash
            self.entries[index] = candidate
        self.version += 1
        admission = Admission(True, candidate.expr_hash, replaced, best.delta_objective, self.version)
        self.history.append(asdict(admission))
        return admission

