from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoolScore:
    objective: float
    mean_ic: float
    daily_ic: tuple[float, ...]
    weights: tuple[float, ...]
    standard_error: float = float("nan")


@dataclass(frozen=True)
class CandidateScore:
    candidate_hash: str
    pool_score: PoolScore
    delta_objective: float
    shaped_reward: float
    replaced_hash: str | None = None


@dataclass
class PoolEntry:
    expression: str
    expr_hash: str
    signal: object = field(repr=False)
    metadata: dict[str, object] = field(default_factory=dict)
