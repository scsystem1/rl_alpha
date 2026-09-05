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
class PoolIncrement:
    mean_delta: float
    standard_error: float
    penalty: float
    reward: float
    valid_days: int = 0
    fold_means: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_means", tuple(self.fold_means))


@dataclass(frozen=True)
class CandidateScore:
    candidate_hash: str
    pool_score: PoolScore
    # Kept as the public search fitness name. Its value is the add-only delta,
    # never the result of the later capacity pruning decision.
    delta_objective: float
    shaped_reward: float
    replaced_hash: str | None = None
    valid: bool = True
    reason: str = "ok"
    delta_add: float | None = None
    saliency: tuple[float, ...] = ()
    eviction_candidates: tuple[str, ...] = ()
    post_prune_delta: float = float("-inf")
    self_evicted: bool = False
    formally_rechecked: bool = False
    positive_not_admitted: bool = False
    reward_valid_days: int = 0
    reward_valid_observations: int = 0
    reward_valid_day_rate: float = 0.0
    reward_observation_rate: float = 0.0
    reward_scale: float | None = None
    add_increment: PoolIncrement | None = None
    post_prune_increment: PoolIncrement | None = None

    def __post_init__(self) -> None:
        if self.delta_add is None:
            object.__setattr__(self, "delta_add", self.delta_objective)


@dataclass
class PoolEntry:
    expression: str
    expr_hash: str
    signal: object = field(repr=False)
    metadata: dict[str, object] = field(default_factory=dict)
