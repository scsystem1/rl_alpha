from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..dsl.ast import Node
from ..utils.hashing import sha256_text


@dataclass(frozen=True)
class Candidate:
    node: Node | None
    generator: str
    parents: tuple[str, ...] = ()
    raw_text: str | None = None

    @property
    def expression(self) -> str:
        return self.node.canonical() if self.node is not None else (self.raw_text or "")

    @property
    def expr_hash(self) -> str:
        return self.node.expr_hash if self.node is not None else sha256_text("invalid:" + (self.raw_text or ""))


@dataclass(frozen=True)
class CandidateOutcome:
    expr_hash: str
    expression: str
    valid: bool
    reason: str
    market_evaluated: bool
    delta_objective: float = 0.0
    shaped_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchContext:
    pool_version: int
    pool_formulas: tuple[str, ...]
    pool_weights: tuple[float, ...]
    train_objective: float
    valid_unique_evaluations: int
    budget: int
    history_summary: tuple[dict[str, Any], ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetLedger:
    limit: int
    raw_proposals: int = 0
    valid_unique_evaluations: int = 0
    duplicates: int = 0
    invalid: int = 0
    tokens: int = 0
    gpu_seconds: float = 0.0

    @property
    def exhausted(self) -> bool:
        return self.valid_unique_evaluations >= self.limit

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.valid_unique_evaluations)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BudgetLedger":
        return cls(**state)
