from __future__ import annotations

import base64
import pickle
import random
from typing import Any

from ..dsl.grammar import sample_ast
from .models import Candidate, CandidateOutcome, SearchContext


class RandomSearcher:
    def __init__(self, seed: int, max_depth: int = 6):
        self.rng = random.Random(seed)
        self.max_depth = max_depth
        self.observed = 0

    def propose(self, context: SearchContext, n: int) -> list[Candidate]:
        return [Candidate(sample_ast(self.rng, self.max_depth), "random") for _ in range(n)]

    def observe(self, outcomes: list[CandidateOutcome]) -> None:
        self.observed += len(outcomes)

    def state_dict(self) -> dict[str, Any]:
        encoded = base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii")
        return {"rng_state": encoded, "observed": self.observed, "max_depth": self.max_depth}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"])))
        self.observed = int(state["observed"])
        self.max_depth = int(state["max_depth"])
