from __future__ import annotations

from typing import Any, Protocol

from .models import Candidate, CandidateOutcome, SearchContext


class Searcher(Protocol):
    def propose(self, context: SearchContext, n: int) -> list[Candidate]: ...

    def observe(self, outcomes: list[CandidateOutcome]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
