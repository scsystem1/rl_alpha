from __future__ import annotations

from .base import RewardObjective


class R0Objective(RewardObjective):
    def objective_name(self) -> str:
        return "r0"
