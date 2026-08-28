from __future__ import annotations

import numpy as np

from .base import RewardObjective
from ..factors.records import PoolScore


class R1Objective(RewardObjective):
    def _objective_inputs(
        self, signals: list[np.ndarray], raw_common: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        return self._neutralized_inputs(signals, raw_common)

    def objective_name(self) -> str:
        return "r1"
