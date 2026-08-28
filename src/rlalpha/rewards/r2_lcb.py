from __future__ import annotations

import numpy as np

from .base import RewardObjective
from .statistics import lcb_score
from ..factors.records import PoolScore


class R2LCBObjective(RewardObjective):
    def __init__(self, *args, hac_lag: int = 20, critical_value: float = 1.645, **kwargs):
        super().__init__(*args, **kwargs)
        self.hac_lag = hac_lag
        self.critical_value = critical_value

    def _objective_inputs(
        self, signals: list[np.ndarray], raw_common: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        return self._neutralized_inputs(signals, raw_common)

    def _score_from_daily(self, daily: np.ndarray, weights: np.ndarray) -> PoolScore:
        objective, mean, standard_error = lcb_score(daily, self.hac_lag, self.critical_value)
        if int(np.isfinite(daily).sum()) < self.required_valid_days():
            objective = float("-inf")
        return PoolScore(objective, mean, tuple(map(float, daily)), tuple(map(float, weights)), standard_error)

    def objective_name(self) -> str:
        return "r2"
