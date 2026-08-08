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

    def score_pool(self, signals: list[np.ndarray]) -> PoolScore:
        if not signals:
            return PoolScore(0.0, 0.0, tuple(), tuple(), 0.0)
        residual_signals, residual_label = self._neutralized_inputs(signals)
        daily, weights = self._daily_ic(residual_signals, residual_label)
        objective, mean, standard_error = lcb_score(daily, self.hac_lag, self.critical_value)
        return PoolScore(objective, mean, tuple(map(float, daily)), tuple(map(float, weights)), standard_error)
