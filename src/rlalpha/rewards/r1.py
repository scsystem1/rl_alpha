from __future__ import annotations

import numpy as np

from .base import RewardObjective
from ..factors.records import PoolScore


class R1Objective(RewardObjective):
    def score_pool(self, signals: list[np.ndarray]) -> PoolScore:
        if not signals:
            return PoolScore(0.0, 0.0, tuple(), tuple())
        residual_signals, residual_label = self._neutralized_inputs(signals)
        daily, weights = self._daily_ic(residual_signals, residual_label)
        mean = float(np.nanmean(daily)) if np.isfinite(daily).any() else float("-inf")
        return PoolScore(mean, mean, tuple(map(float, daily)), tuple(map(float, weights)))

