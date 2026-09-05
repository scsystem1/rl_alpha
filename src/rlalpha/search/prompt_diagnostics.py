from __future__ import annotations

from ..rewards.factory import prompt_objective_for
from ..rewards.walk_forward import DEFAULT_TIME_FOLDS, WalkForwardObjective
from .models import TrainPoolSummary


class PoolPromptDiagnostics:
    """One canonical, train-only numerical summary per frozen pool version."""

    def __init__(self, panel, pool, reward_config=None):
        self.pool = pool
        current = pool.objective
        self.objective = (current if isinstance(current, WalkForwardObjective)
            and current.time_folds == DEFAULT_TIME_FOLDS else prompt_objective_for(panel, reward_config))
        self._key = None
        self._summary = None

    def __call__(self):
        key = (self.pool.version, tuple((entry.expr_hash, id(entry.signal)) for entry in self.pool.entries))
        if key != self._key:
            state = (self.pool.prepared_state() if self.objective is self.pool.objective
                     else self.objective.prepare_pool([entry.signal for entry in self.pool.entries]))
            self._summary = TrainPoolSummary(**self.objective.prompt_summary(state))
            self._key = key
        return self._summary
