from __future__ import annotations

import numpy as np

from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.parser import parse_expression
from rlalpha.dsl.validity import validate_signal
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.r0 import R0Objective
from rlalpha.rewards.r1 import R1Objective
from rlalpha.rewards.r2_lcb import R2LCBObjective


def test_synthetic_dsl_to_pool_reward_pipeline():
    rng = np.random.default_rng(10)
    days, assets = 300, 120
    returns = rng.normal(0, 0.01, size=(days, assets))
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    features = {"$open": close, "$high": close * 1.01, "$low": close * 0.99, "$close": close, "$volume": np.exp(rng.normal(10, 1, size=(days, assets))), "$return": returns}
    nodes = [parse_expression("Delta($close,20)"), parse_expression("Mean($return,20)"), parse_expression("Std($return,20)")]
    signals = [evaluate(node, features) for node in nodes]
    membership = np.ones((days, assets), dtype=bool)
    assert validate_signal(signals[0], membership).valid
    label = np.roll(signals[0], -2, axis=0) + rng.normal(size=(days, assets))
    label[-2:] = np.nan
    exposure = np.stack([np.column_stack([np.ones(assets), rng.normal(size=assets)]) for _ in range(days)])
    for objective in [R0Objective(label, membership), R1Objective(label, membership, exposure), R2LCBObjective(label, membership, exposure, hac_lag=5)]:
        pool = PoolManager(objective, capacity=2)
        admission = pool.consider_group([PoolEntry(node.canonical(), node.expr_hash, signal) for node, signal in zip(nodes, signals)])
        assert admission.admitted
        assert len(pool.entries) == 1
