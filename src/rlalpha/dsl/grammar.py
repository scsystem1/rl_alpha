from __future__ import annotations

import random

from .ast import Call, Constant, Feature, Node, Window
from .operators import BINARY, CONSTANTS, CROSS_SECTIONAL, FEATURES, PAIR_ROLLING, ROLLING, UNARY, WINDOWS


def sample_ast(rng: random.Random, max_depth: int = 6) -> Node:
    def build(depth: int, require_feature: bool = True) -> Node:
        if depth <= 1 or rng.random() < 0.22:
            return Feature(rng.choice(sorted(FEATURES))) if require_feature or rng.random() < 0.7 else Constant(rng.choice(CONSTANTS))
        family = rng.choice(["unary", "binary", "rolling", "pair", "cross"])
        if family == "unary":
            return Call(rng.choice(sorted(UNARY)), (build(depth - 1),))
        if family == "cross":
            return Call(rng.choice(sorted(CROSS_SECTIONAL)), (build(depth - 1),))
        if family == "binary":
            return Call(rng.choice(sorted(BINARY)), (build(depth - 1), build(depth - 1, require_feature=False)))
        if family == "rolling":
            return Call(rng.choice(sorted(ROLLING)), (build(depth - 1), Window(rng.choice(WINDOWS))))
        return Call(rng.choice(sorted(PAIR_ROLLING)), (build(depth - 1), build(depth - 1), Window(rng.choice(WINDOWS))))

    for _ in range(100):
        try:
            node = build(max_depth)
            if node.nodes <= 21 and node.lookback <= 252:
                return node
        except ValueError:
            continue
    return Feature("$close")

