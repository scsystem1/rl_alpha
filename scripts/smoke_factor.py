from __future__ import annotations

import json
import numpy as np

from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.parser import parse_expression


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(300, 20)), axis=0))
    features = {name: close.copy() for name in ["$open", "$high", "$low", "$close", "$volume"]}
    features["$return"] = np.vstack([np.full((1, 20), np.nan), close[1:] / close[:-1] - 1])
    node = parse_expression("CSZScore(Div(Delta($close,20),Std($return,20)))")
    signal = evaluate(node, features)
    print(json.dumps({"expression": node.canonical(), "hash": node.expr_hash, "coverage": float(np.isfinite(signal).mean())}, indent=2))
