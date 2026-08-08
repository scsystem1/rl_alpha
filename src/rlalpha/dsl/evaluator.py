from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .ast import Call, Constant, Feature, Node, Window


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float))


def _rolling_unary(values: np.ndarray, window: int, operator: str) -> np.ndarray:
    frame = _frame(values)
    minimum = max(1, math.ceil(0.8 * window))
    rolling = frame.rolling(window, min_periods=minimum)
    if operator == "Mean": result = rolling.mean()
    elif operator == "Sum": result = rolling.sum()
    elif operator == "Std": result = rolling.std(ddof=1)
    elif operator == "Var": result = rolling.var(ddof=1)
    elif operator == "Max": result = rolling.max()
    elif operator == "Min": result = rolling.min()
    elif operator == "Med": result = rolling.median()
    elif operator == "Mad": result = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    elif operator == "TSRank": result = rolling.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    elif operator == "WMA": result = rolling.apply(lambda x: float(np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()), raw=True)
    elif operator == "EMA": result = frame.ewm(span=window, min_periods=minimum, adjust=False).mean()
    else: raise ValueError(operator)
    output = result.to_numpy(dtype=float, copy=True)
    if operator in {"Std", "Var"}:
        output[np.abs(output) < 1e-12] = np.nan
    return output


def evaluate(node: Node, features: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = next((np.asarray(values).shape for values in features.values()), None)
    if shape is None:
        raise ValueError("features cannot be empty")

    def visit(current: Node) -> np.ndarray:
        if isinstance(current, Feature):
            if current.name not in features:
                raise KeyError(current.name)
            values = np.asarray(features[current.name], dtype=float)
            if values.shape != shape:
                raise ValueError("all features must share [time, asset] shape")
            return values.copy()
        if isinstance(current, Constant):
            return np.full(shape, current.value, dtype=float)
        if isinstance(current, Window):
            raise TypeError("window cannot be evaluated as a signal")
        assert isinstance(current, Call)
        operator = current.operator
        args = [visit(arg) for arg in current.args if not isinstance(arg, Window)]
        if operator == "Abs": return np.abs(args[0])
        if operator == "Sign": return np.sign(args[0])
        if operator == "Log": return np.log(np.abs(args[0]) + 1e-6)
        if operator == "Add": return args[0] + args[1]
        if operator == "Sub": return args[0] - args[1]
        if operator == "Mul": return args[0] * args[1]
        if operator == "Div":
            denominator = args[1]
            sign = np.where(denominator < 0, -1.0, 1.0)
            return args[0] / (sign * np.maximum(np.abs(denominator), 1e-6))
        if operator == "Greater": return (args[0] > args[1]).astype(float)
        if operator == "Less": return (args[0] < args[1]).astype(float)
        if operator in {"CSRank", "CSZScore"}:
            frame = _frame(args[0])
            if operator == "CSRank": return frame.rank(axis=1, pct=True).to_numpy(float)
            mean = frame.mean(axis=1, skipna=True)
            std = frame.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
            return frame.sub(mean, axis=0).div(std, axis=0).to_numpy(float)
        window_node = current.args[-1]
        assert isinstance(window_node, Window)
        window = window_node.value
        if operator == "Ref": return _frame(args[0]).shift(window).to_numpy(float)
        if operator == "Delta": return args[0] - _frame(args[0]).shift(window).to_numpy(float)
        if operator in {"Cov", "Corr"}:
            minimum = max(2, math.ceil(0.8 * window))
            left, right = _frame(args[0]), _frame(args[1])
            pieces = []
            for column in left.columns:
                roll = left[column].rolling(window, min_periods=minimum)
                pieces.append(roll.cov(right[column]) if operator == "Cov" else roll.corr(right[column]))
            return pd.concat(pieces, axis=1).to_numpy(float)
        return _rolling_unary(args[0], window, operator)

    with np.errstate(all="ignore"):
        return visit(node)
