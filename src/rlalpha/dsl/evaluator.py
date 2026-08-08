from __future__ import annotations

import math
import os
from typing import Mapping, Protocol

import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.signal import lfilter
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from numba import njit, prange

from .ast import Call, Constant, Feature, Node, Window


class ArrayCache(Protocol):
    def get(self, key: str) -> np.ndarray | None: ...
    def put(self, key: str, value: np.ndarray, permanent: bool = False) -> None: ...


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float))


@njit(cache=True, parallel=True)
def _rolling_mad(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    rows, columns = values.shape
    output = np.full(values.shape, np.nan)
    for column in prange(columns):
        buffer = np.empty(window)
        deviations = np.empty(window)
        for row in range(rows):
            start = max(0, row - window + 1)
            count = 0
            for source in range(start, row + 1):
                value = values[source, column]
                if np.isfinite(value):
                    buffer[count] = value
                    count += 1
            if count >= minimum:
                median = np.median(buffer[:count])
                for index in range(count):
                    deviations[index] = abs(buffer[index] - median)
                output[row, column] = np.median(deviations[:count])
    return output


def _rolling_unary(values: np.ndarray, window: int, operator: str) -> np.ndarray:
    minimum = max(1, math.ceil(0.8 * window))
    values = np.asarray(values, dtype=float)
    if operator in {"Mean", "Sum", "Std", "Var"}:
        total, count = _window_sum_count(values, window)
        if operator == "Sum":
            output = total
        elif operator == "Mean":
            output = total / count
        else:
            squares, _ = _window_sum_count(values * values, window)
            with np.errstate(all="ignore"):
                variance = (squares - total * total / count) / (count - 1)
            output = np.sqrt(np.maximum(variance, 0)) if operator == "Std" else variance
        output[count < minimum] = np.nan
    elif operator == "Max": output = bn.move_max(values, window, minimum, axis=0)
    elif operator == "Min": output = bn.move_min(values, window, minimum, axis=0)
    elif operator == "Med": output = bn.move_median(values, window, minimum, axis=0)
    elif operator == "Mad": output = _rolling_mad(values, window, minimum)
    elif operator == "TSRank":
        normalized = bn.move_rank(values, window, minimum, axis=0)
        count = _window_sum_count(np.where(np.isfinite(values), 1.0, np.nan), window)[1]
        output = ((normalized + 1) * 0.5 * (count - 1) + 1) / count
    elif operator == "WMA":
        finite = np.isfinite(values)
        weights = np.arange(window, 0, -1, dtype=float)
        numerator = lfilter(weights, [1.0], np.where(finite, values, 0.0), axis=0)
        denominator = lfilter(weights, [1.0], finite.astype(float), axis=0)
        count = lfilter(np.ones(window), [1.0], finite.astype(float), axis=0)
        output = numerator / denominator
        output[count < minimum] = np.nan
    elif operator == "EMA": output = _frame(values).ewm(span=window, min_periods=minimum, adjust=False).mean().to_numpy(float)
    else: raise ValueError(operator)
    if operator in {"Std", "Var"}:
        output[np.abs(output) < 1e-12] = np.nan
    return output


def _window_sum_count(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    total_cumulative = np.cumsum(np.where(finite, values, 0.0), axis=0)
    count_cumulative = np.cumsum(finite, axis=0, dtype=float)
    zeros = np.zeros((min(window, len(values)), values.shape[1]))
    total_prior = np.vstack([zeros, total_cumulative[:-window]]) if len(values) > window else zeros
    count_prior = np.vstack([zeros, count_cumulative[:-window]]) if len(values) > window else zeros
    return total_cumulative - total_prior, count_cumulative - count_prior


def _rolling_pair(left: np.ndarray, right: np.ndarray, window: int, operator: str) -> np.ndarray:
    minimum = max(2, math.ceil(0.8 * window))
    common = np.isfinite(left) & np.isfinite(right)
    x, y = np.where(common, left, np.nan), np.where(common, right, np.nan)
    sx, count = _window_sum_count(x, window)
    sy, _ = _window_sum_count(y, window)
    sxy, _ = _window_sum_count(x * y, window)
    covariance_numerator = sxy - sx * sy / count
    if operator == "Cov":
        result = covariance_numerator / (count - 1)
    else:
        sx2, _ = _window_sum_count(x * x, window)
        sy2, _ = _window_sum_count(y * y, window)
        result = covariance_numerator / np.sqrt(np.maximum((sx2 - sx * sx / count) * (sy2 - sy * sy / count), 0))
    result[count < minimum] = np.nan
    return result


def evaluate(node: Node, features: Mapping[str, np.ndarray], cache: ArrayCache | None = None) -> np.ndarray:
    shape = next((np.asarray(values).shape for values in features.values()), None)
    if shape is None:
        raise ValueError("features cannot be empty")

    def compute(current: Node) -> np.ndarray:
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
            return _rolling_pair(args[0], args[1], window, operator)
        return _rolling_unary(args[0], window, operator)

    def visit(current: Node) -> np.ndarray:
        cacheable = isinstance(current, Call)
        if cache is not None and cacheable:
            cached = cache.get(current.expr_hash)
            if cached is not None:
                return np.asarray(cached)
        result = compute(current)
        if cache is not None and cacheable:
            cache.put(current.expr_hash, result)
        return result

    with np.errstate(all="ignore"):
        return visit(node)
