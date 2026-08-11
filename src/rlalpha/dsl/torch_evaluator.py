from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .ast import Call, Constant, Feature, Node, Window
from .evaluator import _rolling_unary


def evaluate_torch(node: Node, features: Mapping[str, object], *, eligibility_mask: object | None = None):
    """Evaluate with Torch tensors; pandas-backed rolling kernels preserve CPU semantics."""
    import torch

    first = next(iter(features.values()), None)
    if first is None:
        raise ValueError("features cannot be empty")
    template = first if isinstance(first, torch.Tensor) else torch.as_tensor(first)
    shape = tuple(template.shape)
    device = template.device
    dtype = torch.float64
    if eligibility_mask is None:
        eligibility = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        eligibility = torch.as_tensor(eligibility_mask, dtype=torch.bool, device=device)
        if tuple(eligibility.shape) != shape:
            raise ValueError("cross-sectional eligibility mask shape differs from features")

    def tensor(values: object):
        # Zarr/pandas may expose read-only views; Torch tensors need ownership.
        if isinstance(values, torch.Tensor):
            return values.to(dtype=dtype, device=device)
        return torch.as_tensor(np.array(values, copy=True), dtype=dtype, device=device)

    def pandas_result(values: np.ndarray):
        return tensor(values)

    def visit(current: Node):
        if isinstance(current, Feature):
            value = tensor(features[current.name])
            if tuple(value.shape) != shape:
                raise ValueError("all features must share [time, asset] shape")
            return value.clone()
        if isinstance(current, Constant):
            return torch.full(shape, current.value, dtype=dtype, device=device)
        if isinstance(current, Window):
            raise TypeError("window cannot be evaluated as a signal")
        assert isinstance(current, Call)
        operator = current.operator
        args = [visit(arg) for arg in current.args if not isinstance(arg, Window)]
        if operator == "Abs": return torch.abs(args[0])
        if operator == "Sign": return torch.sign(args[0])
        if operator == "Log": return torch.log(torch.abs(args[0]) + 1e-6)
        if operator == "Add": return args[0] + args[1]
        if operator == "Sub": return args[0] - args[1]
        if operator == "Mul": return args[0] * args[1]
        if operator == "Div":
            sign = torch.where(args[1] < 0, -1.0, 1.0)
            return args[0] / (sign * torch.clamp(torch.abs(args[1]), min=1e-6))
        if operator in {"Greater", "Less"}:
            finite = torch.isfinite(args[0]) & torch.isfinite(args[1])
            output = torch.full(shape, torch.nan, dtype=dtype, device=device)
            compared = args[0] > args[1] if operator == "Greater" else args[0] < args[1]
            output[finite] = compared[finite].to(dtype)
            return output
        if operator in {"CSRank", "CSZScore"}:
            values = torch.where(eligibility & torch.isfinite(args[0]), args[0], torch.nan)
            frame = pd.DataFrame(values.detach().cpu().numpy())
            if operator == "CSRank":
                return pandas_result(frame.rank(axis=1, pct=True).to_numpy(float))
            mean = frame.mean(axis=1, skipna=True)
            std = frame.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
            return pandas_result(frame.sub(mean, axis=0).div(std, axis=0).to_numpy(float))
        window_node = current.args[-1]
        assert isinstance(window_node, Window)
        window = window_node.value
        if operator == "Ref":
            output = torch.full_like(args[0], torch.nan)
            output[window:] = args[0][:-window]
            return output
        if operator == "Delta":
            output = torch.full_like(args[0], torch.nan)
            output[window:] = args[0][window:] - args[0][:-window]
            return output
        if operator in {"Cov", "Corr"}:
            minimum = max(2, math.ceil(0.8 * window))
            left = pd.DataFrame(args[0].detach().cpu().numpy())
            right = pd.DataFrame(args[1].detach().cpu().numpy())
            columns = []
            for column in left.columns:
                rolling = left[column].rolling(window, min_periods=minimum)
                columns.append(rolling.cov(right[column]) if operator == "Cov" else rolling.corr(right[column]))
            return pandas_result(pd.concat(columns, axis=1).to_numpy(float))
        values = args[0].detach().cpu().numpy()
        return pandas_result(_rolling_unary(values, window, operator))

    return visit(node)
