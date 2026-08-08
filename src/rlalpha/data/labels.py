from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import DateSplit


def next_close_forward_return(daily: pd.DataFrame, horizon: int = 20, split: DateSplit | None = None) -> pd.DataFrame:
    """Product of returns t+2..t+horizon+1, with an exit date in the same split."""
    frame = daily[["PERMNO", "DlyCalDt", "DlyRet"]].copy().sort_values(["PERMNO", "DlyCalDt"])
    frame["DlyCalDt"] = pd.to_datetime(frame["DlyCalDt"])
    labels = np.full(len(frame), np.nan)
    exits = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    for _, positions in frame.groupby("PERMNO", sort=False).indices.items():
        positions = np.asarray(positions)
        returns = pd.to_numeric(frame.iloc[positions]["DlyRet"], errors="coerce").to_numpy(float)
        dates = frame.iloc[positions]["DlyCalDt"].to_numpy(dtype="datetime64[ns]")
        for local in range(max(0, len(positions) - horizon - 1)):
            window = returns[local + 2 : local + horizon + 2]
            if len(window) == horizon and np.isfinite(window).all():
                labels[positions[local]] = np.prod(1.0 + window) - 1.0
                exits[positions[local]] = dates[local + horizon + 1]
    frame["forward_return_20d"] = labels
    frame["exit_date"] = exits
    if split is not None:
        outside = (frame["DlyCalDt"] < split.start) | (frame["DlyCalDt"] > split.end) | (frame["exit_date"] > split.end)
        frame.loc[outside, "forward_return_20d"] = np.nan
    return frame

