from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


SPLITS = {
    "train": DateSplit("train", pd.Timestamp("2010-01-01"), pd.Timestamp("2018-12-31")),
    "validation": DateSplit("validation", pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")),
    "test": DateSplit("test", pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31")),
}


def split_mask(dates: pd.Series | pd.DatetimeIndex, name: str) -> np.ndarray:
    import numpy as np

    split = SPLITS[name]
    values = pd.to_datetime(dates)
    return np.asarray((values >= split.start) & (values <= split.end))

