from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_parquet(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    for column in frame.columns:
        lower = column.lower()
        if lower.endswith("dt") or lower in {"datadate", "dlycaldt", "mbrstartdt", "mbrenddt", "delistingdt"}:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame

