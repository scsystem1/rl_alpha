from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .splits import SPLITS
from ..dsl.ast import Node
from ..dsl.evaluator import evaluate
from ..factors.cache import SignalCache


@dataclass(frozen=True)
class SplitPanel:
    name: str
    dates: pd.DatetimeIndex
    permnos: np.ndarray
    features: dict[str, np.ndarray]
    daily_return: np.ndarray
    label: np.ndarray
    membership: np.ndarray
    exposures: np.ndarray
    exposure_names: tuple[str, ...]
    target_slice: slice
    subtree_cache: SignalCache = field(default_factory=lambda: SignalCache(max_items=128), repr=False, compare=False)

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.target_slice]

    @property
    def common_mask(self) -> np.ndarray:
        close = self.features["$close"]
        volume = self.features["$volume"]
        return self.membership & np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0) & np.isfinite(self.exposures).all(axis=2)

    def evaluate(self, node: Node) -> np.ndarray:
        return evaluate(node, self.features, self.subtree_cache)[self.target_slice]

    def target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[self.target_slice]


class PanelStore:
    """Read dense artifacts directly by array path, avoiding slow group scans."""

    def __init__(self, processed_root: str | Path):
        self.root = Path(processed_root) / "panel"

    @cached_property
    def index(self) -> dict[str, object]:
        with (self.root / "index.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(self.index["dates"]))

    @property
    def permnos(self) -> np.ndarray:
        return np.asarray(self.index["permnos"], dtype=np.int64)

    def _array(self, relative: str) -> zarr.Array:
        return zarr.open_array(str(self.root / relative), mode="r")

    def load_split(self, name: str, history: int = 252, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> SplitPanel:
        if name not in SPLITS:
            raise KeyError(name)
        split = SPLITS[name]
        requested_start = split.start if start is None else pd.Timestamp(start)
        requested_end = split.end if end is None else pd.Timestamp(end)
        if requested_start < split.start or requested_end > split.end or requested_start > requested_end:
            raise ValueError(f"requested interval must stay inside split {name}")
        selected = np.flatnonzero((self.dates >= requested_start) & (self.dates <= requested_end))
        if not len(selected):
            raise ValueError(f"split {name} contains no dates")
        start = max(0, int(selected[0]) - history)
        stop = int(selected[-1]) + 1
        source = slice(start, stop)
        target = slice(int(selected[0]) - start, stop - start)
        features = {f"${key}": np.asarray(self._array(f"features.zarr/{key}")[source]) for key in ("open", "high", "low", "close", "volume", "return")}
        exposure_array = self._array("risk_exposures.zarr/exposures")
        exposure_group = zarr.open_group(str(self.root / "risk_exposures.zarr"), mode="r")
        exposure_names = tuple(exposure_group.attrs.get("columns", [f"exposure_{index}" for index in range(exposure_array.shape[2])]))
        return SplitPanel(
            name=name,
            dates=self.dates[source],
            permnos=self.permnos,
            features=features,
            daily_return=np.asarray(self._array("returns.zarr/daily_total_return")[source]),
            label=np.asarray(self._array("returns.zarr/forward_return_20d")[source]),
            membership=np.asarray(self._array("membership.zarr/membership")[source], dtype=bool),
            exposures=np.asarray(exposure_array[source]),
            exposure_names=exposure_names,
            target_slice=target,
        )
