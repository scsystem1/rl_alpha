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
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class SplitPanel:
    name: str
    dates: pd.DatetimeIndex
    permnos: np.ndarray
    features: dict[str, np.ndarray]
    daily_return: np.ndarray
    label: np.ndarray
    membership: np.ndarray
    eligibility: np.ndarray
    exposures: np.ndarray
    exposure_names: tuple[str, ...]
    target_slice: slice
    panel_fingerprint: str
    subtree_cache: SignalCache = field(default_factory=lambda: SignalCache(max_items=128), repr=False, compare=False)

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.target_slice]

    @property
    def common_mask(self) -> np.ndarray:
        return self.membership & self.eligibility & np.isfinite(self.exposures).all(axis=2)

    @property
    def cross_sectional_mask(self) -> np.ndarray:
        return self.membership & self.eligibility

    def evaluate(self, node: Node) -> np.ndarray:
        namespace = f"{self.panel_fingerprint}:{self.name}:{self.dates[0]}:{self.dates[-1]}"
        return evaluate(node, self.features, self.subtree_cache, eligibility_mask=self.cross_sectional_mask, cache_namespace=namespace)[self.target_slice]

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

    @cached_property
    def build_manifest(self) -> dict[str, object]:
        import yaml

        with (self.root / "build_manifest.yaml").open(encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if int(manifest.get("artifact_version", -1)) < 3:
            raise RuntimeError("processed panel is legacy_unverified; rebuild with membership-aware evaluator semantics")
        return manifest

    @cached_property
    def risk_build_manifest(self) -> dict[str, object]:
        import yaml

        path = self.root / "risk_build_manifest.yaml"
        if not path.exists():
            raise RuntimeError("risk panel is uncommitted or legacy_unverified; rebuild it before loading splits")
        with path.open(encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if int(manifest.get("artifact_version", -1)) < 3:
            raise RuntimeError("risk panel is legacy_unverified; rebuild with same-window regression semantics")
        if manifest.get("panel_build_fingerprint") != self.build_manifest.get("build_fingerprint"):
            raise RuntimeError("risk panel was built from a different processed panel")
        if not (self.root / "risk_exposures.zarr").exists():
            raise RuntimeError("risk manifest is present but the committed exposure array is missing")
        if not (self.root / "fundamental_lineage.parquet").exists():
            raise RuntimeError("risk manifest is present but point-in-time fundamental lineage is missing")
        return manifest

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
            eligibility=np.asarray(self._array("eligibility.zarr/trade_eligibility")[source], dtype=bool),
            exposures=np.asarray(exposure_array[source]),
            exposure_names=exposure_names,
            target_slice=target,
            panel_fingerprint=stable_hash({
                "panel": self.build_manifest["build_fingerprint"],
                "risk": self.risk_build_manifest["build_fingerprint"],
            }),
        )
