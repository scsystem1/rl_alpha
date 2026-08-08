from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import zarr

from .adjustments import apply_crsp_adjustments, fill_missing_delisting_returns
from .discovery import discover_data_files
from .membership import ACTIVE_MEMBERSHIP_FLAGS
from .splits import SPLITS
from .validate import validate_raw_bundle
from ..utils.hashing import files_fingerprint
from ..utils.hashing import stable_hash
from ..utils.io import write_json, write_yaml

PANEL_ARTIFACT_VERSION = 2


def _write_dense_artifacts(adjusted: pd.DataFrame, membership_intervals: pd.DataFrame, panel_root: Path) -> dict[str, Any]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(adjusted["DlyCalDt"]).unique()))
    permnos = np.asarray(sorted(pd.to_numeric(adjusted["PERMNO"]).astype(int).unique()), dtype=np.int64)
    date_codes = pd.Categorical(pd.to_datetime(adjusted["DlyCalDt"]), categories=dates).codes
    permno_codes = pd.Categorical(pd.to_numeric(adjusted["PERMNO"]).astype(int), categories=permnos).codes
    shape = (len(dates), len(permnos))
    feature_columns = {"open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close", "volume": "adj_volume", "return": "DlyRet"}
    dense: dict[str, np.ndarray] = {}
    for name, column in feature_columns.items():
        values = np.full(shape, np.nan, dtype=np.float32)
        values[date_codes, permno_codes] = pd.to_numeric(adjusted[column], errors="coerce").to_numpy(np.float32)
        dense[name] = values
    feature_group = zarr.open_group(str(panel_root / "features.zarr"), mode="w")
    for name, values in dense.items():
        feature_group.create_array(name, data=values, chunks=(252, min(256, shape[1])), overwrite=True)
    returns = dense["return"]
    labels = np.full(shape, np.nan, dtype=np.float32)
    if shape[0] > 21:
        compounded = np.ones((shape[0] - 21, shape[1]), dtype=np.float64)
        complete = np.ones_like(compounded, dtype=bool)
        for offset in range(2, 22):
            values = returns[offset : offset + shape[0] - 21].astype(np.float64)
            complete &= np.isfinite(values)
            compounded *= np.where(np.isfinite(values), 1.0 + values, 1.0)
        labels[: shape[0] - 21] = np.where(complete, compounded - 1.0, np.nan).astype(np.float32)
    allowed = np.zeros(shape[0], dtype=bool)
    for split in SPLITS.values():
        for index in range(max(0, shape[0] - 21)):
            allowed[index] |= split.start <= dates[index] <= split.end and dates[index + 21] <= split.end
    labels[~allowed] = np.nan
    return_group = zarr.open_group(str(panel_root / "returns.zarr"), mode="w")
    return_group.create_array("daily_total_return", data=returns, chunks=(252, min(256, shape[1])), overwrite=True)
    return_group.create_array("forward_return_20d", data=labels, chunks=(252, min(256, shape[1])), overwrite=True)
    member = np.zeros(shape, dtype=bool)
    valid_intervals = membership_intervals[membership_intervals["MbrFlg"].astype(str).str.upper().isin(ACTIVE_MEMBERSHIP_FLAGS)]
    permno_to_index = {value: index for index, value in enumerate(permnos)}
    for row in valid_intervals.itertuples(index=False):
        permno = int(row.PERMNO)
        if permno not in permno_to_index:
            continue
        active = (dates >= pd.Timestamp(row.MbrStartDt)) & (dates <= pd.Timestamp(row.MbrEndDt))
        member[active, permno_to_index[permno]] = True
    membership_group = zarr.open_group(str(panel_root / "membership.zarr"), mode="w")
    membership_group.create_array("membership", data=member, chunks=(252, min(256, shape[1])), overwrite=True)
    index = {"dates": [str(value.date()) for value in dates], "permnos": permnos.tolist(), "shape": list(shape), "feature_names": list(feature_columns), "axes": ["trading_date", "PERMNO"]}
    write_json(panel_root / "index.json", index)
    return {"n_dates": shape[0], "n_permnos": shape[1], "member_count_median": float(np.median(member.sum(axis=1))), "label_finite": int(np.isfinite(labels).sum())}


def build_panel(raw_root: str | Path, processed_root: str | Path) -> dict[str, Any]:
    """Build an immutable adjusted daily layer; risk arrays are a later explicit step."""
    validation = validate_raw_bundle(raw_root)
    if not validation["ok"]:
        raise RuntimeError(f"raw data validation failed: {validation['failures']}")
    files = discover_data_files(raw_root)
    source_fingerprint = files_fingerprint(files.values())
    panel_root = Path(processed_root) / "panel"
    manifest_path = panel_root / "build_manifest.yaml"
    destination = panel_root / "daily"
    dense_paths = [panel_root / name for name in ["features.zarr", "returns.zarr", "membership.zarr", "index.json"]]
    if manifest_path.exists() and destination.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            previous = yaml.safe_load(handle) or {}
        if previous.get("source_fingerprint") == source_fingerprint and previous.get("artifact_version") == PANEL_ARTIFACT_VERSION and all(path.exists() for path in dense_paths):
            return {**previous, "reused": True}
    if destination.exists():
        adjusted = pd.read_parquet(destination).sort_values(["DlyCalDt", "PERMNO"])
        adjustment_audit = previous.get("adjustment_audit", {}) if 'previous' in locals() else {}
        delisting_audit = previous.get("delisting_audit", {}) if 'previous' in locals() else {}
    else:
        daily = pd.read_parquet(files["daily"])
        delistings = pd.read_parquet(files["delistings"])
        adjusted, adjustment_audit = apply_crsp_adjustments(daily)
        adjusted, delisting_audit = fill_missing_delisting_returns(adjusted, delistings)
        adjusted = adjusted.sort_values(["DlyCalDt", "PERMNO"])
        destination.mkdir(parents=True, exist_ok=True)
        adjusted["year"] = pd.to_datetime(adjusted["DlyCalDt"]).dt.year
        adjusted.to_parquet(destination, index=False, partition_cols=["year"], compression="zstd")
    membership_intervals = pd.read_parquet(files["membership"])
    dense_audit = _write_dense_artifacts(adjusted, membership_intervals, panel_root)
    manifest = {"artifact_version": PANEL_ARTIFACT_VERSION, "source_fingerprint": source_fingerprint, "rows": len(adjusted), "adjustment_audit": adjustment_audit, "delisting_audit": delisting_audit, "dense_audit": dense_audit}
    manifest["build_fingerprint"] = stable_hash(manifest)
    write_json(panel_root / "qa_report.json", validation | manifest)
    write_yaml(manifest_path, manifest)
    return manifest
