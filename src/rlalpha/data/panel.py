from __future__ import annotations

from pathlib import Path
import hashlib
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml
import zarr
from filelock import FileLock

from .adjustments import apply_crsp_adjustments, fill_missing_delisting_returns
from .discovery import discover_data_files
from .eligibility import trade_eligibility
from .membership import ACTIVE_MEMBERSHIP_FLAGS
from .splits import SPLITS
from .validate import validate_raw_bundle
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import write_json, write_yaml

PANEL_ARTIFACT_VERSION = 3
PANEL_SEMANTICS = {
    "return": "finite CIZ DlyRet; flagged missing delist row may use matched DelRet; unresolved remains NaN",
    "label": "compound t+2 through t+21 with split-contained exit",
    "membership": "inclusive historical interval",
    "trade_eligibility": {
        "SecurityType": ["EQTY"],
        "SecuritySubType": ["COM"],
        "ShareType": ["NS"],
        "PrimaryExch": ["A", "N", "Q"],
        "TradingStatusFlg": ["A"],
        "requirements": ["positive adjusted close", "positive adjusted volume", "finite current total return"],
    },
    "evaluator": "membership-aware-v3",
}


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


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
    eligible = np.zeros(shape, dtype=bool)
    eligible[date_codes, permno_codes] = trade_eligibility(adjusted)
    eligibility_group = zarr.open_group(str(panel_root / "eligibility.zarr"), mode="w")
    eligibility_group.create_array("trade_eligibility", data=eligible, chunks=(252, min(256, shape[1])), overwrite=True)
    universe_counts = pd.DataFrame({
        "date": dates,
        "members": member.sum(axis=1),
        "eligible_all_assets": eligible.sum(axis=1),
        "eligible_members": (member & eligible).sum(axis=1),
        "finite_forward_labels_for_eligible_members": (member & eligible & np.isfinite(labels)).sum(axis=1),
    })
    universe_counts.to_parquet(panel_root / "universe_counts.parquet", index=False)
    formal = universe_counts[universe_counts["date"] >= pd.Timestamp("2010-01-01")]
    abnormal_member_days = int(((formal["members"] < 400) | (formal["members"] > 600)).sum())
    low_eligibility_days = int((formal["eligible_members"] / formal["members"].clip(lower=1) < 0.90).sum())
    sp500_scale_gate_applied = len(permnos) >= 400
    if sp500_scale_gate_applied and (abnormal_member_days or low_eligibility_days):
        raise RuntimeError(f"universe hard gate failed: abnormal_member_days={abnormal_member_days}, low_eligibility_days={low_eligibility_days}")
    content_hashes = {f"feature_{name}": _array_sha256(values) for name, values in dense.items()}
    content_hashes.update({"forward_return_20d": _array_sha256(labels), "membership": _array_sha256(member), "trade_eligibility": _array_sha256(eligible)})
    panel_content_hash = stable_hash(content_hashes)
    index = {"dates": [str(value.date()) for value in dates], "permnos": permnos.tolist(), "shape": list(shape), "feature_names": list(feature_columns), "axes": ["trading_date", "PERMNO"], "panel_content_hash": panel_content_hash}
    write_json(panel_root / "index.json", index)
    return {"n_dates": shape[0], "n_permnos": shape[1], "member_count_median": float(np.median(member.sum(axis=1))), "eligible_member_count_median": float(np.median((member & eligible).sum(axis=1))), "label_finite": int(np.isfinite(labels).sum()), "universe_hard_gates": {"sp500_scale_gate_applied": sp500_scale_gate_applied, "abnormal_member_days": abnormal_member_days, "low_eligibility_days": low_eligibility_days, "member_range": [400, 600], "minimum_eligibility_ratio": 0.90}, "content_hashes": content_hashes, "panel_content_hash": panel_content_hash}


def build_panel(raw_root: str | Path, processed_root: str | Path) -> dict[str, Any]:
    """Build an immutable adjusted daily layer; risk arrays are a later explicit step."""
    validation = validate_raw_bundle(raw_root)
    if not validation["ok"]:
        raise RuntimeError(f"raw data validation failed: {validation['failures']}")
    files = discover_data_files(raw_root)
    source_files = {name: file_fingerprint(path) for name, path in sorted(files.items())}
    source_fingerprint = stable_hash({name: {"size": item["size"], "sha256": item["sha256"]} for name, item in source_files.items()})
    code_files = [
        Path(__file__),
        *(Path(__file__).with_name(name) for name in ("adjustments.py", "contracts.py", "discovery.py", "eligibility.py", "membership.py", "splits.py", "validate.py")),
        Path(__file__).parents[1] / "dsl/evaluator.py",
    ]
    code_fingerprint = stable_hash([{key: value for key, value in file_fingerprint(path).items() if key in {"size", "sha256"}} for path in code_files])
    build_identity = stable_hash({"artifact_version": PANEL_ARTIFACT_VERSION, "source_fingerprint": source_fingerprint, "semantics": PANEL_SEMANTICS, "code_fingerprint": code_fingerprint})
    processed_root = Path(processed_root)
    panel_root = processed_root / "panel"
    manifest_path = panel_root / "build_manifest.yaml"
    dense_paths = [panel_root / name for name in ["daily", "features.zarr", "returns.zarr", "membership.zarr", "eligibility.zarr", "index.json", "universe_counts.parquet"]]
    lock_root = Path(tempfile.gettempdir()) / "rlalpha-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_root / f"panel-{stable_hash(str(processed_root.resolve()))}.lock"))
    processed_root.mkdir(parents=True, exist_ok=True)
    # Synchronization lives in /tmp rather than beside the data/result files;
    # the build itself still commits from a unique directory with os.replace.
    with lock:
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                previous = yaml.safe_load(handle) or {}
            if previous.get("build_identity") == build_identity and all(path.exists() for path in dense_paths):
                return {**previous, "reused": True}
        temporary_root = Path(tempfile.mkdtemp(prefix=".panel-build-", dir=processed_root))
        temporary_panel = temporary_root / "panel"
        temporary_panel.mkdir(parents=True)
        try:
            daily = pd.read_parquet(files["daily"])
            delistings = pd.read_parquet(files["delistings"])
            adjusted, adjustment_audit = apply_crsp_adjustments(daily)
            adjusted, delisting_audit = fill_missing_delisting_returns(adjusted, delistings)
            adjusted = adjusted.sort_values(["DlyCalDt", "PERMNO"], kind="stable").reset_index(drop=True)
            destination = temporary_panel / "daily"
            adjusted_for_write = adjusted.copy()
            adjusted_for_write["year"] = pd.to_datetime(adjusted_for_write["DlyCalDt"]).dt.year
            adjusted_for_write.to_parquet(destination, index=False, partition_cols=["year"], compression="zstd")
            membership_intervals = pd.read_parquet(files["membership"])
            dense_audit = _write_dense_artifacts(adjusted, membership_intervals, temporary_panel)
            manifest = {
                "artifact_version": PANEL_ARTIFACT_VERSION,
                "build_identity": build_identity,
                "source_fingerprint": source_fingerprint,
                "source_files": source_files,
                "code_fingerprint": code_fingerprint,
                "semantics": PANEL_SEMANTICS,
                "rows": len(adjusted),
                "adjustment_audit": adjustment_audit,
                "delisting_audit": delisting_audit,
                "dense_audit": dense_audit,
            }
            manifest["build_fingerprint"] = stable_hash({key: value for key, value in manifest.items() if key != "source_files"})
            write_json(temporary_panel / "qa_report.json", validation | manifest)
            write_yaml(temporary_panel / "build_manifest.yaml", manifest)
            backup = None
            if panel_root.exists():
                prior_hash = str((previous if "previous" in locals() else {}).get("build_fingerprint", "unknown"))[:12]
                backup = processed_root / f"panel.legacy_unverified.{prior_hash}"
                if backup.exists():
                    raise RuntimeError(f"refusing to overwrite existing legacy panel backup: {backup}")
                os.replace(panel_root, backup)
            try:
                os.replace(temporary_panel, panel_root)
            except Exception:
                if backup is not None and backup.exists() and not panel_root.exists():
                    os.replace(backup, panel_root)
                raise
            return manifest
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
