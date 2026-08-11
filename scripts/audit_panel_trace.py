from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import zarr

from rlalpha.data.discovery import discover_data_files
from rlalpha.data.membership import ACTIVE_MEMBERSHIP_FLAGS
from rlalpha.utils.hashing import file_fingerprint, stable_hash
from rlalpha.utils.io import write_json


DEFAULT_DATES = ("2010-06-30", "2018-06-29", "2021-06-30", "2025-06-30")


def build_trace(raw_root: str | Path, processed_root: str | Path, output_path: str | Path) -> dict[str, object]:
    raw_root, panel_root, output_path = Path(raw_root), Path(processed_root) / "panel", Path(output_path)
    index = json.loads((panel_root / "index.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(index["dates"])
    permnos = np.asarray(index["permnos"], dtype=np.int64)
    date_to_index = {date: position for position, date in enumerate(dates)}
    member = zarr.open_array(str(panel_root / "membership.zarr/membership"), mode="r")
    eligible = zarr.open_array(str(panel_root / "eligibility.zarr/trade_eligibility"), mode="r")
    label = zarr.open_array(str(panel_root / "returns.zarr/forward_return_20d"), mode="r")
    daily_return = zarr.open_array(str(panel_root / "returns.zarr/daily_total_return"), mode="r")
    features = {name: zarr.open_array(str(panel_root / f"features.zarr/{name}"), mode="r") for name in ("open", "high", "low", "close", "volume", "return")}
    risk_group = zarr.open_group(str(panel_root / "risk_exposures.zarr"), mode="r")
    risk = risk_group["exposures"]
    exposure_names = list(risk_group.attrs["columns"])

    chosen: list[tuple[pd.Timestamp, int, int]] = []
    for raw_date in DEFAULT_DATES:
        date = pd.Timestamp(raw_date)
        if date not in date_to_index:
            continue
        day = date_to_index[date]
        asset_indices = np.flatnonzero(np.asarray(member[day], dtype=bool) & np.asarray(eligible[day], dtype=bool))[:3]
        chosen.extend((date, day, int(asset)) for asset in asset_indices)
    if not chosen:
        raise RuntimeError("no deterministic trace samples are available")

    selected_dates = {date for date, _, _ in chosen}
    selected_permnos = {int(permnos[asset]) for _, _, asset in chosen}
    daily_columns = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyRet", "DlyVol", "DlyCumFacPr", "DlyCumFacShr", "DlyCap", "SICCD", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
    daily = pd.read_parquet(panel_root / "daily", columns=daily_columns)
    daily["DlyCalDt"] = pd.to_datetime(daily["DlyCalDt"])
    daily = daily[daily["DlyCalDt"].isin(selected_dates) & pd.to_numeric(daily["PERMNO"]).astype(int).isin(selected_permnos)]
    daily_lookup = {(pd.Timestamp(row.DlyCalDt), int(row.PERMNO)): row for row in daily.itertuples(index=False)}

    files = discover_data_files(raw_root)
    membership = pd.read_parquet(files["membership"])
    membership["MbrStartDt"] = pd.to_datetime(membership["MbrStartDt"])
    membership["MbrEndDt"] = pd.to_datetime(membership["MbrEndDt"])
    membership = membership[membership["MbrFlg"].astype(str).str.upper().isin(ACTIVE_MEMBERSHIP_FLAGS)]
    lineage = pd.read_parquet(panel_root / "fundamental_lineage.parquet")
    for column in ("datadate", "available_date", "linkdt", "linkenddt"):
        lineage[column] = pd.to_datetime(lineage[column])

    rows = []
    label_differences = []
    for date, day, asset in chosen:
        permno = int(permnos[asset])
        source = daily_lookup[(date, permno)]
        interval = membership[(pd.to_numeric(membership["PERMNO"]).astype(int) == permno) & (membership["MbrStartDt"] <= date) & (date <= membership["MbrEndDt"])]
        if len(interval) != 1:
            raise RuntimeError(f"expected one active membership interval for {permno} on {date.date()}, found {len(interval)}")
        interval_row = interval.iloc[0]
        candidates = lineage[(pd.to_numeric(lineage["PERMNO"]).astype(int) == permno) & (lineage["available_date"] <= date)]
        candidates = candidates[(candidates["linkdt"] <= date) & (candidates["linkenddt"].isna() | (date <= candidates["linkenddt"]))]
        candidates = candidates[date <= candidates["available_date"] + pd.DateOffset(months=18)]
        fundamental = candidates.sort_values(["available_date", "gvkey"], kind="stable").iloc[-1] if len(candidates) else None
        horizon = np.asarray(daily_return[day + 2 : day + 22, asset], dtype=float)
        manual_label = float(np.prod(1.0 + horizon) - 1.0) if len(horizon) == 20 and np.isfinite(horizon).all() else float("nan")
        stored_label = float(label[day, asset])
        if np.isfinite(manual_label) and np.isfinite(stored_label):
            label_differences.append(abs(manual_label - stored_label))
        record = {
            "date": date,
            "PERMNO": permno,
            "membership_start": interval_row["MbrStartDt"],
            "membership_end": interval_row["MbrEndDt"],
            "membership_flag": interval_row["MbrFlg"],
            "member": bool(member[day, asset]),
            "trade_eligible": bool(eligible[day, asset]),
            "raw_open": source.DlyOpen,
            "raw_high": source.DlyHigh,
            "raw_low": source.DlyLow,
            "raw_close": source.DlyClose,
            "raw_return": source.DlyRet,
            "raw_volume": source.DlyVol,
            "price_adjustment_factor": source.DlyCumFacPr,
            "share_adjustment_factor": source.DlyCumFacShr,
            "dense_open": float(features["open"][day, asset]),
            "dense_high": float(features["high"][day, asset]),
            "dense_low": float(features["low"][day, asset]),
            "dense_close": float(features["close"][day, asset]),
            "dense_volume": float(features["volume"][day, asset]),
            "dense_return": float(features["return"][day, asset]),
            "stored_forward_return_20d": stored_label,
            "manual_forward_return_t_plus_2_to_t_plus_21": manual_label,
            "fundamental_gvkey": None if fundamental is None else fundamental["gvkey"],
            "fundamental_datadate": pd.NaT if fundamental is None else fundamental["datadate"],
            "fundamental_available_date": pd.NaT if fundamental is None else fundamental["available_date"],
            "ccm_link_start": pd.NaT if fundamental is None else fundamental["linkdt"],
            "ccm_link_end": pd.NaT if fundamental is None else fundamental["linkenddt"],
            "ccm_link_type": None if fundamental is None else fundamental["linktype"],
            "ccm_link_primacy": None if fundamental is None else fundamental["linkprim"],
        }
        for exposure_index, name in enumerate(exposure_names):
            record[f"exposure_{name}"] = float(risk[day, asset, exposure_index])
        if not np.isclose(record["dense_close"], float(source.adj_close), equal_nan=True):
            raise RuntimeError("dense adjusted close does not match the audited daily layer")
        if not np.isclose(record["dense_volume"], float(source.adj_volume), equal_nan=True):
            raise RuntimeError("dense adjusted volume does not match the audited daily layer")
        rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["date", "PERMNO"])
    frame.to_parquet(output_path, index=False)
    panel_manifest = yaml.safe_load((panel_root / "build_manifest.yaml").read_text(encoding="utf-8")) or {}
    risk_manifest = yaml.safe_load((panel_root / "risk_build_manifest.yaml").read_text(encoding="utf-8")) or {}
    maximum_label_difference = max(label_differences, default=0.0)
    if maximum_label_difference > 1e-6:
        raise RuntimeError(f"stored/manual forward labels differ by {maximum_label_difference}")
    metadata = {
        "schema_version": 1,
        "sample_policy": {"dates": list(DEFAULT_DATES), "assets_per_date": 3, "asset_order": "lowest eligible member PERMNO"},
        "rows": len(frame),
        "panel_build_fingerprint": panel_manifest.get("build_fingerprint"),
        "risk_build_fingerprint": risk_manifest.get("build_fingerprint"),
        "maximum_manual_label_absolute_difference": maximum_label_difference,
        "trace_fingerprint": stable_hash(frame.astype(str).to_dict(orient="records")),
        "artifact": file_fingerprint(output_path),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/data/sunyuxiang/rl_alpha")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_trace(arguments.raw_root, arguments.processed_root, arguments.output), indent=2))
