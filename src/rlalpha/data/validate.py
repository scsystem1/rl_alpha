from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .discovery import discover_data_files
from .membership import membership_on_date
from ..utils.hashing import files_fingerprint, stable_hash


def validate_raw_bundle(root: str | Path) -> dict[str, Any]:
    files = discover_data_files(root, strict=True)
    daily_path = files["daily"]
    daily = pd.read_parquet(daily_path, columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyRet", "DlyVol", "DlyCumFacPr", "DlyCumFacShr", "DlyCap", "ShrOut"])
    daily["DlyCalDt"] = pd.to_datetime(daily["DlyCalDt"])
    duplicate_count = int(daily.duplicated(["PERMNO", "DlyCalDt"]).sum())
    valid_factor = pd.to_numeric(daily["DlyCumFacPr"], errors="coerce") > 0
    adjusted = pd.DataFrame(index=daily.index)
    for column in ["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"]:
        adjusted[column] = pd.to_numeric(daily[column], errors="coerce") / daily["DlyCumFacPr"].where(valid_factor)
    ohlc_mask = adjusted.notna().all(axis=1)
    high_ok = adjusted["DlyHigh"] >= adjusted[["DlyOpen", "DlyClose"]].max(axis=1)
    low_ok = adjusted["DlyLow"] <= adjusted[["DlyOpen", "DlyClose"]].min(axis=1)
    membership = pd.read_parquet(files["membership"])
    typical = {}
    for date in ("2010-06-30", "2018-06-29", "2021-06-30", "2025-06-30"):
        typical[date] = len(membership_on_date(membership, date))
    ratio = daily["DlyCap"] / (daily["DlyClose"].abs() * daily["ShrOut"])
    finite_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    metadata = {}
    for name, path in files.items():
        parquet = pq.ParquetFile(path)
        metadata[name] = {"path": str(path), "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names}
    failures = []
    if duplicate_count:
        failures.append("duplicate PERMNO-date rows")
    if any(not 450 <= count <= 550 for count in typical.values()):
        failures.append("atypical historical membership count")
    report = {
        "ok": not failures,
        "failures": failures,
        "files": metadata,
        "duplicate_permno_date": duplicate_count,
        "date_range": [str(daily["DlyCalDt"].min().date()), str(daily["DlyCalDt"].max().date())],
        "coverage": {column: float(pd.to_numeric(daily[column], errors="coerce").notna().mean()) for column in ["DlyClose", "DlyRet", "DlyVol"]},
        "adjusted_ohlc": {"rows_checked": int(ohlc_mask.sum()), "high_valid_rate": float(high_ok[ohlc_mask].mean()), "low_valid_rate": float(low_ok[ohlc_mask].mean())},
        "membership_counts": typical,
        "cap_identity_ratio_median": float(finite_ratio.median()),
        "source_fingerprint": files_fingerprint(files.values()),
    }
    report["report_hash"] = stable_hash(report)
    return report

