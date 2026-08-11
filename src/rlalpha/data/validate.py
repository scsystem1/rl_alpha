from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .discovery import discover_data_files
from .membership import membership_on_date
from ..utils.hashing import files_fingerprint, stable_hash


def _invalid_integral_identifiers(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce")
    return int((~np.isfinite(numeric) | (numeric % 1 != 0)).sum())


def validate_raw_bundle(root: str | Path) -> dict[str, Any]:
    files = discover_data_files(root, strict=True)
    daily_path = files["daily"]
    daily_columns = ["PERMNO", "PERMCO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyRet", "DlyVol", "DlyCumFacPr", "DlyCumFacShr", "DlyCap", "ShrOut", "SecurityType", "SecuritySubType", "ShareType", "PrimaryExch", "TradingStatusFlg"]
    daily = pd.read_parquet(daily_path, columns=daily_columns)
    parsed_dates = pd.to_datetime(daily["DlyCalDt"], errors="coerce")
    invalid_daily_dates = int(parsed_dates.isna().sum())
    daily["DlyCalDt"] = parsed_dates
    duplicate_count = int(daily.duplicated(["PERMNO", "DlyCalDt"]).sum())
    permno = pd.to_numeric(daily["PERMNO"], errors="coerce")
    permco = pd.to_numeric(daily["PERMCO"], errors="coerce")
    invalid_permno = _invalid_integral_identifiers(permno)
    invalid_permco = _invalid_integral_identifiers(permco)
    price_factor = pd.to_numeric(daily["DlyCumFacPr"], errors="coerce")
    share_factor = pd.to_numeric(daily["DlyCumFacShr"], errors="coerce")
    valid_factor = np.isfinite(price_factor) & (price_factor > 0)
    valid_share_factor = np.isfinite(share_factor) & (share_factor > 0)
    adjusted = pd.DataFrame(index=daily.index)
    for column in ["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"]:
        adjusted[column] = pd.to_numeric(daily[column], errors="coerce") / daily["DlyCumFacPr"].where(valid_factor)
    ohlc_mask = adjusted.notna().all(axis=1)
    high_ok = adjusted["DlyHigh"] >= adjusted[["DlyOpen", "DlyClose"]].max(axis=1)
    low_ok = adjusted["DlyLow"] <= adjusted[["DlyOpen", "DlyClose"]].min(axis=1)
    membership = pd.read_parquet(files["membership"])
    membership["MbrStartDt"] = pd.to_datetime(membership["MbrStartDt"], errors="coerce")
    membership["MbrEndDt"] = pd.to_datetime(membership["MbrEndDt"], errors="coerce")
    invalid_membership_intervals = int((membership["MbrStartDt"].isna() | membership["MbrEndDt"].isna() | (membership["MbrStartDt"] > membership["MbrEndDt"])).sum())
    active_membership = membership[membership["MbrFlg"].astype(str).str.upper().isin({"Y", "NORM", "1", "TRUE"})].sort_values(["PERMNO", "MbrStartDt", "MbrEndDt"])
    previous_end = active_membership.groupby("PERMNO", sort=False)["MbrEndDt"].shift()
    overlapping_membership_intervals = int((previous_end.notna() & (active_membership["MbrStartDt"] <= previous_end)).sum())
    invalid_membership_permno = _invalid_integral_identifiers(membership["PERMNO"])
    typical = {}
    for date in ("2010-06-30", "2018-06-29", "2021-06-30", "2025-06-30"):
        typical[date] = len(membership_on_date(membership, date))
    ratio = daily["DlyCap"] / (daily["DlyClose"].abs() * daily["ShrOut"])
    finite_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    metadata = {}
    for name, path in files.items():
        parquet = pq.ParquetFile(path)
        metadata[name] = {"path": str(path), "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names}
    market = pd.read_parquet(files["market"], columns=["DlyCalDt", "vwretd"])
    market_dates = pd.to_datetime(market["DlyCalDt"], errors="coerce")
    duplicate_market_dates = int(market_dates.duplicated().sum())
    invalid_market_dates = int(market_dates.isna().sum())
    delistings = pd.read_parquet(files["delistings"], columns=["PERMNO", "DelDlyDt", "DelRet"])
    delisting_dates = pd.to_datetime(delistings["DelDlyDt"], errors="coerce")
    duplicate_delisting_keys = int(pd.DataFrame({"PERMNO": delistings["PERMNO"], "date": delisting_dates}).duplicated().sum())
    invalid_delisting_dates = int(delisting_dates.isna().sum())
    invalid_delisting_permno = _invalid_integral_identifiers(delistings["PERMNO"])
    ccm = pd.read_parquet(files["ccm"], columns=["gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"])
    fundamentals = pd.read_parquet(files["fundamentals"], columns=["gvkey", "datadate"])
    invalid_ccm_permno = _invalid_integral_identifiers(ccm["lpermno"])
    invalid_ccm_gvkey = int((~ccm["gvkey"].astype(str).str.fullmatch(r"\d+")).sum())
    invalid_fundamental_gvkey = int((~fundamentals["gvkey"].astype(str).str.fullmatch(r"\d+")).sum())
    duplicate_ccm_keys = int(ccm.duplicated(["gvkey", "lpermno", "linkdt", "linktype", "linkprim"]).sum())
    duplicate_fundamental_keys = int(fundamentals.duplicated(["gvkey", "datadate"]).sum())
    daily_unsorted_security_histories = int(daily.groupby("PERMNO", sort=False)["DlyCalDt"].apply(lambda values: not values.is_monotonic_increasing).sum())
    finite_return = pd.to_numeric(daily["DlyRet"], errors="coerce")
    invalid_return_below_minus_one = int((finite_return < -1.0).sum())
    nonfinite_nonnull_return = int((daily["DlyRet"].notna() & ~np.isfinite(finite_return)).sum())
    close = pd.to_numeric(daily["DlyClose"], errors="coerce")
    volume = pd.to_numeric(daily["DlyVol"], errors="coerce")
    invalid_nonpositive_close = int((np.isfinite(close) & (close <= 0)).sum())
    invalid_negative_volume = int((np.isfinite(volume) & (volume < 0)).sum())
    nonfinite_nonnull_close = int((daily["DlyClose"].notna() & ~np.isfinite(close)).sum())
    nonfinite_nonnull_volume = int((daily["DlyVol"].notna() & ~np.isfinite(volume)).sum())
    high_violations = int((ohlc_mask & ~high_ok).sum())
    low_violations = int((ohlc_mask & ~low_ok).sum())
    ohlc_violation_rate = (high_violations + low_violations) / max(1, int(ohlc_mask.sum()) * 2)
    schema_type_issues = []
    for name, path in files.items():
        schema = pq.ParquetFile(path).schema_arrow
        for column in ({"daily": ["DlyCalDt"], "membership": ["MbrStartDt", "MbrEndDt"], "market": ["DlyCalDt"], "ccm": ["linkdt", "linkenddt"], "fundamentals": ["datadate"], "delistings": ["DelDlyDt"]}[name]):
            if not pa.types.is_timestamp(schema.field(column).type):
                schema_type_issues.append(f"{name}.{column} is not a timestamp")
    failures = []
    if duplicate_count:
        failures.append("duplicate PERMNO-date rows")
    if invalid_daily_dates or invalid_permno or invalid_permco or invalid_membership_permno or invalid_delisting_permno or invalid_ccm_permno or invalid_ccm_gvkey or invalid_fundamental_gvkey:
        failures.append("invalid daily identifiers or dates")
    if duplicate_ccm_keys or duplicate_fundamental_keys:
        failures.append("duplicate CCM or fundamental source keys")
    if daily_unsorted_security_histories:
        failures.append("daily security histories are not sorted by date")
    if schema_type_issues:
        failures.append("date fields have invalid physical dtypes")
    if invalid_membership_intervals or overlapping_membership_intervals:
        failures.append("invalid or overlapping membership intervals")
    if duplicate_market_dates or invalid_market_dates:
        failures.append("invalid or duplicate market dates")
    if duplicate_delisting_keys or invalid_delisting_dates:
        failures.append("invalid or duplicate delisting keys")
    if invalid_return_below_minus_one or nonfinite_nonnull_return:
        failures.append("illegal daily total return")
    if invalid_nonpositive_close or invalid_negative_volume or nonfinite_nonnull_close or nonfinite_nonnull_volume:
        failures.append("illegal finite close or volume")
    if ohlc_violation_rate > 1e-5:
        failures.append("adjusted OHLC relationship violation rate above 1e-5")
    if float((~valid_factor).mean()) > 0.001 or float((~valid_share_factor).mean()) > 0.001:
        failures.append("invalid adjustment-factor rate above 0.1 percent")
    if not 0.9 <= float(finite_ratio.median()) <= 1.1:
        failures.append("CRSP market-cap unit identity outside tolerance")
    if any(float(pd.to_numeric(daily[column], errors="coerce").notna().mean()) < 0.80 for column in ["DlyClose", "DlyRet", "DlyVol"]):
        failures.append("required daily field coverage below 80 percent")
    if any(not 450 <= count <= 550 for count in typical.values()):
        failures.append("atypical historical membership count")
    report = {
        "ok": not failures,
        "failures": failures,
        "files": metadata,
        "duplicate_permno_date": duplicate_count,
        "invalid_identifiers_and_dates": {"daily_dates": invalid_daily_dates, "permno": invalid_permno, "permco": invalid_permco, "membership_permno": invalid_membership_permno, "delisting_permno": invalid_delisting_permno, "ccm_lpermno": invalid_ccm_permno, "ccm_gvkey": invalid_ccm_gvkey, "fundamental_gvkey": invalid_fundamental_gvkey},
        "duplicate_source_keys": {"ccm": duplicate_ccm_keys, "fundamentals": duplicate_fundamental_keys},
        "ordering": {"daily_unsorted_security_histories": daily_unsorted_security_histories},
        "schema_type_issues": schema_type_issues,
        "membership_intervals": {"invalid": invalid_membership_intervals, "overlapping": overlapping_membership_intervals},
        "market_keys": {"duplicate_dates": duplicate_market_dates, "invalid_dates": invalid_market_dates},
        "delisting_keys": {"duplicate": duplicate_delisting_keys, "invalid_dates": invalid_delisting_dates},
        "illegal_values": {"return_below_minus_one": invalid_return_below_minus_one, "nonfinite_nonnull_return": nonfinite_nonnull_return, "nonpositive_finite_close": invalid_nonpositive_close, "nonfinite_nonnull_close": nonfinite_nonnull_close, "negative_finite_volume": invalid_negative_volume, "nonfinite_nonnull_volume": nonfinite_nonnull_volume},
        "invalid_adjustment_factors": {"price": int((~valid_factor).sum()), "share": int((~valid_share_factor).sum())},
        "date_range": [str(daily["DlyCalDt"].min().date()), str(daily["DlyCalDt"].max().date())],
        "coverage": {column: float(pd.to_numeric(daily[column], errors="coerce").notna().mean()) for column in ["DlyClose", "DlyRet", "DlyVol"]},
        "adjusted_ohlc": {"rows_checked": int(ohlc_mask.sum()), "high_valid_rate": float(high_ok[ohlc_mask].mean()), "low_valid_rate": float(low_ok[ohlc_mask].mean()), "high_violations": high_violations, "low_violations": low_violations, "violation_rate": float(ohlc_violation_rate), "hard_threshold": 1e-5},
        "membership_counts": typical,
        "cap_identity_ratio_median": float(finite_ratio.median()),
        "source_fingerprint": files_fingerprint(files.values()),
    }
    report["report_hash"] = stable_hash(report)
    return report
