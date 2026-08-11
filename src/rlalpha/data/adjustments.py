from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .splits import SPLITS


def apply_crsp_adjustments(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol", "DlyCumFacPr", "DlyCumFacShr"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"missing adjustment fields: {sorted(missing)}")
    out = daily.copy()
    price_factor = pd.to_numeric(out["DlyCumFacPr"], errors="coerce")
    share_factor = pd.to_numeric(out["DlyCumFacShr"], errors="coerce")
    valid_price = np.isfinite(price_factor) & (price_factor > 0)
    valid_share = np.isfinite(share_factor) & (share_factor > 0)
    for raw, adjusted in (("DlyOpen", "adj_open"), ("DlyHigh", "adj_high"), ("DlyLow", "adj_low"), ("DlyClose", "adj_close")):
        values = pd.to_numeric(out[raw], errors="coerce")
        out[adjusted] = np.where(valid_price, values / price_factor, np.nan)
    volume = pd.to_numeric(out["DlyVol"], errors="coerce")
    out["adj_volume"] = np.where(valid_share, volume * share_factor, np.nan)
    return out, {"invalid_price_factor": int((~valid_price).sum()), "invalid_share_factor": int((~valid_share).sum())}


def fill_missing_delisting_returns(daily: pd.DataFrame, delistings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill only flagged, missing CIZ returns; never compound a finite DlyRet."""
    out = daily.copy()
    flagged = out["DlyDelFlg"].astype(str).str.upper().eq("Y")
    missing = pd.to_numeric(out["DlyRet"], errors="coerce").isna()
    eligible = flagged & missing
    lookup = delistings[["PERMNO", "DelDlyDt", "DelRet"]].copy()
    lookup["DelDlyDt"] = pd.to_datetime(lookup["DelDlyDt"])
    if lookup.duplicated(["PERMNO", "DelDlyDt"]).any():
        raise ValueError("duplicate PERMNO-DelDlyDt keys in delisting table")
    keys = out.loc[eligible, ["PERMNO", "DlyCalDt"]].copy()
    keys["_row"] = keys.index
    merged = keys.merge(lookup, left_on=["PERMNO", "DlyCalDt"], right_on=["PERMNO", "DelDlyDt"], how="left")
    delisting_return = pd.to_numeric(merged["DelRet"], errors="coerce")
    finite = pd.Series(np.isfinite(delisting_return), index=merged.index)
    if finite.any():
        out.loc[merged.loc[finite, "_row"], "DlyRet"] = merged.loc[finite, "DelRet"].to_numpy()
    split_counts = {}
    eligible_dates = pd.to_datetime(out.loc[eligible, "DlyCalDt"])
    filled_dates = pd.to_datetime(out.loc[merged.loc[finite, "_row"], "DlyCalDt"])
    for name, split in SPLITS.items():
        eligible_count = int(eligible_dates.between(split.start, split.end).sum())
        filled_count = int(filled_dates.between(split.start, split.end).sum())
        split_counts[name] = {"eligible_missing": eligible_count, "filled": filled_count, "unresolved": eligible_count - filled_count}
    before = pd.to_numeric(daily["DlyRet"], errors="coerce")
    after = pd.to_numeric(out["DlyRet"], errors="coerce")
    finite_after = after[np.isfinite(after)]
    quantiles = finite_after.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).to_dict()
    return out, {
        "eligible_missing": int(eligible.sum()),
        "filled": int(finite.sum()),
        "unresolved": int(eligible.sum() - finite.sum()),
        "per_split": split_counts,
        "raw_finite_coverage": float(np.isfinite(before).mean()),
        "final_finite_coverage": float(np.isfinite(after).mean()),
        "final_return_quantiles": {str(key): float(value) for key, value in quantiles.items()},
        "no_fill_sensitivity_changed_rows": int(finite.sum()),
    }
