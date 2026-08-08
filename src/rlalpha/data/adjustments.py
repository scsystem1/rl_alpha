from __future__ import annotations

import numpy as np
import pandas as pd


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


def fill_missing_delisting_returns(daily: pd.DataFrame, delistings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill only flagged, missing CIZ returns; never compound a finite DlyRet."""
    out = daily.copy()
    flagged = out["DlyDelFlg"].astype(str).str.upper().eq("Y")
    missing = pd.to_numeric(out["DlyRet"], errors="coerce").isna()
    eligible = flagged & missing
    lookup = delistings[["PERMNO", "DelDlyDt", "DelRet"]].copy()
    lookup["DelDlyDt"] = pd.to_datetime(lookup["DelDlyDt"])
    lookup = lookup.drop_duplicates(["PERMNO", "DelDlyDt"], keep="last")
    keys = out.loc[eligible, ["PERMNO", "DlyCalDt"]].copy()
    keys["_row"] = keys.index
    merged = keys.merge(lookup, left_on=["PERMNO", "DlyCalDt"], right_on=["PERMNO", "DelDlyDt"], how="left")
    finite = pd.to_numeric(merged["DelRet"], errors="coerce").notna()
    if finite.any():
        out.loc[merged.loc[finite, "_row"], "DlyRet"] = merged.loc[finite, "DelRet"].to_numpy()
    return out, {"eligible_missing": int(eligible.sum()), "filled": int(finite.sum()), "unresolved": int(eligible.sum() - finite.sum())}

