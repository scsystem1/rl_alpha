from __future__ import annotations

import pandas as pd

ACTIVE_MEMBERSHIP_FLAGS = frozenset({"Y", "NORM", "1", "TRUE"})


def membership_on_date(membership: pd.DataFrame, date: str | pd.Timestamp) -> set[int]:
    date = pd.Timestamp(date)
    start = pd.to_datetime(membership["MbrStartDt"])
    end = pd.to_datetime(membership["MbrEndDt"])
    valid_flag = membership["MbrFlg"].astype(str).str.upper().isin(ACTIVE_MEMBERSHIP_FLAGS)
    active = (start <= date) & (date <= end) & valid_flag
    return set(pd.to_numeric(membership.loc[active, "PERMNO"]).astype(int))


def attach_membership(daily: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    rows = daily[["PERMNO", "DlyCalDt"]].reset_index(names="_index")
    intervals = membership[["PERMNO", "MbrStartDt", "MbrEndDt", "MbrFlg"]].copy()
    intervals = intervals[intervals["MbrFlg"].astype(str).str.upper().isin(ACTIVE_MEMBERSHIP_FLAGS)]
    merged = rows.merge(intervals, on="PERMNO", how="left")
    inside = (merged["MbrStartDt"] <= merged["DlyCalDt"]) & (merged["DlyCalDt"] <= merged["MbrEndDt"])
    active_indices = set(merged.loc[inside, "_index"].tolist())
    return pd.Series(daily.index.isin(active_indices), index=daily.index, name="is_member")
