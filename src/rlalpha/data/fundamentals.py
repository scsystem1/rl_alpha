from __future__ import annotations

import numpy as np
import pandas as pd


def filter_standard_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["indfmt"].eq("INDL")
        & frame["datafmt"].eq("STD")
        & frame["popsrc"].eq("D")
        & frame["consol"].eq("C")
        & frame["curcd"].eq("USD")
    )
    out = frame.loc[mask].copy()
    out["datadate"] = pd.to_datetime(out["datadate"])
    out = out.drop_duplicates()
    if out.duplicated(["gvkey", "datadate"]).any():
        raise ValueError("conflicting duplicate gvkey-datadate fundamentals")
    out["available_date"] = out["datadate"] + pd.DateOffset(months=6)
    return out


def select_ccm_links(ccm: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    links = ccm.copy()
    links = links[
        links["USEDFLAG"].eq(1)
        & links["linktype"].isin(["LC", "LU", "LS"])
        & links["linkprim"].isin(["P", "C"])
    ]
    links["linkdt"] = pd.to_datetime(links["linkdt"])
    links["linkenddt"] = pd.to_datetime(links["linkenddt"])
    source = fundamentals.copy()
    source["datadate"] = pd.to_datetime(source["datadate"])
    merged = source.merge(links, on="gvkey", how="inner")
    valid = (merged["linkdt"] <= merged["datadate"]) & (merged["linkenddt"].isna() | (merged["datadate"] <= merged["linkenddt"]))
    merged = merged.loc[valid].copy()
    merged["_prim"] = merged["linkprim"].map({"P": 0, "C": 1})
    merged["_type"] = merged["linktype"].map({"LC": 0, "LU": 1, "LS": 2})
    merged["_gvkey_tie"] = merged["gvkey"].astype(str)
    merged = merged.sort_values(["lpermno", "datadate", "_prim", "linkdt", "_type", "_gvkey_tie"], ascending=[True, True, True, False, True, True], kind="stable")
    merged = merged.drop_duplicates(["lpermno", "datadate"], keep="first")
    return merged.drop(columns=["_prim", "_type", "_gvkey_tie"]).rename(columns={"lpermno": "PERMNO"})


def compute_accounting_exposures(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = frame.copy()
    preferred = out[["pstkrv", "pstkl", "pstk"]].bfill(axis=1).iloc[:, 0].fillna(0.0)
    shareholders = out["seq"].where(out["seq"].notna(), out["ceq"] + out["pstk"].fillna(0.0))
    out["book_equity"] = shareholders + out["txditc"].fillna(0.0) - preferred
    positive_be = out["book_equity"] > 0
    out["operating_profitability"] = np.where(
        positive_be & out["revt"].notna() & out["cogs"].notna(),
        (out["revt"] - out["cogs"] - out["xsga"].fillna(0.0) - out["xint"].fillna(0.0)) / out["book_equity"],
        np.nan,
    )
    out = out.sort_values(["gvkey", "datadate"])
    previous_assets = out.groupby("gvkey", sort=False)["at"].shift()
    out["investment"] = np.where((out["at"] > 0) & (previous_assets > 0), out["at"] / previous_assets - 1.0, np.nan)
    out["leverage"] = (out["dltt"].fillna(0.0) + out["dlc"].fillna(0.0)) / out["at"].where(out["at"] > 0)
    audit = {"xsga_zero_fill_rate": float(out["xsga"].isna().mean()), "xint_zero_fill_rate": float(out["xint"].isna().mean()), "nonpositive_book_equity_rate": float((~positive_be).mean())}
    return out, audit


def asof_fundamentals(stock_days: pd.DataFrame, fundamentals: pd.DataFrame, max_age_months: int = 18) -> pd.DataFrame:
    """Backward as-of join on available_date, preventing future accounting use."""
    left = stock_days.copy()
    left["DlyCalDt"] = pd.to_datetime(left["DlyCalDt"])
    right = fundamentals.copy()
    right["available_date"] = pd.to_datetime(right["available_date"])
    pieces = []
    for permno, days in left.groupby("PERMNO", sort=False):
        records = right[right["PERMNO"].eq(permno)].sort_values("available_date")
        days = days.sort_values("DlyCalDt")
        if records.empty:
            joined = days.copy()
            joined["available_date"] = pd.NaT
        else:
            joined = pd.merge_asof(days, records, left_on="DlyCalDt", right_on="available_date", direction="backward", suffixes=("", "_fund"))
        pieces.append(joined)
    result = pd.concat(pieces, ignore_index=True) if pieces else left
    age_limit = result["available_date"] + pd.DateOffset(months=max_age_months)
    stale = result["available_date"].notna() & (result["DlyCalDt"] > age_limit)
    fund_columns = [column for column in result.columns if column not in left.columns]
    result.loc[stale, fund_columns] = np.nan
    return result
