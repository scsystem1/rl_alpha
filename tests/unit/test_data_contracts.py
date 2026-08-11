from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from rlalpha.data.adjustments import apply_crsp_adjustments, fill_missing_delisting_returns
from rlalpha.data.fundamentals import asof_fundamentals, compute_accounting_exposures, filter_standard_fundamentals, select_ccm_links
from rlalpha.data.labels import next_close_forward_return
from rlalpha.data.membership import membership_on_date
from rlalpha.data.eligibility import trade_eligibility
from rlalpha.data.splits import DateSplit
from rlalpha.data.panel import build_panel


def test_crsp_adjustment_and_invalid_factors():
    raw = pd.DataFrame({"DlyOpen": [10, 10], "DlyHigh": [12, 12], "DlyLow": [8, 8], "DlyClose": [11, 11], "DlyVol": [100, 100], "DlyCumFacPr": [2, 0], "DlyCumFacShr": [3, -1]})
    adjusted, audit = apply_crsp_adjustments(raw)
    assert adjusted.loc[0, "adj_close"] == 5.5
    assert adjusted.loc[0, "adj_volume"] == 300
    assert np.isnan(adjusted.loc[1, "adj_close"])
    assert audit == {"invalid_price_factor": 1, "invalid_share_factor": 1}


def test_membership_boundaries_are_inclusive():
    membership = pd.DataFrame({"PERMNO": [1, 2], "MbrStartDt": pd.to_datetime(["2020-01-02", "2020-01-02"]), "MbrEndDt": pd.to_datetime(["2020-01-06", "2020-01-06"]), "MbrFlg": ["Y", "NORM"]})
    assert membership_on_date(membership, "2020-01-02") == {1, 2}
    assert membership_on_date(membership, "2020-01-06") == {1, 2}
    assert membership_on_date(membership, "2020-01-07") == set()


def test_next_close_twenty_day_label_and_split_boundary():
    dates = pd.bdate_range("2020-01-01", periods=25)
    daily = pd.DataFrame({"PERMNO": 1, "DlyCalDt": dates, "DlyRet": 0.01})
    result = next_close_forward_return(daily)
    assert np.isclose(result.loc[0, "forward_return_20d"], 1.01**20 - 1)
    assert result.loc[0, "exit_date"] == dates[21]
    split = DateSplit("x", dates[0], dates[20])
    bounded = next_close_forward_return(daily, split=split)
    assert np.isnan(bounded.loc[0, "forward_return_20d"])


def test_delisting_return_is_never_multiplied_twice():
    daily = pd.DataFrame({"PERMNO": [1, 2], "DlyCalDt": pd.to_datetime(["2020-01-02"] * 2), "DlyDelFlg": ["Y", "Y"], "DlyRet": [-0.4, np.nan]})
    delist = pd.DataFrame({"PERMNO": [1, 2], "DelDlyDt": pd.to_datetime(["2020-01-02"] * 2), "DelRet": [-0.5, -0.6]})
    result, audit = fill_missing_delisting_returns(daily, delist)
    assert result.loc[0, "DlyRet"] == -0.4
    assert result.loc[1, "DlyRet"] == -0.6
    assert audit["filled"] == 1


def test_duplicate_delisting_key_is_a_hard_failure():
    daily = pd.DataFrame({"PERMNO": [1], "DlyCalDt": pd.to_datetime(["2020-01-02"]), "DlyDelFlg": ["Y"], "DlyRet": [np.nan]})
    delist = pd.DataFrame({"PERMNO": [1, 1], "DelDlyDt": pd.to_datetime(["2020-01-02"] * 2), "DelRet": [-0.5, -0.6]})
    with __import__("pytest").raises(ValueError, match="duplicate"):
        fill_missing_delisting_returns(daily, delist)


def test_ccm_primary_link_priority_and_latest_start():
    ccm = pd.DataFrame({"gvkey": ["1", "1", "1"], "linkprim": ["C", "P", "P"], "linktype": ["LC"] * 3, "lpermno": [10] * 3, "USEDFLAG": [1] * 3, "linkdt": pd.to_datetime(["2000-01-01", "1999-01-01", "2001-01-01"]), "linkenddt": [pd.NaT] * 3})
    fundamentals = pd.DataFrame({"gvkey": ["1"], "datadate": pd.to_datetime(["2020-12-31"])})
    result = select_ccm_links(ccm, fundamentals)
    assert len(result) == 1
    assert result.iloc[0]["linkprim"] == "P"
    assert result.iloc[0]["linkdt"] == pd.Timestamp("2001-01-01")


def test_compustat_six_month_lag_asof():
    raw = pd.DataFrame({"gvkey": ["1"], "datadate": pd.to_datetime(["2019-12-31"]), "indfmt": ["INDL"], "datafmt": ["STD"], "popsrc": ["D"], "consol": ["C"], "curcd": ["USD"], "value": [7]})
    filtered = filter_standard_fundamentals(raw)
    filtered["PERMNO"] = 1
    days = pd.DataFrame({"PERMNO": [1, 1], "DlyCalDt": pd.to_datetime(["2020-06-29", "2020-06-30"])})
    result = asof_fundamentals(days, filtered)
    assert pd.isna(result.loc[0, "value"])
    assert result.loc[1, "value"] == 7


def test_accounting_denominators_require_positive_assets_and_book_equity():
    frame = pd.DataFrame({
        "gvkey": ["1", "1", "1"], "datadate": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31"]),
        "seq": [10.0, -1.0, 12.0], "ceq": [10.0, -1.0, 12.0], "pstk": [0.0] * 3,
        "pstkrv": [np.nan] * 3, "pstkl": [np.nan] * 3, "txditc": [0.0] * 3,
        "revt": [20.0] * 3, "cogs": [10.0] * 3, "xsga": [1.0] * 3, "xint": [1.0] * 3,
        "at": [100.0, -5.0, 120.0], "dltt": [10.0] * 3, "dlc": [1.0] * 3,
    })
    result, _ = compute_accounting_exposures(frame)
    assert np.isnan(result.loc[1, "operating_profitability"])
    assert np.isnan(result.loc[1, "investment"])
    assert np.isnan(result.loc[2, "investment"])
    assert np.isnan(result.loc[1, "leverage"])


def test_trade_eligibility_is_explicit_common_equity_screen():
    frame = pd.DataFrame({
        "SecurityType": ["EQTY"] * 4,
        "SecuritySubType": ["COM"] * 4,
        "ShareType": ["NS", "SB", "NS", "NS"],
        "PrimaryExch": ["N", "N", "X", "Q"],
        "TradingStatusFlg": ["A", "A", "A", "X"],
        "adj_close": [10.0] * 4,
        "adj_volume": [100.0] * 4,
        "DlyRet": [0.01] * 4,
    })
    assert np.array_equal(trade_eligibility(frame), [True, False, False, False])


def test_panel_source_change_rebuilds_data_instead_of_relabeling_old_cache(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=24)
    daily = pd.DataFrame({
        "PERMNO": 1, "PERMCO": 1, "SICCD": 100, "DlyCalDt": dates,
        "DlyOpen": 10.0, "DlyHigh": 11.0, "DlyLow": 9.0, "DlyClose": 10.0,
        "DlyRet": 0.01, "DlyVol": 100.0, "DlyCumFacPr": 1.0, "DlyCumFacShr": 1.0,
        "DlyCap": 1000.0, "ShrOut": 100.0, "DlyDelFlg": "N", "DlyRetMissFlg": "NA",
        "SecurityType": "EQTY", "SecuritySubType": "COM", "ShareType": "NS",
        "PrimaryExch": "N", "TradingStatusFlg": "A",
    })
    membership = pd.DataFrame({"PERMNO": [1], "MbrStartDt": [dates[0]], "MbrEndDt": [dates[-1]], "MbrFlg": ["Y"], "INDFAM": ["S&P"]})
    delist = pd.DataFrame({"PERMNO": pd.Series(dtype=int), "DelDlyDt": pd.Series(dtype="datetime64[ns]"), "DelRet": pd.Series(dtype=float)})
    files = {"daily": raw / "daily.parquet", "membership": raw / "membership.parquet", "delistings": raw / "delist.parquet"}
    daily.to_parquet(files["daily"], index=False); membership.to_parquet(files["membership"], index=False); delist.to_parquet(files["delistings"], index=False)
    monkeypatch.setattr("rlalpha.data.panel.validate_raw_bundle", lambda root: {"ok": True, "failures": []})
    monkeypatch.setattr("rlalpha.data.panel.discover_data_files", lambda root: files)
    first = build_panel(raw, processed)
    assert np.isclose(zarr.open_array(str(processed / "panel/features.zarr/close"), mode="r")[0, 0], 10.0)
    daily["DlyClose"] = 20.0
    daily.to_parquet(files["daily"], index=False)
    second = build_panel(raw, processed)
    assert first["build_identity"] != second["build_identity"]
    assert np.isclose(zarr.open_array(str(processed / "panel/features.zarr/close"), mode="r")[0, 0], 20.0)
    assert list(processed.glob("panel.legacy_unverified.*"))
