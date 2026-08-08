from __future__ import annotations

import numpy as np
import pandas as pd

from rlalpha.data.adjustments import apply_crsp_adjustments, fill_missing_delisting_returns
from rlalpha.data.fundamentals import asof_fundamentals, filter_standard_fundamentals, select_ccm_links
from rlalpha.data.labels import next_close_forward_return
from rlalpha.data.membership import membership_on_date
from rlalpha.data.splits import DateSplit


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
