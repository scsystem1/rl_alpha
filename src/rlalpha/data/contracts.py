from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetContract:
    name: str
    required: frozenset[str]


CONTRACTS = {
    "daily": DatasetContract("daily", frozenset({"PERMNO", "DlyCalDt", "DlyRet", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol", "DlyCumFacPr", "DlyCumFacShr", "DlyCap", "ShrOut", "DlyDelFlg"})),
    "membership": DatasetContract("membership", frozenset({"PERMNO", "MbrStartDt", "MbrEndDt", "MbrFlg", "INDFAM"})),
    "market": DatasetContract("market", frozenset({"DlyCalDt", "vwretd", "vwretx", "ewretd", "sprtrn", "spindx"})),
    "ccm": DatasetContract("ccm", frozenset({"gvkey", "linkprim", "linktype", "lpermno", "USEDFLAG", "linkdt", "linkenddt"})),
    "fundamentals": DatasetContract("fundamentals", frozenset({"gvkey", "datadate", "indfmt", "datafmt", "popsrc", "consol", "curcd", "at", "seq", "ceq", "revt", "cogs", "dltt", "dlc"})),
    "delistings": DatasetContract("delistings", frozenset({"PERMNO", "DelDlyDt", "DelRet"})),
}

DAILY_FEATURE_MAP = {
    "$open": "adj_open",
    "$high": "adj_high",
    "$low": "adj_low",
    "$close": "adj_close",
    "$volume": "adj_volume",
    "$return": "DlyRet",
}

