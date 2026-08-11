from __future__ import annotations

import numpy as np
import pandas as pd


ELIGIBLE_SECURITY_TYPES = frozenset({"EQTY"})
ELIGIBLE_SECURITY_SUBTYPES = frozenset({"COM"})
ELIGIBLE_SHARE_TYPES = frozenset({"NS"})
ELIGIBLE_PRIMARY_EXCHANGES = frozenset({"A", "N", "Q"})
ELIGIBLE_TRADING_STATUSES = frozenset({"A"})


def trade_eligibility(frame: pd.DataFrame) -> np.ndarray:
    """Known-at-close common-equity/tradability screen for a daily table."""
    required = {
        "SecurityType", "SecuritySubType", "ShareType", "PrimaryExch",
        "TradingStatusFlg", "adj_close", "adj_volume", "DlyRet",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing trade eligibility fields: {sorted(missing)}")

    def normalized(column: str) -> pd.Series:
        return frame[column].astype("string").str.strip().str.upper()

    close = pd.to_numeric(frame["adj_close"], errors="coerce")
    volume = pd.to_numeric(frame["adj_volume"], errors="coerce")
    returns = pd.to_numeric(frame["DlyRet"], errors="coerce")
    return (
        normalized("SecurityType").isin(ELIGIBLE_SECURITY_TYPES)
        & normalized("SecuritySubType").isin(ELIGIBLE_SECURITY_SUBTYPES)
        & normalized("ShareType").isin(ELIGIBLE_SHARE_TYPES)
        & normalized("PrimaryExch").isin(ELIGIBLE_PRIMARY_EXCHANGES)
        & normalized("TradingStatusFlg").isin(ELIGIBLE_TRADING_STATUSES)
        & np.isfinite(close) & close.gt(0)
        & np.isfinite(volume) & volume.gt(0)
        & np.isfinite(returns)
    ).to_numpy(dtype=bool)
