from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ff12 import FF12_NAMES, ff12_industry

STYLE_NAMES = ("size", "beta_252", "momentum_12_1", "reversal_1m", "resid_vol_252", "amihud_20", "book_to_market", "operating_profitability", "investment", "leverage")


@dataclass(frozen=True)
class ExposureDay:
    matrix: np.ndarray
    columns: tuple[str, ...]
    diagnostics: dict[str, object]


def preprocess_exposures(styles: pd.DataFrame, sic: pd.Series) -> ExposureDay:
    n = len(styles)
    processed = np.zeros((n, len(STYLE_NAMES)), dtype=float)
    diagnostics: dict[str, object] = {"styles": {}}
    for index, name in enumerate(STYLE_NAMES):
        values = pd.to_numeric(styles[name], errors="coerce").to_numpy(float) if name in styles else np.full(n, np.nan)
        finite = np.isfinite(values)
        record = {"missing_rate": float((~finite).mean()), "disabled": False}
        if finite.sum() < max(3, min(30, n // 4)):
            record["disabled"] = True
            processed[:, index] = 0.0
        else:
            low, high = np.quantile(values[finite], [0.01, 0.99])
            clipped = np.clip(values, low, high)
            median = np.nanmedian(clipped)
            clipped[~np.isfinite(clipped)] = median
            std = clipped.std(ddof=0)
            if std <= 1e-12:
                record["disabled"] = True
                processed[:, index] = 0.0
            else:
                processed[:, index] = (clipped - clipped.mean()) / std
                record.update({"mean": float(processed[:, index].mean()), "std": float(processed[:, index].std(ddof=0)), "fill_rate": float((~finite).mean())})
        diagnostics["styles"][name] = record
    industries = sic.map(ff12_industry)
    industry_names = FF12_NAMES[:-1]
    dummies = np.column_stack([(industries == name).to_numpy(float) for name in industry_names])
    matrix = np.column_stack([np.ones(n), dummies, processed])
    columns = ("intercept",) + tuple(f"industry_{name}" for name in industry_names) + STYLE_NAMES
    diagnostics["condition_number"] = float(np.linalg.cond(matrix)) if n else float("nan")
    return ExposureDay(matrix, columns, diagnostics)


def compute_market_styles(daily: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Compute the six price/return styles without future observations."""
    frame = daily.copy().merge(market[["DlyCalDt", "vwretd"]], on="DlyCalDt", how="left").sort_values(["PERMNO", "DlyCalDt"])
    frame["size"] = np.log(pd.to_numeric(frame["DlyCap"], errors="coerce").where(frame["DlyCap"] > 0))
    frame["dollar_volume"] = frame["DlyClose"].abs() * frame["DlyVol"]
    outputs = []
    for _, group in frame.groupby("PERMNO", sort=False):
        group = group.copy()
        returns = pd.to_numeric(group["DlyRet"], errors="coerce")
        market_returns = pd.to_numeric(group["vwretd"], errors="coerce")
        covariance = returns.rolling(252, min_periods=126).cov(market_returns)
        variance = market_returns.rolling(252, min_periods=126).var()
        group["beta_252"] = covariance / variance.where(variance.abs() > 1e-12)
        residual = returns - group["beta_252"] * market_returns
        group["resid_vol_252"] = residual.rolling(252, min_periods=126).std()
        log_growth = np.log1p(returns.where(returns > -1))
        group["reversal_1m"] = np.expm1(log_growth.rolling(21, min_periods=21).sum())
        group["momentum_12_1"] = np.expm1(log_growth.shift(21).rolling(232, min_periods=232).sum())
        illiquidity = returns.abs() / (group["dollar_volume"].abs() + 1e-12)
        group["amihud_20"] = np.log(illiquidity.rolling(20, min_periods=15).mean())
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)

