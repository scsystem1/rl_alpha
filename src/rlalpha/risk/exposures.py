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


def _rolling_market_regression(
    stock_return: pd.Series,
    market_return: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 126,
) -> tuple[pd.Series, pd.Series]:
    """Rolling market beta and same-window regression residual volatility.

    Each output date is estimated from one common-observation OLS window with
    an intercept.  In particular, residual volatility is *not* the rolling
    standard deviation of residuals formed with a different beta on each day.
    The returned residual standard deviation uses ``ddof=1``, matching an
    explicit standard deviation of the residual vector in the window.
    """
    stock = pd.to_numeric(stock_return, errors="coerce").astype(float)
    market = pd.to_numeric(market_return, errors="coerce").astype(float)
    common = stock.notna() & market.notna() & np.isfinite(stock) & np.isfinite(market)
    x = market.where(common, 0.0)
    y = stock.where(common, 0.0)
    rolling = lambda values: values.rolling(window, min_periods=1).sum()
    count = rolling(common.astype(float))
    sx, sy = rolling(x), rolling(y)
    sxx, syy, sxy = rolling(x * x), rolling(y * y), rolling(x * y)
    with np.errstate(all="ignore"):
        centered_xx = sxx - sx * sx / count
        centered_yy = syy - sy * sy / count
        centered_xy = sxy - sx * sy / count
        beta = centered_xy / centered_xx
        residual_ss = centered_yy - beta * centered_xy
        residual_variance = residual_ss / (count - 1.0)
    valid = (count >= min_periods) & (centered_xx > 1e-12) & (residual_variance >= -1e-12)
    beta = beta.where(valid)
    residual_volatility = np.sqrt(residual_variance.clip(lower=0.0)).where(valid)
    return beta, residual_volatility


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
    diagnostics["rank"] = int(np.linalg.matrix_rank(matrix)) if n else 0
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
        group["beta_252"], group["resid_vol_252"] = _rolling_market_regression(returns, market_returns)
        log_growth = np.log1p(returns.where(returns > -1))
        group["reversal_1m"] = np.expm1(log_growth.rolling(21, min_periods=21).sum())
        group["momentum_12_1"] = np.expm1(log_growth.shift(21).rolling(232, min_periods=232).sum())
        illiquidity = returns.abs() / (group["dollar_volume"].abs() + 1e-12)
        group["amihud_20"] = np.log(illiquidity.rolling(20, min_periods=15).mean())
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)
