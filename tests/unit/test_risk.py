from __future__ import annotations

import numpy as np
import pandas as pd

from rlalpha.risk.exposures import STYLE_NAMES, _rolling_market_regression, preprocess_exposures
from rlalpha.risk.ff12 import ff12_industry
from rlalpha.risk.neutralize import RiskNeutralizer
from rlalpha.risk.builder import _accounting_arrays


def test_ff12_representative_sics():
    expected = {100: "NoDur", 2510: "Durbl", 3500: "Manuf", 1311: "Enrgy", 2810: "Chems", 3571: "BusEq", 4813: "Telcm", 4931: "Utils", 5411: "Shops", 2834: "Hlth", 6020: "Money", 9999: "Other"}
    assert {sic: ff12_industry(sic) for sic in expected} == expected
    assert ff12_industry(None) == "Other"


def test_balanced22_shape_and_daily_standardization():
    rng = np.random.default_rng(0)
    styles = pd.DataFrame({name: rng.normal(size=100) for name in STYLE_NAMES})
    styles.loc[:4, "beta_252"] = np.nan
    result = preprocess_exposures(styles, pd.Series(np.tile([100, 2510, 3500, 6020], 25)))
    assert result.matrix.shape == (100, 22)
    assert len(result.columns) == 22
    assert np.allclose(result.matrix[:, -10:].mean(axis=0), 0, atol=1e-12)


def test_ols_residuals_are_orthogonal():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
    values = x @ rng.normal(size=6) + rng.normal(scale=0.1, size=200)
    residual, diagnostics = RiskNeutralizer().residualize_vector("x", values, x, np.ones(200, dtype=bool))
    assert np.max(np.abs(x.T @ residual / 200)) < 1e-10
    assert diagnostics["max_residual_exposure"] < 1e-10


def test_residual_volatility_uses_one_market_model_per_window():
    market = pd.Series(np.linspace(-0.03, 0.04, 8))
    noise = np.array([0.01, -0.02, 0.00, 0.03, -0.01, 0.02, -0.03, 0.01])
    stock = pd.Series(0.005 + 1.7 * market.to_numpy() + noise)
    beta, residual_vol = _rolling_market_regression(stock, market, window=5, min_periods=4)
    x = np.column_stack([np.ones(5), market.iloc[-5:].to_numpy()])
    y = stock.iloc[-5:].to_numpy()
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    expected_residual = y - x @ coefficient
    assert np.isclose(beta.iloc[-1], coefficient[1])
    assert np.isclose(residual_vol.iloc[-1], expected_residual.std(ddof=1))


def test_accounting_exposure_does_not_outlive_ccm_link_interval():
    dates = pd.DatetimeIndex(["2020-07-01", "2020-07-31", "2020-08-03"])
    records = pd.DataFrame({
        "PERMNO": [1], "available_date": [pd.Timestamp("2020-07-01")],
        "linkdt": [pd.Timestamp("2010-01-01")], "linkenddt": [pd.Timestamp("2020-07-31")],
        "book_equity": [10.0], "operating_profitability": [0.2], "investment": [0.1],
        "leverage": [0.3], "sich": [3571],
    })
    arrays, sic = _accounting_arrays(records, dates, np.array([1]), np.ones((3, 1)) * 1000)
    assert np.isfinite(arrays["book_to_market"][:2, 0]).all()
    assert np.isnan(arrays["book_to_market"][2, 0])
    assert np.isnan(sic[2, 0])
