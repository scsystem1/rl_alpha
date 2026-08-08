from __future__ import annotations

import numpy as np
import pandas as pd

from rlalpha.risk.exposures import STYLE_NAMES, preprocess_exposures
from rlalpha.risk.ff12 import ff12_industry
from rlalpha.risk.neutralize import RiskNeutralizer


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

