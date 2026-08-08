from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlalpha.data.store import PanelStore
from rlalpha.dsl.parser import parse_expression


@pytest.mark.real_data
def test_real_data_small_interval_dsl_smoke():
    root = Path("/data/sunyuxiang/rl_alpha/processed")
    if not (root / "panel/index.json").exists():
        pytest.skip("real processed panel is unavailable")
    panel = PanelStore(root).load_split("validation", history=30, start="2019-01-02", end="2019-03-29")
    signal = panel.evaluate(parse_expression("Delta($close,20)"))
    assert signal.shape == panel.target(panel.membership).shape
    assert np.isfinite(signal).any()
    assert len(panel.exposure_names) == 22
    assert panel.exposure_names[0] == "intercept"
