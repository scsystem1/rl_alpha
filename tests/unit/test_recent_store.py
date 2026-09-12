from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rlalpha.data.store import PanelStore


def _store(monkeypatch, tmp_path):
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    returns = np.full((len(dates), 4), .001, dtype=np.float32)
    arrays = {f"features.zarr/{name}": returns.copy() for name in ("open", "high", "low", "close", "volume", "return")}
    arrays.update({"returns.zarr/daily_total_return": returns,
        "returns.zarr/forward_return_20d": np.full_like(returns, np.nan),
        "membership.zarr/membership": np.ones_like(returns, dtype=bool),
        "eligibility.zarr/trade_eligibility": np.ones_like(returns, dtype=bool),
        "risk_exposures.zarr/exposures": np.ones((*returns.shape, 1))})
    reads = []

    class Array:
        def __init__(self, name):
            self.name, self.shape = name, arrays[name].shape

        def __getitem__(self, key):
            reads.append((self.name, key))
            return arrays[self.name][key]

    store = PanelStore(tmp_path)
    store.__dict__.update(index={"dates": dates.astype(str).tolist(), "permnos": [1, 2, 3, 4]},
        build_manifest={"build_fingerprint": "panel"}, risk_build_manifest={"build_fingerprint": "risk"})
    monkeypatch.setattr(store, "_array", Array)
    monkeypatch.setattr("rlalpha.data.store.zarr.open_group", lambda *a, **k: SimpleNamespace(attrs={"columns": ["constant"]}))
    return store, arrays, reads


@pytest.mark.parametrize("boundary,start,end", [("2018-12-20", "2018-07-01", "2020-06-30"), ("2021-12-20", "2021-07-01", "2022-06-30")])
def test_interval_restores_old_boundary_labels_and_censors_its_own_end(monkeypatch, tmp_path, boundary, start, end):
    store, arrays, reads = _store(monkeypatch, tmp_path)
    panel = store.load_interval("train", start, end)
    labels = panel.target(panel.label)
    idx = panel.target_dates.get_loc(boundary)
    assert labels[idx, 0] == pytest.approx((1 + float(np.float32(.001))) ** 20 - 1)
    assert np.isnan(labels[-21:]).all()
    assert np.isfinite(labels[:-21]).all()
    assert panel.target_slice.start == 252
    assert np.isnan(panel.label[:panel.target_slice.start]).all()
    assert all(item.stop <= store.dates.searchsorted(pd.Timestamp(end), side="right") for _, item in reads)
    assert not any(name == "returns.zarr/forward_return_20d" for name, _ in reads)


def test_interval_respects_missing_returns_and_never_uses_future(monkeypatch, tmp_path):
    store, arrays, _ = _store(monkeypatch, tmp_path)
    end = "2020-06-30"
    position = store.dates.get_loc("2020-01-15")
    arrays["returns.zarr/daily_total_return"][position, 0] = np.nan
    first = store.load_interval("train", "2018-07-01", end)
    idx = first.target_dates.get_loc("2020-01-15")
    assert np.isnan(first.target(first.label)[idx - 21:idx - 1, 0]).all()
    arrays["returns.zarr/daily_total_return"][store.dates > end] = 9.0
    second = store.load_interval("train", "2018-07-01", end)
    np.testing.assert_equal(first.label, second.label)
    np.testing.assert_equal(first.daily_return, second.daily_return)


def test_interval_validation_and_legacy_labels(monkeypatch, tmp_path):
    store, _, _ = _store(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="chronological"):
        store.load_interval("train", "2020-01-01", "2019-01-01")
    with pytest.raises(ValueError, match="cover"):
        store.load_interval("test", "2022-01-01", "2025-12-31")
    with pytest.raises(ValueError, match="nonnegative"):
        store.load_interval("train", "2019-01-01", "2020-01-01", history=-1)
    legacy = store.load_split("train", start="2018-07-01", end="2018-12-31")
    assert np.isnan(legacy.label).all()
