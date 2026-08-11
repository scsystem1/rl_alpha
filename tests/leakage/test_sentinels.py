from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr
import hashlib
import yaml

from rlalpha.data.fundamentals import asof_fundamentals
from rlalpha.data.labels import next_close_forward_return
from rlalpha.data.splits import SPLITS
from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.parser import parse_expression
from rlalpha.leakage.guards import ReadOnlyStateGuard, assert_train_only_context
from rlalpha.utils.hashing import stable_hash
from rlalpha.data.panel import build_panel
from rlalpha.data.store import PanelStore
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.r0 import R0Objective


def test_changing_test_data_does_not_change_train_artifact_hash():
    dates = pd.bdate_range("2018-12-01", "2022-02-01")
    values = np.arange(len(dates), dtype=float)
    train = values[dates <= SPLITS["train"].end]
    before = stable_hash(train.tolist())
    values[dates >= SPLITS["test"].start] = -999
    assert stable_hash(values[dates <= SPLITS["train"].end].tolist()) == before


def test_changing_future_price_does_not_change_past_factor_signal():
    rng = np.random.default_rng(0)
    close = rng.normal(size=(300, 5))
    features = {name: close.copy() for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    node = parse_expression("Mean(Delta($close,5),20)")
    before = evaluate(node, features)
    features["$close"][201:] = 1e9
    after = evaluate(node, features)
    assert np.allclose(before[:201], after[:201], equal_nan=True)


def test_unavailable_fundamental_change_does_not_change_exposure_at_t():
    days = pd.DataFrame({"PERMNO": [1], "DlyCalDt": pd.to_datetime(["2020-06-01"])})
    fundamentals = pd.DataFrame({"PERMNO": [1], "available_date": pd.to_datetime(["2020-07-01"]), "book_equity": [1.0]})
    first = asof_fundamentals(days, fundamentals)
    fundamentals.loc[0, "book_equity"] = 999.0
    second = asof_fundamentals(days, fundamentals)
    assert pd.isna(first.loc[0, "book_equity"]) and pd.isna(second.loc[0, "book_equity"])


def test_last_train_label_exit_does_not_cross_split():
    dates = pd.bdate_range("2018-11-01", "2019-02-01")
    daily = pd.DataFrame({"PERMNO": 1, "DlyCalDt": dates, "DlyRet": 0.001})
    labels = next_close_forward_return(daily, split=SPLITS["train"])
    valid = labels[labels["forward_return_20d"].notna()]
    assert (valid["exit_date"] <= SPLITS["train"].end).all()


def test_search_context_rejects_validation_or_test_metrics():
    assert_train_only_context({"train_objective": 0.01, "pool": []})
    with pytest.raises(ValueError):
        assert_train_only_context({"test_ic": 0.2})


def test_read_only_guard_detects_mutation():
    state = {"pool": ["a"], "model": "hash"}
    with ReadOnlyStateGuard(lambda: state):
        _ = state["pool"]
    with pytest.raises(RuntimeError):
        with ReadOnlyStateGuard(lambda: state):
            state["pool"].append("b")


def test_test_source_mutation_cannot_change_train_panel_reward_admission_or_update_input(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    train_dates = pd.bdate_range("2018-01-02", periods=80)
    test_dates = pd.bdate_range("2022-01-03", periods=40)
    dates = train_dates.append(test_dates)
    rows = []
    for asset in range(5):
        close = 10.0 + asset + np.arange(len(dates)) * 0.01
        for index, date in enumerate(dates):
            rows.append({
                "PERMNO": asset + 1, "PERMCO": asset + 1, "SICCD": 100, "DlyCalDt": date,
                "DlyOpen": close[index], "DlyHigh": close[index] + 1, "DlyLow": close[index] - 1, "DlyClose": close[index],
                "DlyRet": 0.001 * (asset + 1), "DlyVol": 100.0, "DlyCumFacPr": 1.0, "DlyCumFacShr": 1.0,
                "DlyCap": close[index] * 100, "ShrOut": 100.0, "DlyDelFlg": "N", "DlyRetMissFlg": "NA",
                "SecurityType": "EQTY", "SecuritySubType": "COM", "ShareType": "NS", "PrimaryExch": "N", "TradingStatusFlg": "A",
            })
    daily = pd.DataFrame(rows)
    membership = pd.DataFrame({"PERMNO": np.arange(1, 6), "MbrStartDt": dates.min(), "MbrEndDt": dates.max(), "MbrFlg": "Y", "INDFAM": "S&P"})
    delist = pd.DataFrame({"PERMNO": pd.Series(dtype=int), "DelDlyDt": pd.Series(dtype="datetime64[ns]"), "DelRet": pd.Series(dtype=float)})
    files = {"daily": raw / "daily.parquet", "membership": raw / "membership.parquet", "delistings": raw / "delist.parquet"}
    membership.to_parquet(files["membership"], index=False); delist.to_parquet(files["delistings"], index=False)
    monkeypatch.setattr("rlalpha.data.panel.validate_raw_bundle", lambda root: {"ok": True, "failures": []})
    monkeypatch.setattr("rlalpha.data.panel.discover_data_files", lambda root: files)

    def pipeline(processed, source):
        source.to_parquet(files["daily"], index=False)
        build_panel(raw, processed)
        index = __import__("json").loads((processed / "panel/index.json").read_text())
        shape = tuple(index["shape"])
        group = zarr.open_group(str(processed / "panel/risk_exposures.zarr"), mode="w")
        group.create_array("exposures", data=np.ones(shape + (1,), dtype=np.float32), overwrite=True)
        group.attrs["columns"] = ["intercept"]
        panel_manifest = yaml.safe_load((processed / "panel/build_manifest.yaml").read_text())
        pd.DataFrame(columns=["PERMNO", "gvkey", "datadate", "available_date"]).to_parquet(processed / "panel/fundamental_lineage.parquet", index=False)
        (processed / "panel/risk_build_manifest.yaml").write_text(yaml.safe_dump({
            "artifact_version": 3,
            "panel_build_fingerprint": panel_manifest["build_fingerprint"],
            "build_fingerprint": "synthetic-risk-v2",
        }))
        panel = PanelStore(processed).load_split("train")
        node = parse_expression("Mean($close,5)")
        signal = panel.evaluate(node)
        label = panel.target(panel.label)
        mask = panel.target(panel.common_mask) & np.isfinite(label)
        objective = R0Objective(label, mask)
        pool = PoolManager(objective, capacity=2, min_delta=-1.0)
        admission = pool.consider_group([PoolEntry(node.canonical(), node.expr_hash, signal)])
        digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return {"signal": digest(signal), "reward": pool._score(pool.entries).objective, "admission": stable_hash(pool.history), "update_input": stable_hash({"signal": digest(signal), "label": digest(label), "mask": digest(mask)})}

    before = pipeline(tmp_path / "processed_a", daily)
    changed = daily.copy()
    test_rows = changed["DlyCalDt"] >= SPLITS["test"].start
    changed.loc[test_rows, "DlyClose"] *= 1000
    changed.loc[test_rows, "DlyRet"] = -0.75
    after = pipeline(tmp_path / "processed_b", changed)
    assert before == after
