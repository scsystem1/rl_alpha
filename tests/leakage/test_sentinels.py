from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlalpha.data.fundamentals import asof_fundamentals
from rlalpha.data.labels import next_close_forward_return
from rlalpha.data.splits import SPLITS
from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.parser import parse_expression
from rlalpha.leakage.guards import ReadOnlyStateGuard, assert_train_only_context
from rlalpha.utils.hashing import stable_hash


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

