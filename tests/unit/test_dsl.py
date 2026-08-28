from __future__ import annotations

import random

import numpy as np
import pytest

from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.grammar import sample_ast
from rlalpha.dsl.parser import ExpressionSyntaxError, parse_expression, parse_llm_response
from rlalpha.dsl.validity import validate_signal
from rlalpha.factors.cache import SignalCache
from rlalpha.utils.numerics import finite_corr


def test_roundtrip_canonical_hash_and_commutative_deduplication():
    first = parse_expression("Add($close,Mean($return,20))")
    second = parse_expression("Add(Mean($return,20),$close)")
    assert first.canonical() == second.canonical()
    assert first.expr_hash == second.expr_hash
    assert parse_expression(first.canonical()).canonical() == first.canonical()
    assert parse_expression("Greater($close,$open)").expr_hash != parse_expression("Greater($open,$close)").expr_hash


def test_strict_llm_protocol_and_no_python_syntax():
    assert parse_llm_response("<expr>Mean($close,5)</expr>").canonical() == "Mean($close,5)"
    with pytest.raises(ExpressionSyntaxError):
        parse_llm_response("text <expr>Mean($close,5)</expr>")
    with pytest.raises(ExpressionSyntaxError):
        parse_expression("__import__('os').system('id')")


def test_lookback_and_fixed_window_constant_rules():
    assert parse_expression("Mean(Ref($close,20),120)").lookback == 139
    with pytest.raises(ValueError):
        parse_expression("Mean($close,7)")
    with pytest.raises(ValueError):
        parse_expression("Add($close,3)")


def test_protected_div_log_ref_and_cross_sectional_zscore():
    close = np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]])
    features = {name: close for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    divided = evaluate(parse_expression("Div($close,0.01)"), features)
    assert np.allclose(divided, close / 0.01)
    logged = evaluate(parse_expression("Log($close)"), features)
    assert np.isfinite(logged).all()
    lagged = evaluate(parse_expression("Ref($close,1)"), features)
    assert np.isnan(lagged[0]).all() and np.allclose(lagged[1], close[0])
    zscore = evaluate(parse_expression("CSZScore($close)"), features)
    assert np.allclose(np.nanmean(zscore, axis=1), 0)


def test_comparisons_propagate_non_finite_operands():
    left = np.array([[1.0, np.nan, np.inf, 0.0]])
    right = np.array([[0.0, 1.0, 2.0, np.nan]])
    features = {
        "$open": right,
        "$high": left,
        "$low": right,
        "$close": left,
        "$volume": right,
        "$return": left,
    }
    greater = evaluate(parse_expression("Greater($close,$open)"), features)
    less = evaluate(parse_expression("Less($close,$open)"), features)
    assert np.array_equal(greater[:, :1], np.array([[1.0]]))
    assert np.array_equal(less[:, :1], np.array([[0.0]]))
    assert np.isnan(greater[:, 1:]).all()
    assert np.isnan(less[:, 1:]).all()


def test_cross_sectional_operators_use_only_same_day_eligible_universe():
    close = np.array([[1.0, 2.0, 1_000_000.0], [3.0, 4.0, -1_000_000.0]])
    features = {name: close.copy() for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    eligible = np.array([[True, True, False], [True, True, False]])
    rank = evaluate(parse_expression("CSRank($close)"), features, eligibility_mask=eligible)
    zscore = evaluate(parse_expression("CSZScore($close)"), features, eligibility_mask=eligible)
    assert np.allclose(rank[:, :2], [[0.5, 1.0], [0.5, 1.0]])
    assert np.isnan(rank[:, 2]).all()
    assert np.allclose(zscore[:, :2], [[-1.0, 1.0], [-1.0, 1.0]])
    assert np.isnan(zscore[:, 2]).all()

    changed = {name: values.copy() for name, values in features.items()}
    changed["$close"][:, 2] *= -7
    reranked = evaluate(parse_expression("CSRank($close)"), changed, eligibility_mask=eligible)
    assert np.allclose(rank[:, :2], reranked[:, :2])


def test_cross_sectional_mask_does_not_erase_pre_membership_rolling_history():
    close = np.arange(1.0, 13.0).reshape(6, 2)
    features = {name: close.copy() for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    eligible = np.ones_like(close, dtype=bool)
    eligible[:4, 1] = False
    result = evaluate(parse_expression("CSRank(Mean($close,5))"), features, eligibility_mask=eligible)
    # Asset 1 enters on day four. Mean still uses its five historically
    # available observations, while only that day's eligible assets are ranked.
    assert np.isfinite(result[4, 1])
    assert result[4, 1] == 1.0


def test_mask_fingerprint_prevents_cross_universe_cache_reuse():
    close = np.array([[1.0, 2.0, 100.0], [2.0, 3.0, 200.0]])
    features = {name: close for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    node = parse_expression("CSRank($close)")
    cache = SignalCache(max_items=8)
    first_mask = np.array([[True, True, False], [True, True, False]])
    second_mask = np.ones_like(first_mask)
    first = evaluate(node, features, cache, eligibility_mask=first_mask)
    second = evaluate(node, features, cache, eligibility_mask=second_mask)
    assert not np.allclose(first, second, equal_nan=True)
    assert len(cache._memory) >= 2


def test_persistent_signal_cache_preserves_fresh_dtype(tmp_path):
    cache = SignalCache(tmp_path)
    values = np.array([[1.123456789]], dtype=np.float64)
    cache.put("factor", values, permanent=True)
    restored = SignalCache(tmp_path).get("factor")
    assert restored is not None
    assert restored.dtype == values.dtype
    assert np.array_equal(restored, values)


def test_typed_grammar_generates_ten_thousand_bounded_asts():
    rng = random.Random(9)
    for _ in range(10_000):
        node = sample_ast(rng)
        assert node.depth <= 6
        assert node.nodes <= 21
        assert node.lookback <= 252


def test_numpy_and_torch_evaluators_match_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from rlalpha.dsl.torch_evaluator import evaluate_torch

    rng = np.random.default_rng(12)
    close = rng.lognormal(size=(80, 12))
    features = {name: close + index for index, name in enumerate(sorted(["$open", "$high", "$low", "$close", "$volume", "$return"]))}
    node = parse_expression("CSZScore(Div(Delta($close,20),Std($return,20)))")
    expected = evaluate(node, features)
    actual = evaluate_torch(node, {key: torch.as_tensor(value) for key, value in features.items()}).cpu().numpy()
    assert np.allclose(actual, expected, equal_nan=True, atol=1e-10)


def test_numpy_and_torch_match_for_nan_mask_ties_and_small_cross_sections():
    torch = pytest.importorskip("torch")
    from rlalpha.dsl.torch_evaluator import evaluate_torch

    close = np.array([[1.0, 1.0, 9.0], [np.nan, 2.0, 3.0], [4.0, np.inf, 4.0]])
    opened = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 4.0], [5.0, 0.0, 4.0]])
    features = {name: close.copy() for name in ["$high", "$low", "$close", "$volume", "$return"]}
    features["$open"] = opened
    eligible = np.array([[True, True, False], [True, True, True], [True, False, True]])
    for formula in ("CSRank($close)", "CSZScore($close)", "CSRank(Greater($close,$open))"):
        node = parse_expression(formula)
        expected = evaluate(node, features, eligibility_mask=eligible)
        actual = evaluate_torch(node, {key: torch.as_tensor(value) for key, value in features.items()}, eligibility_mask=torch.as_tensor(eligible)).cpu().numpy()
        assert np.allclose(actual, expected, equal_nan=True, atol=1e-12)


def test_ast_subtrees_are_shared_through_signal_cache():
    values = np.arange(600, dtype=float).reshape(30, 20)
    features = {name: values for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    node = parse_expression("Add(Mean($close,20),Mean($close,20))")
    cache = SignalCache(max_items=8)
    actual = evaluate(node, features, cache)
    mean_hash = parse_expression("Mean($close,20)").expr_hash
    mean_key = next(key for key in cache._memory if key.endswith(mean_hash))
    assert any(key.endswith(node.expr_hash) for key in cache._memory)
    assert np.allclose(actual, 2 * cache.get(mean_key), equal_nan=True)


def test_vectorized_validity_pool_correlation_matches_daily_reference():
    rng = np.random.default_rng(73)
    signal = rng.normal(size=(300, 120))
    pool = 0.4 * signal + rng.normal(size=signal.shape)
    membership = rng.random(signal.shape) > 0.03
    signal[4:9, :5] = np.nan
    result = validate_signal(signal, membership, [pool])
    valid_days = np.flatnonzero((np.isfinite(signal) & membership).sum(axis=1) >= 100)
    daily = [finite_corr(signal[day][membership[day]], pool[day][membership[day]]) for day in valid_days]
    assert np.isclose(result.mean_abs_daily_corr, np.nanmean(np.abs(daily)), atol=1e-12)


def test_near_duplicate_metric_does_not_cancel_alternating_signs():
    rng = np.random.default_rng(91)
    signal = rng.normal(size=(300, 120))
    pool = signal.copy()
    pool[1::2] *= -1
    membership = np.ones_like(signal, dtype=bool)
    result = validate_signal(signal, membership, [pool])
    assert not result.valid
    assert result.reason == "near_duplicate_signal"
    assert result.mean_abs_daily_corr > 0.99
    assert abs(result.pooled_correlation) < 0.05


def test_signal_coverage_is_not_overwritten_by_pool_correlation_coverage():
    rng = np.random.default_rng(912)
    signal = rng.normal(size=(300, 120))
    pool = rng.normal(size=signal.shape)
    pool[:150] = np.nan
    membership = np.ones_like(signal, dtype=bool)
    result = validate_signal(signal, membership, [pool])
    assert result.valid
    assert result.coverage == 1.0
    assert np.isclose(result.correlation_coverage, 0.5)


def test_numba_rank_correlation_matches_scipy_with_ties_and_missing_values():
    from scipy.stats import rankdata
    from rlalpha.dsl.validity import _daily_rank_corr_exact, _daily_rank_corr_exact_serial

    rng = np.random.default_rng(913)
    signal = np.round(rng.normal(size=(30, 140)), 1)
    pool = np.round(rng.normal(size=signal.shape), 1)
    membership = rng.random(signal.shape) > 0.08
    signal[3:7, :11] = np.nan
    pool[9:13, 17:29] = np.nan
    valid_days = (np.isfinite(signal) & membership).sum(axis=1) >= 100
    actual = _daily_rank_corr_exact(signal, pool, membership, valid_days)
    serial = _daily_rank_corr_exact_serial(signal, pool, membership, valid_days)
    expected = np.full(len(signal), np.nan)
    for day in np.flatnonzero(valid_days):
        common = membership[day] & np.isfinite(signal[day]) & np.isfinite(pool[day])
        if common.sum() >= 3:
            expected[day] = finite_corr(rankdata(signal[day, common]), rankdata(pool[day, common]))
    assert np.allclose(actual, expected, equal_nan=True, atol=1e-12)
    assert np.allclose(serial, expected, equal_nan=True, atol=1e-12)


def test_intrinsic_validity_failure_skips_pool_redundancy(monkeypatch):
    import rlalpha.dsl.validity as validity

    signal = np.full((300, 120), np.nan)
    signal[:, :10] = 1.0
    membership = np.ones_like(signal, dtype=bool)
    monkeypatch.setattr(
        validity,
        "_daily_rank_corr_exact",
        lambda *args: (_ for _ in ()).throw(AssertionError("redundancy should be skipped")),
    )
    result = validity.validate_signal(signal, membership, [np.ones_like(signal)])
    assert result.reason == "coverage_failure"


@pytest.mark.parametrize("formula", ["Mean($close,20)", "Var($return,10)", "Med($close,5)", "Mad($return,5)", "TSRank($close,10)", "WMA($volume,5)", "Corr($close,$volume,20)"])
def test_optimized_rolling_kernels_match_reference_path(formula):
    torch = pytest.importorskip("torch")
    from rlalpha.dsl.torch_evaluator import evaluate_torch

    rng = np.random.default_rng(22)
    values = rng.normal(size=(50, 7))
    values[3, 2] = np.nan
    features = {name: values + index for index, name in enumerate(sorted(["$open", "$high", "$low", "$close", "$volume", "$return"]))}
    node = parse_expression(formula)
    expected = evaluate_torch(node, {key: torch.as_tensor(value) for key, value in features.items()}).cpu().numpy()
    actual = evaluate(node, features)
    assert np.allclose(actual, expected, equal_nan=True, atol=1e-9)
