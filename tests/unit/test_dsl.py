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


def test_ast_subtrees_are_shared_through_signal_cache():
    values = np.arange(600, dtype=float).reshape(30, 20)
    features = {name: values for name in ["$open", "$high", "$low", "$close", "$volume", "$return"]}
    node = parse_expression("Add(Mean($close,20),Mean($close,20))")
    cache = SignalCache(max_items=8)
    actual = evaluate(node, features, cache)
    mean_hash = parse_expression("Mean($close,20)").expr_hash
    assert mean_hash in cache._memory
    assert node.expr_hash in cache._memory
    assert np.allclose(actual, 2 * cache.get(mean_hash), equal_nan=True)


def test_vectorized_validity_pool_correlation_matches_daily_reference():
    rng = np.random.default_rng(73)
    signal = rng.normal(size=(300, 120))
    pool = 0.4 * signal + rng.normal(size=signal.shape)
    membership = rng.random(signal.shape) > 0.03
    signal[4:9, :5] = np.nan
    result = validate_signal(signal, membership, [pool])
    valid_days = np.flatnonzero((np.isfinite(signal) & membership).sum(axis=1) >= 100)
    daily = [finite_corr(signal[day][membership[day]], pool[day][membership[day]]) for day in valid_days]
    assert np.isclose(result.max_pool_correlation, abs(np.nanmean(daily)), atol=1e-12)


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
