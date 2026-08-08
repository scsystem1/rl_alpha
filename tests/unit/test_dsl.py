from __future__ import annotations

import random

import numpy as np
import pytest

from rlalpha.dsl.evaluator import evaluate
from rlalpha.dsl.grammar import sample_ast
from rlalpha.dsl.parser import ExpressionSyntaxError, parse_expression, parse_llm_response


def test_roundtrip_canonical_hash_and_commutative_deduplication():
    first = parse_expression("Add($close,Mean($return,20))")
    second = parse_expression("Add(Mean($return,20),$close)")
    assert first.canonical() == second.canonical()
    assert first.expr_hash == second.expr_hash
    assert parse_expression(first.canonical()).canonical() == first.canonical()


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

