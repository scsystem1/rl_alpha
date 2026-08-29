from __future__ import annotations

from collections.abc import Iterable

from ..dsl.operators import (
    BINARY,
    CONSTANTS,
    CROSS_SECTIONAL,
    FEATURES,
    OPERATORS,
    PAIR_ROLLING,
    ROLLING,
    UNARY,
    WINDOWS,
)
from ..leakage.guards import assert_train_only_context
from ..utils.hashing import stable_hash
from .models import SearchContext


PROMPT_VERSION = "unified_compact_v7"
SYSTEM_PROMPT = "Propose one alpha factor. Return exactly one <expr>...</expr> block and no explanation."

MAX_FACTOR_DEPTH = 6


def _grammar_alternatives(values: Iterable[object]) -> str:
    return " | ".join(f'"{value}"' for value in values)


def _build_dsl_grammar(max_depth: int = MAX_FACTOR_DEPTH) -> str:
    """Build a finite-depth grammar whose every expression contains a feature."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    lines = [
        f'root ::= "<expr>" factor_{max_depth} "</expr>"',
        f"feature ::= {_grammar_alternatives(sorted(FEATURES))}",
        f"constant ::= {_grammar_alternatives(CONSTANTS)}",
        f"window ::= {_grammar_alternatives(WINDOWS)}",
        f"unary_operator ::= {_grammar_alternatives(sorted(UNARY))}",
        f"binary_operator ::= {_grammar_alternatives(sorted(BINARY))}",
        f"rolling_operator ::= {_grammar_alternatives(sorted(ROLLING))}",
        f"pair_rolling_operator ::= {_grammar_alternatives(sorted(PAIR_ROLLING))}",
        f"cross_sectional_operator ::= {_grammar_alternatives(sorted(CROSS_SECTIONAL))}",
        "factor_1 ::= feature",
        "value_1 ::= factor_1 | constant",
    ]
    for depth in range(2, max_depth + 1):
        child = depth - 1
        lines.extend(
            [
                (
                    f"factor_{depth} ::= feature"
                    f' | unary_operator "(" factor_{child} ")"'
                    f' | binary_operator "(" factor_{child} "," value_{child} ")"'
                    f' | binary_operator "(" constant "," factor_{child} ")"'
                    f' | rolling_operator "(" factor_{child} "," window ")"'
                    f' | pair_rolling_operator "(" factor_{child} "," value_{child} "," window ")"'
                    f' | pair_rolling_operator "(" constant "," factor_{child} "," window ")"'
                    f' | cross_sectional_operator "(" factor_{child} ")"'
                ),
                f"value_{depth} ::= factor_{depth} | constant",
            ]
        )
    return "\n".join(lines)


# Structured decoding consumes this grammar separately from the chat prompt.
# Generating it from the typed operator registry keeps Base-LLM and GRPO in
# lockstep without spending the 2B model's natural-language context window.
DSL_GRAMMAR = _build_dsl_grammar()


def _pool_lines(context: SearchContext) -> str:
    return "\n".join(f"- {formula}" for formula in context.pool_formulas) or "- empty"


def build_messages(context: SearchContext) -> list[dict[str, str]]:
    """Build the one shared prompt used by Base-LLM and formal GRPO.

    The caller may sample this same prompt eight times, but may not select a
    different hint/template per answer.  Reward variants are deliberately not
    exposed so the prompt itself cannot confound the R0/R1/R2 comparison.
    """
    payload = context.to_prompt_dict()
    assert_train_only_context(payload)
    user = f"""Current pool:
{_pool_lines(context)}

Allowed elements:
features: {', '.join(sorted(FEATURES))}
operators: {', '.join(sorted(OPERATORS))}
windows: {', '.join(map(str, WINDOWS))}
constants: {', '.join(map(str, CONSTANTS))}

Goal: create one valid, non-duplicate factor that best complements and improves the current pool on training data.
Hints: consider momentum, mean reversion, volatility, price-volume interaction, multi-horizon structure, and cross-sectional ranking; combine signals when useful and never use future information. Keep nodes<=21. Along each path, cumulative lookback must be <=252: Ref/Delta add w; other rolling operators add w-1.

Output: <expr>FORMULA</expr>"""
    assert_train_only_context({"system": SYSTEM_PROMPT, "user": user})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_contract() -> dict[str, object]:
    payload = {
        "version": PROMPT_VERSION,
        "system": SYSTEM_PROMPT,
        "grammar": DSL_GRAMMAR,
        "features": sorted(FEATURES),
        "operators": sorted(OPERATORS),
        "windows": list(WINDOWS),
        "constants": list(CONSTANTS),
        "output_schema": "<expr>FORMULA</expr>",
        "sampling_protocol": "one identical prompt, eight independent completions",
    }
    return {**payload, "hash": stable_hash(payload)}
