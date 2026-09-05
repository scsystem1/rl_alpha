from __future__ import annotations

from collections.abc import Iterable
import math

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


PROMPT_VERSION = "unified_rolling_summary_v8"
SYSTEM_PROMPT = "Propose one alpha factor. Return exactly one <expr>...</expr> block and no explanation."

USER_TEMPLATE = """Task: predict 20-trading-day returns from next-close entry to t+21 close exit.
Goal: add one valid, non-duplicate factor that improves the pool's mean daily cross-sectional Pearson RNIC after risk neutralization. The Balanced-22 risk benchmark removes intercept, industry and style exposures. Seek information beyond this benchmark using only past/current data.

Current pool:
{pool}
{summary}
Allowed elements:
features: {features}
operators: {operators}
windows: {windows}
constants: {constants}

Keep nodes<=21 and depth<=6. Along each path, cumulative lookback must be <=252: Ref/Delta add w; other rolling operators add w-1.
Output: <expr>FORMULA</expr>"""

POOL_ROW_TEMPLATE = "- w={weight:+.3f} {formula}"
SUMMARY_TEMPLATE = (
    "Training rolling RNIC={rnic:+.4f}: weights fitted on earlier years, scored on later years.\n"
    "w averages signed normalized fit coefficients across three folds; negative means inverse use. Coefficients are not factor importance.\n"
)

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
    summary = context.prompt_summary
    if summary is None:
        return "\n".join(f"- {formula}" for formula in context.pool_formulas) or "- empty"
    if (len(summary.normalized_fold_weights) != len(context.pool_formulas)
            or not math.isfinite(summary.oof_mean_rnic)
            or not all(math.isfinite(w) for w in summary.normalized_fold_weights)):
        raise ValueError("prompt summary must contain finite, formula-aligned training evidence")
    return "\n".join(POOL_ROW_TEMPLATE.format(weight=weight, formula=formula) for formula, weight in
        zip(context.pool_formulas, summary.normalized_fold_weights, strict=True)) or "- empty"


def build_messages(context: SearchContext) -> list[dict[str, str]]:
    """Build the one shared prompt used by Base-LLM and formal GRPO.

    The caller may sample this same prompt eight times, but may not select a
    different hint/template per answer.  Reward variants are deliberately not
    exposed so the prompt itself cannot confound the R0/R1/R2 comparison.
    """
    payload = context.to_prompt_dict()
    assert_train_only_context(payload)
    summary = ""
    if context.prompt_summary is not None:
        summary = SUMMARY_TEMPLATE.format(rnic=context.prompt_summary.oof_mean_rnic)
    user = USER_TEMPLATE.format(pool=_pool_lines(context), summary=summary,
        features=', '.join(sorted(FEATURES)), operators=', '.join(sorted(OPERATORS)),
        windows=', '.join(map(str, WINDOWS)), constants=', '.join(map(str, CONSTANTS)))
    assert_train_only_context({"system": SYSTEM_PROMPT, "user": user})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_contract() -> dict[str, object]:
    payload = {
        "version": PROMPT_VERSION,
        "system": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "pool_row_template": POOL_ROW_TEMPLATE,
        "summary_template": SUMMARY_TEMPLATE,
        "summary": "canonical neutralized rolling RNIC; mean signed L1-normalized fold weights; train only",
        "grammar": DSL_GRAMMAR,
        "features": sorted(FEATURES),
        "operators": sorted(OPERATORS),
        "windows": list(WINDOWS),
        "constants": list(CONSTANTS),
        "output_schema": "<expr>FORMULA</expr>",
        "sampling_protocol": "one identical prompt, eight independent completions",
    }
    return {**payload, "hash": stable_hash(payload)}
