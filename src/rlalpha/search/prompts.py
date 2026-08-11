from __future__ import annotations

from ..dsl.operators import CONSTANTS, FEATURES, OPERATORS, WINDOWS
from ..leakage.guards import assert_train_only_context
from ..utils.hashing import stable_hash
from .models import SearchContext


PROMPT_VERSION = "unified_compact_v1"
SYSTEM_PROMPT = "Propose one alpha factor. Return exactly one <expr>...</expr> block and no explanation."

# This grammar is enforced by structured decoding.  It is part of the prompt
# contract/hash, but repeating the full production rules in natural language
# only wastes the 2B model's context window.
DSL_GRAMMAR = r'''
root ::= "<expr>" expr "</expr>"
expr ::= feature | constant | unary | binary | rolling | pair_rolling | cross_sectional
feature ::= "$open" | "$high" | "$low" | "$close" | "$volume" | "$return"
constant ::= "-2.0" | "-1.0" | "-0.5" | "-0.01" | "0.01" | "0.5" | "1.0" | "2.0"
window ::= "1" | "5" | "10" | "20" | "40" | "60" | "120" | "252"
unary ::= ("Abs" | "Sign" | "Log") "(" expr ")"
binary ::= ("Add" | "Sub" | "Mul" | "Div" | "Greater" | "Less") "(" expr "," expr ")"
rolling ::= ("Ref" | "Mean" | "Sum" | "Std" | "Var" | "Max" | "Min" | "Med" | "Mad" | "Delta" | "WMA" | "EMA" | "TSRank") "(" expr "," window ")"
pair_rolling ::= ("Cov" | "Corr") "(" expr "," expr "," window ")"
cross_sectional ::= ("CSRank" | "CSZScore") "(" expr ")"
'''.strip()


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
Hints: consider momentum, mean reversion, volatility, price-volume interaction, multi-horizon structure, and cross-sectional ranking; combine signals when useful and never use future information. Keep depth<=6, nodes<=21, lookback<=252.

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
