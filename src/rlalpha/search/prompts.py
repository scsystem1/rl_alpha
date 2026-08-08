from __future__ import annotations

from ..dsl.operators import CONSTANTS, FEATURES, OPERATORS, WINDOWS
from ..leakage.guards import assert_train_only_context
from .models import SearchContext

SYSTEM_PROMPT = "You generate one syntactically exact alpha formula. Use only the case-sensitive grammar supplied by the user. Output exactly one <expr>...</expr> block and no explanation."

HINTS = ("momentum", "mean reversal", "volatility", "price-volume interaction", "multi-scale structure")

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


def build_messages(context: SearchContext, hint: str) -> list[dict[str, str]]:
    payload = context.to_prompt_dict()
    assert_train_only_context(payload)
    pool = "\n".join(f"- {formula} | weight={context.pool_weights[index] if index < len(context.pool_weights) else 0.0:.8g}" for index, formula in enumerate(context.pool_formulas)) or "- empty"
    user = f"""Exploration hint: {hint}
Features (include the $ prefix): {', '.join(sorted(FEATURES))}
Exact case-sensitive grammar:
signal := feature | constant | Unary(signal) | Binary(signal,signal) | Rolling(signal,window) | PairRolling(signal,signal,window) | CrossSectional(signal)
Unary := Abs | Sign | Log
Binary := Add | Sub | Mul | Div | Greater | Less
Rolling := Ref | Mean | Sum | Std | Var | Max | Min | Med | Mad | Delta | WMA | EMA | TSRank
PairRolling := Cov | Corr
CrossSectional := CSRank | CSZScore
Windows: {', '.join(map(str, WINDOWS))}
Constants: {', '.join(map(str, CONSTANTS))}
Limits: depth<=6, nodes<=21, lookback<=252. No future access and no Python.
Valid examples: <expr>CSRank(Delta($close,20))</expr> ; <expr>Div(Mean($return,20),Add(Std($return,20),0.01))</expr> ; <expr>Sub(Corr($close,$volume,20),Corr($return,$volume,5))</expr>
Never invent SMA, SIGMA, Ema, $window, $vol, infix arithmetic, semicolons, or extra arguments.
Current train-only pool version: {context.pool_version}
Current train objective: {context.train_objective:.8g}
Current pool:
{pool}
Propose a valid unique complement to the pool. Output only <expr>FORMULA</expr>."""
    assert_train_only_context({"system": SYSTEM_PROMPT, "user": user})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
