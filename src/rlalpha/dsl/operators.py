from __future__ import annotations

FEATURES = frozenset({"$open", "$high", "$low", "$close", "$volume", "$return"})
WINDOWS = (1, 5, 10, 20, 40, 60, 120, 252)
CONSTANTS = (-2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0)
UNARY = frozenset({"Abs", "Sign", "Log"})
BINARY = frozenset({"Add", "Sub", "Mul", "Div", "Greater", "Less"})
ROLLING = frozenset({"Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad", "Delta", "WMA", "EMA", "TSRank"})
PAIR_ROLLING = frozenset({"Cov", "Corr"})
CROSS_SECTIONAL = frozenset({"CSRank", "CSZScore"})
OPERATORS = UNARY | BINARY | ROLLING | PAIR_ROLLING | CROSS_SECTIONAL
ARITY = {**{name: 1 for name in UNARY | CROSS_SECTIONAL}, **{name: 2 for name in BINARY | ROLLING}, **{name: 3 for name in PAIR_ROLLING}}
COMMUTATIVE = frozenset({"Add", "Mul", "Greater", "Less"})

