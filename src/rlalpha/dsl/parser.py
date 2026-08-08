from __future__ import annotations

import re

from .ast import Call, Constant, Feature, Node, Window, validate_limits
from .operators import ARITY, OPERATORS, PAIR_ROLLING, ROLLING

TOKEN = re.compile(r"\s*(\$[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*|[-+]?(?:\d+(?:\.\d*)?|\.\d+)|[(),])")
EXPR_TAG = re.compile(r"^\s*<expr>\s*(.*?)\s*</expr>\s*$", re.DOTALL | re.IGNORECASE)


class ExpressionSyntaxError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str):
        self.tokens = TOKEN.findall(text)
        joined = "".join(self.tokens).replace(" ", "")
        if joined != re.sub(r"\s+", "", text):
            raise ExpressionSyntaxError("invalid token")
        self.position = 0

    def take(self) -> str:
        if self.position >= len(self.tokens):
            raise ExpressionSyntaxError("unexpected end of expression")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def parse(self, window_expected: bool = False) -> Node:
        token = self.take()
        if token.startswith("$"):
            return Feature(token.lower())
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", token):
            value = float(token)
            return Window(int(value)) if window_expected and value.is_integer() else Constant(value)
        if token not in OPERATORS:
            raise ExpressionSyntaxError(f"unknown operator {token}")
        if self.take() != "(":
            raise ExpressionSyntaxError(f"expected '(' after {token}")
        args = []
        for index in range(ARITY[token]):
            args.append(self.parse(window_expected=(token in ROLLING | PAIR_ROLLING and index == ARITY[token] - 1)))
            separator = self.take()
            expected = ")" if index == ARITY[token] - 1 else ","
            if separator != expected:
                raise ExpressionSyntaxError(f"expected '{expected}', got '{separator}'")
        return Call(token, tuple(args))


def parse_expression(text: str, enforce_limits: bool = True) -> Node:
    parser = _Parser(text)
    node = parser.parse()
    if parser.position != len(parser.tokens):
        raise ExpressionSyntaxError("trailing tokens")
    if enforce_limits:
        validate_limits(node)
    return node


def parse_llm_response(text: str) -> Node:
    match = EXPR_TAG.fullmatch(text)
    if not match:
        raise ExpressionSyntaxError("response must contain exactly one <expr>...</expr>")
    return parse_expression(match.group(1))

