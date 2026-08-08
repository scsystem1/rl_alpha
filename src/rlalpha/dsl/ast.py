from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .operators import ARITY, COMMUTATIVE, CONSTANTS, FEATURES, OPERATORS, PAIR_ROLLING, ROLLING, WINDOWS
from ..utils.hashing import sha256_text


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(float(value), ".12g")


class Node:
    def canonical(self) -> str:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError

    @property
    def nodes(self) -> int:
        raise NotImplementedError

    @property
    def lookback(self) -> int:
        raise NotImplementedError

    @property
    def expr_hash(self) -> str:
        return sha256_text(self.canonical())


@dataclass(frozen=True)
class Feature(Node):
    name: str

    def __post_init__(self) -> None:
        if self.name not in FEATURES:
            raise ValueError(f"unknown feature {self.name}")

    def canonical(self) -> str:
        return self.name

    depth = property(lambda self: 1)
    nodes = property(lambda self: 1)
    lookback = property(lambda self: 0)


@dataclass(frozen=True)
class Constant(Node):
    value: float

    def __post_init__(self) -> None:
        if not any(abs(self.value - allowed) < 1e-12 for allowed in CONSTANTS):
            raise ValueError(f"constant {self.value} is outside the fixed set")

    def canonical(self) -> str:
        return _number(self.value)

    depth = property(lambda self: 1)
    nodes = property(lambda self: 1)
    lookback = property(lambda self: 0)


@dataclass(frozen=True)
class Window(Node):
    value: int

    def __post_init__(self) -> None:
        if self.value not in WINDOWS:
            raise ValueError(f"window {self.value} is outside {WINDOWS}")

    def canonical(self) -> str:
        return str(self.value)

    depth = property(lambda self: 0)
    nodes = property(lambda self: 0)
    lookback = property(lambda self: 0)


@dataclass(frozen=True)
class Call(Node):
    operator: str
    args: tuple[Node, ...]

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(f"unknown operator {self.operator}")
        if len(self.args) != ARITY[self.operator]:
            raise ValueError(f"{self.operator} expects {ARITY[self.operator]} args")
        if self.operator in ROLLING | PAIR_ROLLING:
            if not isinstance(self.args[-1], Window):
                raise ValueError(f"{self.operator} requires a window as its final argument")
        elif any(isinstance(arg, Window) for arg in self.args):
            raise ValueError("window may only be used by rolling operators")
        if not any(isinstance(node, Feature) for node in walk(self)):
            raise ValueError("an expression must contain a feature")

    def canonical(self) -> str:
        rendered = [arg.canonical() for arg in self.args]
        if self.operator in COMMUTATIVE:
            rendered.sort()
        return f"{self.operator}({','.join(rendered)})"

    @property
    def depth(self) -> int:
        return 1 + max((arg.depth for arg in self.args), default=0)

    @property
    def nodes(self) -> int:
        return 1 + sum(arg.nodes for arg in self.args)

    @property
    def lookback(self) -> int:
        children = max((arg.lookback for arg in self.args if not isinstance(arg, Window)), default=0)
        if self.operator in ROLLING | PAIR_ROLLING:
            window = self.args[-1]
            assert isinstance(window, Window)
            return children + (window.value if self.operator in {"Ref", "Delta"} else max(0, window.value - 1))
        return children


def walk(node: Node) -> Iterable[Node]:
    yield node
    if isinstance(node, Call):
        for arg in node.args:
            yield from walk(arg)


def validate_limits(node: Node, max_depth: int = 6, max_nodes: int = 21, max_lookback: int = 252) -> None:
    if node.depth > max_depth:
        raise ValueError(f"depth {node.depth} exceeds {max_depth}")
    if node.nodes > max_nodes:
        raise ValueError(f"nodes {node.nodes} exceeds {max_nodes}")
    if node.lookback > max_lookback:
        raise ValueError(f"lookback {node.lookback} exceeds {max_lookback}")

