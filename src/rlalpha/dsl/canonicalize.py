from __future__ import annotations

from .ast import Node


def canonicalize(node: Node) -> tuple[str, str]:
    return node.canonical(), node.expr_hash

