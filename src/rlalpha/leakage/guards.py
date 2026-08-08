from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..utils.hashing import stable_hash

FORBIDDEN_CONTEXT_TERMS = ("validation_ic", "validation_metric", "test_ic", "test_metric", "test_return")


def assert_train_only_context(context: dict[str, Any]) -> None:
    serialized = str(context).lower()
    found = [term for term in FORBIDDEN_CONTEXT_TERMS if term in serialized]
    if found:
        raise ValueError(f"search context contains forbidden non-train metrics: {found}")


@dataclass
class ReadOnlyStateGuard:
    snapshot: Callable[[], Any]

    def __enter__(self) -> "ReadOnlyStateGuard":
        self.before = stable_hash(self.snapshot())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        after = stable_hash(self.snapshot())
        if exc_type is None and after != self.before:
            raise RuntimeError("read-only evaluation mutated protected state")

