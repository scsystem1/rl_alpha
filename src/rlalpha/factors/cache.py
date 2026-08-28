from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..utils.io import atomic_write_bytes


class SignalCache:
    def __init__(self, root: str | Path | None = None, max_items: int = 64):
        self.root = None if root is None else Path(root)
        self.max_items = max_items
        self._memory: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        path = None if self.root is None else self.root / f"{key}.npy"
        if path is not None and path.exists():
            value = np.load(path, mmap_mode="r")
            self.put(key, value)
            return value
        return None

    def put(self, key: str, value: np.ndarray, permanent: bool = False) -> None:
        self._memory[key] = np.asarray(value)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_items:
            self._memory.popitem(last=False)
        path = None if self.root is None else self.root / f"{key}.npy"
        if permanent and path is not None and not path.exists():
            import io

            buffer = io.BytesIO()
            # Preserve the evaluator dtype exactly.  Down-casting only on the
            # persisted/resume path makes fresh and resumed experiments differ.
            np.save(buffer, np.asarray(value), allow_pickle=False)
            atomic_write_bytes(path, buffer.getvalue())

    def discard(self, key: str) -> None:
        self._memory.pop(key, None)

    def prune(self, retained: set[str]) -> None:
        """Remove signals that cannot participate in resume or the final pool."""
        for key in list(self._memory):
            if key not in retained:
                self._memory.pop(key, None)
        if self.root is None or not self.root.exists():
            return
        for path in self.root.glob("*.npy"):
            if path.stem not in retained:
                path.unlink()
