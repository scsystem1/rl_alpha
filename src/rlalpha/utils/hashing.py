from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def files_fingerprint(paths: Iterable[str | Path]) -> str:
    records = [file_fingerprint(path) for path in sorted(map(Path, paths))]
    return stable_hash(records)


def directory_fingerprint(path: str | Path) -> dict[str, Any]:
    """Content fingerprint for an immutable directory tree.

    Modification times and absolute paths are intentionally excluded so an
    atomically renamed checkpoint keeps the same identity.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        fingerprint = file_fingerprint(item)
        files.append({"path": item.relative_to(root).as_posix(), "size": fingerprint["size"], "sha256": fingerprint["sha256"]})
    if not files:
        raise RuntimeError(f"cannot fingerprint an empty directory: {root}")
    return {"path": str(root.resolve()), "file_count": len(files), "total_size": sum(item["size"] for item in files), "sha256": stable_hash(files)}
