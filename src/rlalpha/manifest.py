from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .utils.hashing import file_fingerprint, stable_hash


def git_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not (path / ".git").exists():
        return {"path": str(path), "commit": None, "dirty": None}
    commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(path), "status", "--short"], check=False, capture_output=True, text=True).stdout
    return {"path": str(path.resolve()), "commit": commit or None, "dirty": bool(status.strip()), "status": status.splitlines()}


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(paths: Any, data_files: list[Path] | None = None) -> dict[str, Any]:
    data_records = [file_fingerprint(path) for path in (data_files or [])]
    manifest = {
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": package_versions(["numpy", "pandas", "pyarrow", "scipy", "statsmodels", "torch", "transformers", "vllm", "verl"]),
        "repositories": {
            "ours": git_info(paths.code_root.parent),
            "alphagen": git_info(paths.alphagen_root),
            "quantevolver": git_info(paths.quantevolver_root),
        },
        "data_files": data_records,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest

