from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import PathsConfig
from .data.discovery import discover_data_files
from .manifest import build_manifest


def _path_check(path: Path, require_write: bool = False) -> dict[str, Any]:
    exists = path.exists()
    result = {"path": str(path), "exists": exists, "is_dir": path.is_dir() if exists else False}
    if require_write:
        parent = path if exists else path.parent
        result["writable"] = parent.exists() and os_access_write(parent)
    return result


def os_access_write(path: Path) -> bool:
    import os

    return os.access(path, os.W_OK)


def run_doctor(paths: PathsConfig) -> dict[str, Any]:
    discovery = discover_data_files(paths.raw_data_root, strict=False)
    nvidia = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    solvers: list[str] = []
    try:
        import cvxpy as cp

        solvers = cp.installed_solvers()
    except ImportError:
        pass
    model_candidates = []
    if paths.model_search_root.exists():
        for config in paths.model_search_root.glob("**/config.json"):
            lowered = str(config.parent).lower()
            if "qwen3.5" in lowered and "2b" in lowered:
                names = {item.name for item in config.parent.iterdir() if item.is_file()}
                if ("tokenizer.json" in names or "tokenizer_config.json" in names) and any(name.endswith(".safetensors") for name in names):
                    model_candidates.append(str(config.parent.resolve()))
    data_files = list(discovery.values())
    return {
        "ok": bool(discovery.get("daily") and discovery.get("membership")),
        "paths": {
            "code_root": _path_check(paths.code_root),
            "raw_data_root": _path_check(paths.raw_data_root),
            "processed_root": _path_check(paths.processed_root, require_write=True),
            "cache_root": _path_check(paths.cache_root, require_write=True),
            "runs_root": _path_check(paths.runs_root, require_write=True),
            "model_search_root": _path_check(paths.model_search_root),
        },
        "data_discovery": {key: str(value) for key, value in discovery.items()},
        "gpu": {"available": bool(nvidia and nvidia.returncode == 0), "message": (nvidia.stdout or nvidia.stderr).strip() if nvidia else "nvidia-smi not installed"},
        "qwen3_5_2b": {"resolved": model_candidates[0] if len(model_candidates) == 1 else None, "candidates": sorted(set(model_candidates)), "unique": len(set(model_candidates)) == 1},
        "solvers": solvers,
        "manifest": build_manifest(paths, data_files),
    }
