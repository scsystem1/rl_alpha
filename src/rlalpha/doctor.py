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
    manifest = build_manifest(paths, data_files)
    required = {"torch": "2.11.0", "transformers": "5.10.4", "vllm": "0.26.0", "ray": "2.56.1", "peft": "0.20.0"}
    dependency_ok = all((manifest["packages"].get(name) or "").startswith(version) for name, version in required.items()) and manifest["packages"].get("verl") is not None
    model_unique = len(set(model_candidates)) == 1
    solver_ok = {"OSQP", "CLARABEL"}.issubset(solvers)
    data_ok = bool(discovery.get("daily") and discovery.get("membership"))
    gpu_ok = bool(nvidia and nvidia.returncode == 0)
    return {
        "ok": data_ok and dependency_ok and model_unique and solver_ok and gpu_ok,
        "checks": {"data": data_ok, "dependencies": dependency_ok, "model": model_unique, "solvers": solver_ok, "gpu": gpu_ok},
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
        "qwen3_5_2b": {"resolved": model_candidates[0] if model_unique else None, "candidates": sorted(set(model_candidates)), "unique": model_unique, "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"},
        "solvers": solvers,
        "manifest": manifest,
    }
