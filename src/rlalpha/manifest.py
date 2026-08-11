from __future__ import annotations

import os
import platform
import hashlib
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
    diff = subprocess.run(["git", "-C", str(path), "diff", "--binary", "HEAD"], check=False, capture_output=True).stdout
    untracked_output = subprocess.run(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"], check=False, capture_output=True).stdout
    untracked = []
    for raw in untracked_output.split(b"\0"):
        if not raw:
            continue
        candidate = path / os.fsdecode(raw)
        if candidate.is_file():
            record = file_fingerprint(candidate)
            untracked.append({key: record[key] for key in ("path", "size", "sha256")})
    dirty_patch_hash = stable_hash({"tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "untracked": untracked})
    return {"path": str(path.resolve()), "commit": commit or None, "dirty": bool(status.strip()), "status": status.splitlines(), "dirty_patch_hash": dirty_patch_hash}


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    paths: Any,
    data_files: list[Path] | None = None,
    *,
    effective_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    prompt: dict[str, Any] | None = None,
    reward_version: str | None = None,
    evaluator_version: str | None = None,
) -> dict[str, Any]:
    data_records = [file_fingerprint(path) for path in (data_files or [])]
    manifest = {
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": package_versions(["numpy", "pandas", "pyarrow", "scipy", "statsmodels", "torch", "transformers", "vllm", "verl", "ray", "peft", "cvxpy", "osqp"]),
        "repositories": {
            "ours": git_info(paths.code_root),
            "alphagen": git_info(paths.alphagen_root),
            "quantevolver": git_info(paths.quantevolver_root),
        },
        "data_files": data_records,
        "cuda": {"visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "physical_gpu": os.getenv("RLALPHA_PHYSICAL_GPU")},
    }
    panel_records = []
    for name in ("build_manifest.yaml", "risk_build_manifest.yaml", "index.json"):
        candidate = Path(paths.processed_root) / "panel" / name
        if candidate.exists():
            panel_records.append(file_fingerprint(candidate))
    manifest["panel_artifacts"] = panel_records
    manifest["effective_config"] = {"value": effective_config, "hash": stable_hash(effective_config)} if effective_config is not None else None
    manifest["prompt"] = prompt
    manifest["reward_version"] = reward_version
    manifest["evaluator_version"] = evaluator_version
    if model_config:
        model_path = Path(model_config["path"])
        candidates = {"config": model_path / "config.json", "tokenizer": model_path / "tokenizer.json"}
        weights = sorted(model_path.glob("*.safetensors"))
        if len(weights) == 1:
            candidates["weights"] = weights[0]
        actual = {name: file_fingerprint(path) for name, path in candidates.items() if path.exists()}
        manifest["model_runtime"] = {"path": str(model_path.resolve()), "revision": model_config.get("revision"), "declared": model_config.get("fingerprint"), "files": actual, "runtime_hash": stable_hash(actual)}
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest
