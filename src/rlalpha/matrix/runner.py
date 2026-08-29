from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from filelock import FileLock, Timeout

from ..config import load_paths, load_yaml
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.experiment_log import append_event, update_progress
from ..manifest import git_info


GPU_THRESHOLDS_MIB = {2: 34 * 1024, 3: 28 * 1024, 4: 14 * 1024}
GPU_MEMORY_UTILIZATION = {2: "0.18", 3: "0.15", 4: "0.18"}


@lru_cache(maxsize=16)
def _repository_identity(path: str) -> dict[str, object]:
    record = git_info(path)
    return {key: record.get(key) for key in ("commit", "dirty", "dirty_patch_hash")}


@lru_cache(maxsize=16)
def _cached_file_identity(path: str, size: int, mtime_ns: int) -> dict[str, object]:
    record = file_fingerprint(Path(path))
    return {key: record[key] for key in ("path", "size", "sha256")}


def _gpu_free_mib() -> dict[int, int]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode:
        return {}
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines() if "," in line}


def _gpu_for(method: str, reward: str, seed: int, experiment: dict[str, Any] | None = None) -> int | None:
    candidates = _gpu_candidates(method, reward, seed, experiment)
    return candidates[0] if candidates else None


def _gpu_candidates(
    method: str,
    reward: str,
    seed: int,
    experiment: dict[str, Any] | None = None,
) -> list[int]:
    configured = (experiment or {}).get("gpu_devices", {}).get(method)
    if configured:
        offset = {"r0": 0, "r1": 1, "r2_lcb": 2}.get(reward, 0)
        start = (seed + offset) % len(configured)
        return [int(configured[(start + index) % len(configured)]) for index in range(len(configured))]
    if method == "base_llm":
        return [4]
    if method == "grpo_llm":
        devices = [2, 3]
        start = (seed + {"r0": 0, "r1": 1, "r2_lcb": 2}[reward]) % len(devices)
        return devices[start:] + devices[:start]
    return []


def _cell_dir(root: Path, method: str, reward: str, seed: int) -> Path:
    return root / method / reward / f"seed_{seed}"


def _cell_progress(root: Path, method: str, reward: str, seed: int) -> Path:
    return _cell_dir(root, method, reward, seed) / "progress.json"


def _matrix_progress(root: Path, cells: list[tuple[str, str, int]]) -> dict[str, Any]:
    states = {}
    for method, reward, seed in cells:
        path = _cell_progress(root, method, reward, int(seed))
        states[f"{method}/{reward}/seed_{seed}"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "pending"}
    progress_path = root / "progress.json"
    prior = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    merged = dict(prior.get("cells") or {})
    merged.update(states)
    update_progress(progress_path, experiment_id=root.name, cells=merged)
    return states


def _contains_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered or "cublas_status_alloc_failed" in lowered


def _expected_cell_identity(config: Path, paths: Any, method: str, reward: str, seed: int, steps: int) -> str:
    referenced = [
        config,
        Path(paths.code_root) / f"configs/search/{method}.yaml",
        Path(paths.code_root) / f"configs/reward/{reward}.yaml",
        Path(paths.code_root) / "configs/data/sp500.yaml",
        Path(paths.code_root) / "configs/eval/preliminary.yaml",
    ]
    if method in {"base_llm", "grpo_llm"}:
        referenced.append(Path(paths.code_root) / "configs/model/qwen3_5_2b.yaml")
    panel = Path(paths.processed_root) / "panel"
    referenced.extend(panel / name for name in ("build_manifest.yaml", "risk_build_manifest.yaml", "index.json"))
    records = []
    for path in referenced:
        if path.exists():
            record = file_fingerprint(path)
            records.append({"path": str(path.resolve()), "size": record["size"], "sha256": record["sha256"]})
        else:
            records.append({"path": str(path.resolve()), "missing": True})
    repositories = {
        "ours": _repository_identity(str(Path(paths.code_root).resolve())),
        "alphagen": _repository_identity(str(Path(paths.alphagen_root).resolve())),
        "quantevolver": _repository_identity(str(Path(paths.quantevolver_root).resolve())),
    }
    model_runtime = []
    if method in {"base_llm", "grpo_llm"}:
        model_config = load_yaml(Path(paths.code_root) / "configs/model/qwen3_5_2b.yaml")["model"]
        model_path = Path(model_config["path"])
        candidates = [model_path / "config.json", model_path / "tokenizer.json", *sorted(model_path.glob("*.safetensors"))]
        for candidate in candidates:
            if candidate.exists():
                stat = candidate.stat()
                if candidate.suffix == ".safetensors":
                    model_runtime.append({"path": str(candidate.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "declared_sha256": model_config["fingerprint"]["weights_sha256"]})
                else:
                    model_runtime.append(_cached_file_identity(str(candidate.resolve()), stat.st_size, stat.st_mtime_ns))
    return stable_hash({"schema_version": 3, "method": method, "reward": reward, "seed": seed, "search_steps": steps, "inputs": records, "repositories": repositories, "model_runtime": model_runtime})


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _cell_acceptance(directory: Path, steps: int) -> tuple[bool, str | None]:
    metrics_path = directory / "train_metrics.json"
    manifest_path = directory / "manifest.yaml"
    final_pool_path = directory / "final_pool.json"
    if not metrics_path.exists() or not manifest_path.exists() or not final_pool_path.exists():
        return False, "required cell artifacts are missing"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid train metrics: {exc}"
    if int(metrics.get("completed_steps", -1)) < steps:
        return False, "fixed search-step budget was not met"
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return False, f"invalid manifest: {exc}"
    if int(manifest.get("completed_steps", -1)) < steps or int(manifest.get("search_steps", -1)) != steps:
        return False, "manifest does not record the completed fixed search-step budget"
    if not manifest.get("manifest_hash") or not manifest.get("panel_artifacts") or not manifest.get("evaluator_version"):
        return False, "manifest is missing required identity fields"
    return True, None


def _run_matrix_unlocked(config: str | Path, experiment_id: str, resume: bool = True, poll_seconds: int = 30, methods: list[str] | None = None, rewards: list[str] | None = None) -> dict[str, Any]:
    config = Path(config).resolve()
    raw = load_yaml(config)
    experiment = raw["experiment"]
    max_cpu_jobs = max(1, int(experiment.get("max_cpu_jobs", 4)))
    paths = load_paths(config)
    root = paths.runs_root / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    if "cells" in experiment:
        cells = [(method, reward, seed) for (method, reward), seed in itertools.product(experiment["cells"], experiment["seeds"])]
    else:
        cells = list(itertools.product(experiment["methods"], experiment["rewards"], experiment["seeds"]))
    selected_methods = set(methods or ())
    if selected_methods:
        unknown = selected_methods - {str(cell[0]) for cell in cells}
        if unknown:
            raise ValueError(f"methods are not configured for this experiment: {sorted(unknown)}")
        cells = [cell for cell in cells if cell[0] in selected_methods]
    selected_rewards = set(rewards or ())
    if selected_rewards:
        configured_rewards = {str(cell[1]) for cell in cells}
        unknown = selected_rewards - configured_rewards
        if unknown:
            raise ValueError(f"rewards are not configured for this experiment: {sorted(unknown)}")
        cells = [cell for cell in cells if cell[1] in selected_rewards]
    if not resume:
        occupied = [str(_cell_dir(root, method, reward, int(seed))) for method, reward, seed in cells if _cell_dir(root, method, reward, int(seed)).exists() and any(_cell_dir(root, method, reward, int(seed)).iterdir())]
        if occupied:
            raise RuntimeError(f"--no-resume refuses existing cell directories: {occupied}; use a new experiment ID")
    if any(method in {"base_llm", "grpo_llm"} for method, _, _ in cells) and not bool(experiment.get("auto_start_expensive_jobs", False)):
        raise RuntimeError("expensive Base-LLM/GRPO cells are disabled by experiment.auto_start_expensive_jobs=false")
    steps = int(experiment.get("search_steps", 250))
    append_event(root / "experiment.log", "matrix_started", experiment_id=experiment_id, cells=len(cells), search_steps=steps, candidates_per_step=int(experiment.get("proposal_group_size", 8)))
    gpu_thresholds = {
        **GPU_THRESHOLDS_MIB,
        **{int(device): int(value) for device, value in experiment.get("gpu_min_free_mib", {}).items()},
    }
    gpu_memory_utilization = {
        **GPU_MEMORY_UTILIZATION,
        **{int(device): str(value) for device, value in experiment.get("gpu_memory_utilization", {}).items()},
    }
    pending: list[tuple[str, str, int, int, int]] = []
    external: dict[tuple[str, str, int], dict[str, Any]] = {}
    for method, reward, seed in cells:
        cell = _cell_dir(root, method, reward, int(seed))
        state_path = cell / "progress.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {}
        identity = _expected_cell_identity(config, paths, method, reward, int(seed), steps)
        if state.get("status") == "complete" and state.get("cell_identity") == identity and int(state.get("search_steps", -1)) == steps:
            continue
        if state.get("status") == "running" and _pid_alive(state.get("pid")):
            external[(method, reward, int(seed))] = state
            continue
        pending.append((method, reward, int(seed), int(state.get("attempt", 0)) + 1, int(state.get("rollout_microbatch", 8))))
    running: dict[subprocess.Popen[str], tuple[str, str, int, float, Any, int | None, int, int, int]] = {}
    while pending or running or external:
        free = _gpu_free_mib()
        busy_gpus = {item[5] for item in running.values()} | {item.get("gpu") for item in external.values()}
        cpu_jobs = sum(item[5] is None for item in running.values()) + sum(item.get("gpu") is None for item in external.values())
        total_cpu_threads = sum(
            int(experiment.get("grpo_ray_cpus", 8)) if item[0] == "grpo_llm" else int(experiment.get("cpu_threads_per_job", 8))
            for item in running.values()
        ) + sum(
            int(experiment.get("grpo_ray_cpus", 8)) if cell[0] == "grpo_llm" else int(experiment.get("cpu_threads_per_job", 8))
            for cell in external
        )
        base_busy = any(item[0] == "base_llm" for item in running.values()) or any(
            cell[0] == "base_llm" for cell in external
        )
        for cell in list(pending):
            method, reward, seed, attempt, microbatch = cell
            gpu_options = _gpu_candidates(method, reward, seed, experiment)
            gpu = next(
                (
                    device for device in gpu_options
                    if device not in busy_gpus
                    and free.get(device, 0) >= gpu_thresholds.get(device, 32 * 1024)
                ),
                gpu_options[0] if gpu_options else None,
            )
            required_threads = int(experiment.get("grpo_ray_cpus", 8)) if method == "grpo_llm" else int(experiment.get("cpu_threads_per_job", 8))
            if total_cpu_threads + required_threads > int(experiment.get("max_total_cpu_threads", 90)):
                continue
            if gpu is None and cpu_jobs >= max_cpu_jobs:
                continue
            if gpu is not None and (gpu in busy_gpus or free.get(gpu, 0) < gpu_thresholds.get(gpu, 32 * 1024)):
                continue
            if method == "base_llm" and base_busy:
                continue
            directory = _cell_dir(root, method, reward, seed)
            directory.mkdir(parents=True, exist_ok=True)
            state_path = directory / "progress.json"
            cell_identity = _expected_cell_identity(config, paths, method, reward, seed, steps)
            update_progress(state_path, status="running", method=method, reward=reward, seed=seed, search_steps=steps, cell_identity=cell_identity, gpu=gpu, attempt=attempt, rollout_microbatch=microbatch, started_at=time.time(), gpu_free_mib=free.get(gpu) if gpu is not None else None)
            append_event(directory / "experiment.log", "cell_started", method=method, reward=reward, seed=seed, search_steps=steps, gpu=gpu, attempt=attempt)
            append_event(root / "experiment.log", "cell_started", cell=f"{method}/{reward}/seed_{seed}", gpu=gpu, attempt=attempt)
            runtime_dir = directory / "details"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            log_handle = (runtime_dir / "runtime.log").open("a", encoding="utf-8")
            log_offset = log_handle.tell()
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["RLALPHA_PHYSICAL_GPU"] = str(gpu)
                env["RLALPHA_VLLM_MEMORY_UTILIZATION"] = gpu_memory_utilization.get(gpu, "0.18")
            thread_count = int(experiment.get("grpo_ray_cpus", 8)) if method == "grpo_llm" else int(experiment.get("cpu_threads_per_job", 8))
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
                env[variable] = str(thread_count)
            if method == "grpo_llm":
                env["RLALPHA_GRPO_MICROBATCH"] = str(microbatch)
            command = [sys.executable, "-m", "rlalpha.cli", "search", "run", "--method", method, "--reward", reward, "--seed", str(seed), "--steps", str(steps), "--experiment-id", experiment_id, "--config", str(config)]
            process = subprocess.Popen(command, cwd=paths.code_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            update_progress(state_path, status="running", method=method, reward=reward, seed=seed, search_steps=steps, cell_identity=cell_identity, gpu=gpu, pid=process.pid, attempt=attempt, rollout_microbatch=microbatch, started_at=time.time(), gpu_free_mib=free.get(gpu) if gpu is not None else None)
            running[process] = (method, reward, seed, time.time(), log_handle, gpu, attempt, microbatch, log_offset)
            pending.remove(cell)
            busy_gpus.add(gpu)
            cpu_jobs += gpu is None
            base_busy = base_busy or method == "base_llm"
            total_cpu_threads += required_threads
            _matrix_progress(root, [(str(a), str(b), int(c)) for a, b, c in cells])
        if not running and (pending or external):
            time.sleep(poll_seconds)
        for process, details in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            method, reward, seed, started, log_handle, gpu, attempt, microbatch, log_offset = details
            log_handle.close()
            log_path = _cell_dir(root, method, reward, seed) / "details/runtime.log"
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_offset)
                attempt_log = handle.read()
            state_path = _cell_dir(root, method, reward, seed) / "progress.json"
            oom_retry = returncode != 0 and method == "grpo_llm" and _contains_cuda_oom(attempt_log) and microbatch > 1
            accepted, acceptance_error = _cell_acceptance(_cell_dir(root, method, reward, seed), steps) if returncode == 0 else (False, None)
            status = "retrying_after_oom" if oom_retry else ("complete" if returncode == 0 and accepted else "failed")
            next_microbatch = max(1, microbatch // 2) if oom_retry else microbatch
            cell_identity = _expected_cell_identity(config, paths, method, reward, seed, steps)
            update_progress(state_path, status=status, method=method, reward=reward, seed=seed, search_steps=steps, cell_identity=cell_identity, gpu=gpu, returncode=returncode, attempt=attempt, rollout_microbatch=next_microbatch, started_at=started, finished_at=time.time(), wall_seconds=time.time() - started, acceptance_error=acceptance_error, error_tail=attempt_log[-4000:] if returncode else None)
            append_event(_cell_dir(root, method, reward, seed) / "experiment.log", "cell_finished", status=status, returncode=returncode, wall_seconds=round(time.time() - started, 3))
            append_event(root / "experiment.log", "cell_finished", cell=f"{method}/{reward}/seed_{seed}", status=status, returncode=returncode)
            if oom_retry:
                pending.append((method, reward, seed, attempt + 1, next_microbatch))
            del running[process]
            _matrix_progress(root, [(str(a), str(b), int(c)) for a, b, c in cells])
        for cell, state in list(external.items()):
            if _pid_alive(state.get("pid")):
                continue
            method, reward, seed = cell
            directory = _cell_dir(root, method, reward, seed)
            metrics_path = directory / "train_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            expected_identity = _expected_cell_identity(config, paths, method, reward, seed, steps)
            if int(metrics.get("completed_steps", -1)) >= steps and state.get("cell_identity") == expected_identity:
                update_progress(directory / "progress.json", **{**state, "status": "complete", "recovered_after_runner_restart": True, "finished_at": time.time()})
            else:
                pending.append((method, reward, seed, int(state.get("attempt", 0)) + 1, int(state.get("rollout_microbatch", 8))))
            del external[cell]
        if running:
            time.sleep(min(poll_seconds, 30))
    states = {}
    for method, reward, seed in cells:
        path = _cell_dir(root, method, reward, int(seed)) / "progress.json"
        states[f"{method}/{reward}/seed_{seed}"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}
    progress_path = root / "progress.json"
    prior = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    merged_states = dict(prior.get("cells") or {})
    merged_states.update(states)
    update_progress(progress_path, experiment_id=experiment_id, cells=merged_states)
    append_event(root / "experiment.log", "matrix_finished", completed=sum(state.get("status") == "complete" for state in states.values()), failed=sum(state.get("status") == "failed" for state in states.values()))
    return {"experiment_id": experiment_id, "cells": states}


def run_matrix(config: str | Path, experiment_id: str, resume: bool = True, poll_seconds: int = 30, methods: list[str] | None = None, rewards: list[str] | None = None) -> dict[str, Any]:
    paths = load_paths(config)
    root = paths.runs_root / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    lock_root = Path(tempfile.gettempdir()) / "rlalpha-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_root / f"matrix-{stable_hash(str(root.resolve()))}.lock"))
    try:
        with lock.acquire(timeout=0):
            return _run_matrix_unlocked(config, experiment_id, resume, poll_seconds, methods, rewards)
    except Timeout as exc:
        raise RuntimeError(f"another matrix runner owns {lock.lock_file}") from exc
