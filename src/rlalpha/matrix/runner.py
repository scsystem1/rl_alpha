from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from filelock import FileLock, Timeout

from ..config import load_paths, load_yaml
from ..utils.io import write_json


GPU_THRESHOLDS_MIB = {2: 34 * 1024, 3: 28 * 1024, 4: 14 * 1024}
GPU_MEMORY_UTILIZATION = {2: "0.18", 3: "0.15", 4: "0.18"}


def _gpu_free_mib() -> dict[int, int]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode:
        return {}
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines() if "," in line}


def _gpu_for(method: str, reward: str, seed: int) -> int | None:
    if method == "base_llm":
        return 4
    if method == "grpo_llm":
        return (2, 3)[(seed + {"r0": 0, "r1": 1, "r2_lcb": 2}[reward]) % 2]
    return None


def _cell_dir(root: Path, method: str, reward: str, seed: int) -> Path:
    return root / method / reward / f"seed_{seed}"


def _contains_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered or "cublas_status_alloc_failed" in lowered


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _run_matrix_unlocked(config: str | Path, experiment_id: str, resume: bool = True, poll_seconds: int = 30) -> dict[str, Any]:
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
    budget = int(experiment["valid_unique_budget"])
    pending: list[tuple[str, str, int, int, int]] = []
    external: dict[tuple[str, str, int], dict[str, Any]] = {}
    for method, reward, seed in cells:
        cell = _cell_dir(root, method, reward, int(seed))
        state_path = cell / "cell_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {}
        if state.get("status") == "complete":
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
        base_busy = 4 in busy_gpus
        for cell in list(pending):
            method, reward, seed, attempt, microbatch = cell
            gpu = _gpu_for(method, reward, seed)
            if gpu is None and cpu_jobs >= max_cpu_jobs:
                continue
            if gpu is not None and (gpu in busy_gpus or free.get(gpu, 0) < GPU_THRESHOLDS_MIB[gpu]):
                continue
            if method == "base_llm" and base_busy:
                continue
            directory = _cell_dir(root, method, reward, seed)
            (directory / "logs").mkdir(parents=True, exist_ok=True)
            state_path = directory / "cell_state.json"
            write_json(state_path, {"status": "running", "method": method, "reward": reward, "seed": seed, "budget": budget, "gpu": gpu, "attempt": attempt, "rollout_microbatch": microbatch, "started_at": time.time(), "gpu_free_mib": free.get(gpu) if gpu is not None else None})
            log_handle = (directory / "logs/search.log").open("a", encoding="utf-8")
            log_offset = log_handle.tell()
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["RLALPHA_PHYSICAL_GPU"] = str(gpu)
                env["RLALPHA_VLLM_MEMORY_UTILIZATION"] = GPU_MEMORY_UTILIZATION[gpu]
            if method == "grpo_llm":
                env["RLALPHA_GRPO_MICROBATCH"] = str(microbatch)
            if gpu is None:
                for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
                    env[variable] = str(int(experiment.get("cpu_threads_per_job", 8)))
            command = [sys.executable, "-m", "rlalpha.cli", "search", "run", "--method", method, "--reward", reward, "--seed", str(seed), "--budget", str(budget), "--experiment-id", experiment_id, "--config", str(config)]
            process = subprocess.Popen(command, cwd=paths.code_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            write_json(state_path, {"status": "running", "method": method, "reward": reward, "seed": seed, "budget": budget, "gpu": gpu, "pid": process.pid, "attempt": attempt, "rollout_microbatch": microbatch, "started_at": time.time(), "gpu_free_mib": free.get(gpu) if gpu is not None else None})
            running[process] = (method, reward, seed, time.time(), log_handle, gpu, attempt, microbatch, log_offset)
            pending.remove(cell)
            busy_gpus.add(gpu)
            cpu_jobs += gpu is None
            base_busy = base_busy or gpu == 4
        if not running and (pending or external):
            time.sleep(poll_seconds)
        for process, details in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            method, reward, seed, started, log_handle, gpu, attempt, microbatch, log_offset = details
            log_handle.close()
            log_path = _cell_dir(root, method, reward, seed) / "logs/search.log"
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_offset)
                attempt_log = handle.read()
            state_path = _cell_dir(root, method, reward, seed) / "cell_state.json"
            oom_retry = returncode != 0 and method == "grpo_llm" and _contains_cuda_oom(attempt_log) and microbatch > 1
            status = "retrying_after_oom" if oom_retry else ("complete" if returncode == 0 else "failed")
            next_microbatch = max(1, microbatch // 2) if oom_retry else microbatch
            write_json(state_path, {"status": status, "method": method, "reward": reward, "seed": seed, "budget": budget, "gpu": gpu, "returncode": returncode, "attempt": attempt, "rollout_microbatch": next_microbatch, "started_at": started, "finished_at": time.time(), "wall_seconds": time.time() - started, "error_tail": attempt_log[-4000:] if returncode else None})
            if oom_retry:
                pending.append((method, reward, seed, attempt + 1, next_microbatch))
            del running[process]
        for cell, state in list(external.items()):
            if _pid_alive(state.get("pid")):
                continue
            method, reward, seed = cell
            directory = _cell_dir(root, method, reward, seed)
            metrics_path = directory / "train_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            if int(metrics.get("valid_unique_evaluations", -1)) >= budget:
                write_json(directory / "cell_state.json", {**state, "status": "complete", "recovered_after_runner_restart": True, "finished_at": time.time()})
            else:
                pending.append((method, reward, seed, int(state.get("attempt", 0)) + 1, int(state.get("rollout_microbatch", 8))))
            del external[cell]
        if running:
            time.sleep(min(poll_seconds, 30))
    states = {}
    for method, reward, seed in cells:
        path = _cell_dir(root, method, reward, int(seed)) / "cell_state.json"
        states[f"{method}/{reward}/seed_{seed}"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}
    write_json(root / "matrix_state.json", states)
    return {"experiment_id": experiment_id, "cells": states}


def run_matrix(config: str | Path, experiment_id: str, resume: bool = True, poll_seconds: int = 30) -> dict[str, Any]:
    paths = load_paths(config)
    root = paths.runs_root / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / "matrix_runner.lock"))
    try:
        with lock.acquire(timeout=0):
            return _run_matrix_unlocked(config, experiment_id, resume, poll_seconds)
    except Timeout as exc:
        raise RuntimeError(f"another matrix runner owns {lock.lock_file}") from exc
