from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...config import load_paths, load_yaml


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("episode_*.pt"):
        try:
            episode = int(path.stem.removeprefix("episode_"))
        except ValueError:
            continue
        candidates.append((episode, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def run_alphasage(
    config_path: str | Path,
    reward: str,
    seed: int,
    steps: int,
    experiment_id: str,
    resume: bool = True,
) -> dict[str, Any]:
    """Run AlphaSAGE in its own checkout while preserving its native reward.

    One RLAlpha accounting step represents eight AlphaSAGE proposal slots. The
    fixed fairness budget counts raw proposals, including invalid and duplicate
    proposals; it does not require 800 candidates to pass validation.
    """
    if reward != "r0":
        raise ValueError(f"AlphaSAGE baseline requires reward=r0, got {reward}")
    raw = load_yaml(config_path)
    if raw.get("protocol") != "recent_alpha_v1":
        raise ValueError("the matrix AlphaSAGE adapter requires a resolved recent_alpha_v1 window")
    experiment = raw["experiment"]
    group_size = int(experiment.get("proposal_group_size", 8))
    if group_size != 8:
        raise ValueError(f"AlphaSAGE fairness protocol requires 8 candidates per step, got {group_size}")
    paths = load_paths(config_path)
    alphasage_root = Path(os.getenv("RLALPHA_ALPHASAGE_ROOT", Path(paths.code_root).parent / "baseline/AlphaSAGE"))
    trainer = alphasage_root / "train_rlalpha.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"AlphaSAGE RLAlpha trainer is missing: {trainer}")
    run_dir = paths.runs_root / experiment_id / "alphasage" / reward / f"seed_{seed}"
    checkpoint = _latest_checkpoint(run_dir) if resume else None
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"--no-resume refuses existing AlphaSAGE artifacts in {run_dir}")

    search = load_yaml(paths.code_root / "configs/search/alphasage.yaml")["search"]
    factor_budget = int(steps) * group_size
    command = [
        sys.executable,
        str(trainer),
        "--config", str(Path(config_path).resolve()),
        "--experiment-id", experiment_id,
        "--seed", str(seed),
        "--episodes", str(factor_budget),
        "--raw-proposal-budget", str(factor_budget),
        "--search-steps", str(steps),
        "--proposal-group-size", str(group_size),
        "--pool-capacity", str(int(experiment.get("pool_capacity", 20))),
        "--update-frequency", "64",
        "--log-frequency", "64",
        "--hidden-dim", str(int(search.get("hidden_dim", 128))),
        "--learning-rate", str(float(search.get("learning_rate", 1e-4))),
        "--entropy-coef", str(float(search.get("entropy_coef", 0.01))),
        "--entropy-temperature", str(float(search.get("entropy_temperature", 1.0))),
        "--mask-dropout-prob", str(float(search.get("mask_dropout_prob", 1.0))),
        "--ssl-weight", str(float(search.get("ssl_weight", 1.0))),
        "--novelty-weight", str(float(search.get("novelty_weight", 0.3))),
        "--mutual-ic-threshold", str(float(search.get("mutual_ic_threshold", 0.3))),
        "--final-weight-ratio", str(float(search.get("final_weight_ratio", 0.0))),
        "--max-expression-tokens", str(int(search.get("max_expression_tokens", 20))),
        "--device", "cuda:0",
    ]
    if checkpoint is not None:
        command.extend(["--resume-from", str(checkpoint)])
    completed = subprocess.run(command, cwd=alphasage_root, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"AlphaSAGE trainer exited with status {completed.returncode}")
    metrics_path = run_dir / "train_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"AlphaSAGE trainer did not write {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        int(metrics.get("completed_steps", -1)) != int(steps)
        or int(metrics.get("candidates_per_step", -1)) != group_size
        or int(metrics.get("factor_budget", -1)) != factor_budget
        or int(metrics.get("raw_proposals", -1)) != factor_budget
    ):
        raise RuntimeError(
            f"AlphaSAGE trainer did not complete the fixed {steps}x{group_size} accounting budget"
        )
    return {"run_dir": str(run_dir), **metrics}
