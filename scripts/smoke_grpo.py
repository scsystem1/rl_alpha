from __future__ import annotations

"""Real QuantEvolver/Verl optimizer-update and resume smoke on a small panel."""

import argparse
import json
import math
import os
from pathlib import Path

from rlalpha.config import load_paths, load_yaml
from rlalpha.data.store import PanelStore
from rlalpha.factors.pool import PoolManager
from rlalpha.search.base_llm import resolve_model_path
from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator
from rlalpha.search.run import objective_for
from rlalpha.utils.hashing import file_fingerprint, stable_hash
from rlalpha.utils.io import write_json, write_yaml


def _actor_model_hash(checkpoint: Path) -> str:
    actor = checkpoint / "actor"
    candidates = [
        path
        for path in actor.rglob("*")
        if path.is_file() and "optim" not in path.name.lower() and "extra" not in path.name.lower()
    ]
    if not candidates:
        raise RuntimeError(f"no actor model files found under {actor}")
    return stable_hash(
        [
            {
                "path": path.relative_to(actor).as_posix(),
                "sha256": file_fingerprint(path)["sha256"],
                "size": path.stat().st_size,
            }
            for path in sorted(candidates)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/revision_v3_full_smoke.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs_v3/grpo_optimizer_smoke_cuda3"))
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--reward", choices=("r0", "r1", "r2_lcb"), default="r0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.updates < 2:
        raise SystemExit("the acceptance smoke requires at least two optimizer updates")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly (for this host use CUDA_VISIBLE_DEVICES=3)")
    config_path = args.config.resolve()
    paths = load_paths(config_path)
    method_config = load_yaml(paths.code_root / "configs/search/grpo_llm.yaml")["search"]
    model_config = load_yaml(paths.code_root / "configs/model/qwen3_5_2b.yaml")
    reward_config = load_yaml(paths.code_root / f"configs/reward/{args.reward}.yaml")
    effective = {
        "paths": paths.model_dump(mode="json"),
        "search": method_config,
        **model_config,
        **reward_config,
        "invocation": {
            "experiment_id": "grpo_optimizer_smoke_cuda3",
            "method": "grpo_llm",
            "reward": args.reward,
            "seed": args.seed,
            "budget": args.updates * int(model_config["rollout"]["n"]),
        },
        "smoke_panel": {"train_start": args.train_start, "train_end": args.train_end},
    }
    if int(effective["actor"]["ppo_epochs"]) < 2:
        raise RuntimeError("acceptance smoke requires at least two PPO epochs so ratio/clipping can become active")
    effective["model"]["path"] = str(resolve_model_path(effective))
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "checkpoint.json"
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing smoke run {run_dir}; pass --resume or choose a new --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "effective_config.yaml", effective)
    panel = PanelStore(paths.processed_root).load_split("train", start=args.train_start, end=args.train_end)
    pool = PoolManager(objective_for(args.reward, panel), capacity=20)
    coordinator = VerlGRPOStageCoordinator(
        pool,
        panel.evaluate,
        panel.target(panel.common_mask),
        effective["invocation"]["budget"],
        run_dir,
        effective,
        paths.quantevolver_root,
        paths.processed_root,
        args.reward,
        args.seed,
        train_start=args.train_start,
        train_end=args.train_end,
    )
    if args.resume and checkpoint_path.exists():
        coordinator.load_checkpoint()
    completed = []
    while coordinator.updates < args.updates:
        completed.append(coordinator.run_stage())
    checkpoints = sorted((run_dir / "checkpoints/verl_committed").glob("*/global_step_*"))
    if len(checkpoints) < 2:
        raise RuntimeError("smoke did not create two committed Verl checkpoints")
    latest = checkpoints[-args.updates :]
    model_hashes = [_actor_model_hash(path) for path in latest]
    if len(set(model_hashes[-2:])) != 2:
        raise RuntimeError("actor model parameters did not change between the last two optimizer updates")
    metric_paths = [Path(event["metrics_path"]) for event in coordinator.events if event["kind"] == "stage_end"]
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths[-args.updates :]]
    if not all(math.isfinite(float(item["actor/grad_norm"])) and float(item["actor/grad_norm"]) > 0 for item in metrics):
        raise RuntimeError("at least one claimed optimizer update has a non-positive/non-finite gradient norm")
    required_finite = (
        "actor/pg_loss",
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/kl_loss",
        "actor/entropy",
        "actor/lr",
        "critic/advantages/mean",
        "domain/advantage_std",
    )
    if not all(all(math.isfinite(float(item[key])) for key in required_finite) for item in metrics):
        raise RuntimeError("formal-GRPO smoke emitted a missing/non-finite loss metric")
    if not any(float(item["actor/pg_clipfrac"]) > 0 for item in metrics):
        raise RuntimeError("PPO clipping never activated; check ppo_epochs and old-policy log probabilities")
    if not any(abs(float(item["actor/ppo_kl"])) > 1e-9 for item in metrics):
        raise RuntimeError("old/current-policy importance ratio stayed identically one")
    if not any(float(item["actor/kl_loss"]) > 0 for item in metrics):
        raise RuntimeError("reference-policy KL loss never activated")
    for checkpoint in latest:
        actor_files = [path.name for path in (checkpoint / "actor").iterdir() if path.is_file()]
        if not any("optim" in name for name in actor_files) or not any("extra_state" in name for name in actor_files):
            raise RuntimeError(f"checkpoint lacks optimizer/scheduler/RNG state: {checkpoint}")
    report = {
        "status": "passed",
        "physical_gpu": os.getenv("RLALPHA_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "updates": coordinator.updates,
        "stage": coordinator.stage,
        "valid_unique_evaluations": coordinator.ledger.valid_unique_evaluations,
        "raw_proposals": coordinator.ledger.raw_proposals,
        "checkpoint_paths": [str(path) for path in latest],
        "actor_model_hashes": model_hashes,
        "resume_checkpoint_used": str(latest[-2]),
        "required_loss_metrics": [
            {
                key: item[key]
                for key in (
                    "training/global_step",
                    "actor/pg_loss",
                    "actor/pg_clipfrac",
                    "actor/ppo_kl",
                    "actor/kl_loss",
                    "actor/entropy",
                    "actor/grad_norm",
                    "actor/lr",
                    "critic/advantages/mean",
                    "domain/advantage_std",
                    "domain/invalid_rate",
                    "domain/unique_rate",
                    "domain/reward_mean",
                    "domain/reward_std",
                )
            }
            for item in metrics
        ],
    }
    write_json(run_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
