from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any


def _load_verl_base(quantevolver_root: Path):
    """Load the config matching the installed Verl trainer API.

    QuantEvolver's checked-in base config targets the pre reward-loop API.  The
    currently installed Verl exposes the same PPO/GRPO trainer core but requires
    the generated reward-loop, agent-loop and checkpoint-engine sections.  Use
    Verl's own generated reference config when it is available and retain the
    QuantEvolver config only as a compatibility fallback for older installs.
    """
    from omegaconf import OmegaConf

    spec = importlib.util.find_spec("verl")
    if spec is not None and spec.origin:
        generated = Path(spec.origin).resolve().parent / "trainer/config/_generated_ppo_trainer.yaml"
        if generated.exists():
            return OmegaConf.load(generated), generated
    fallback = quantevolver_root / "configs/verl_ppo_trainer_base.yaml"
    return OmegaConf.load(fallback), fallback


def build_verl_grpo_config(
    quantevolver_root: str | Path,
    effective: dict[str, Any],
    train_file: str | Path,
    validation_file: str | Path,
    run_dir: str | Path,
    *,
    experiment_name: str,
    resume_from_path: str | Path | None = None,
    prompt_groups: int | None = None,
    expected_global_step: int = 1,
    total_training_steps: int | None = None,
    reward_function_path: str | Path | None = None,
    agent_loop_config_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    online_dataset: bool = False,
):
    """Compose RLAlpha settings onto QuantEvolver's real Verl trainer config."""
    from omegaconf import OmegaConf

    quantevolver_root, run_dir = Path(quantevolver_root), Path(run_dir)
    base, base_path = _load_verl_base(quantevolver_root)
    model, rollout, actor = effective["model"], effective["rollout"], effective["actor"]
    experiment = effective.get("experiment", {})
    checkpoint_interval = int(actor.get("checkpoint_interval", 50))
    checkpoint_keep = int(actor.get("checkpoint_keep", 2))
    ray_cpus = int(experiment.get("grpo_ray_cpus", 8))
    object_store_bytes = int(float(experiment.get("grpo_object_store_gib", 8)) * 1024**3)
    prompt_groups = int(prompt_groups or 1)
    rollout_n = int(rollout["n"])
    if prompt_groups != 1:
        raise ValueError(f"GRPO fairness protocol requires one prompt group per optimizer update, got {prompt_groups}")
    if rollout_n != 8:
        raise ValueError(f"GRPO fairness protocol requires exactly eight completions, got {rollout_n}")
    seed = int(effective.get("invocation", {}).get("seed", effective.get("seed", 0)))
    micro_batch = max(1, min(prompt_groups * rollout_n, int(os.getenv("RLALPHA_GRPO_MICROBATCH", "2"))))
    max_steps = int(total_training_steps or max(expected_global_step, 1))
    current_api = "reward" in base and "ray_kwargs" in base
    overrides = OmegaConf.create({
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {
            "train_files": [str(Path(train_file).resolve())],
            "val_files": [str(Path(validation_file).resolve())],
            "train_batch_size": prompt_groups,
            "val_batch_size": prompt_groups,
            "max_prompt_length": int(rollout["max_model_len"]) - int(rollout["response_length"]),
            "max_response_length": int(rollout["response_length"]),
            "return_raw_chat": True,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "dataloader_num_workers": 0,
            "validation_shuffle": False,
            "shuffle": False,
            "seed": seed,
            "apply_chat_template_kwargs": {"enable_thinking": False},
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(model["path"]),
                "use_remove_padding": bool(model["use_remove_padding"]),
                "enable_gradient_checkpointing": bool(model["enable_gradient_checkpointing"]),
                "trust_remote_code": bool(model["trust_remote_code"]),
                "override_config": {"attn_implementation": "sdpa"},
                "lora_rank": int(actor["lora_rank"]),
                "lora_alpha": int(actor["lora_alpha"]),
                "target_modules": actor["lora_target_modules"],
            },
            "rollout": {
                "name": rollout["name"], "mode": "async", "n": rollout_n,
                "temperature": float(rollout["temperature"]),
                "response_length": int(rollout["response_length"]),
                "max_model_len": int(rollout["max_model_len"]),
                "tensor_model_parallel_size": 1,
                "log_prob_micro_batch_size_per_gpu": micro_batch,
                "gpu_memory_utilization": float(os.getenv("RLALPHA_VLLM_MEMORY_UTILIZATION", "0.45")),
                "enforce_eager": True,
                "free_cache_engine": True,
                # A dummy vLLM model requires Verl to perform a one-time full
                # base-weight transfer before sending the LoRA adapter.  The
                # current Qwen3.5 HF/PEFT names contain ``.base_layer`` while
                # vLLM's packed QKV modules do not, so that compatibility path
                # fails before the first rollout.  Load the immutable local
                # safetensors base directly; subsequent synchronizations then
                # contain adapter tensors only, which is also substantially
                # cheaper at every staged restart.
                "load_format": "safetensors",
                "calculate_log_probs": True,
                "seed": seed,
                "val_kwargs": {"n": 1, "do_sample": False},
            },
            "actor": {
                # Verl defines this in prompt units and multiplies by rollout.n
                # before the actor update.
                "ppo_mini_batch_size": prompt_groups,
                "ppo_micro_batch_size_per_gpu": micro_batch,
                "use_dynamic_bsz": bool(actor["use_dynamic_bsz"]),
                "use_kl_loss": bool(actor["use_kl_loss"]),
                "kl_loss_coef": float(actor["kl_loss_coef"]),
                "kl_loss_type": actor["kl_loss_type"],
                "clip_ratio": float(actor["clip_ratio"]),
                "clip_ratio_low": float(actor["clip_ratio_low"]),
                "clip_ratio_high": float(actor["clip_ratio_high"]),
                "ppo_epochs": int(actor["ppo_epochs"]),
                "entropy_coeff": float(actor["entropy_coeff"]),
                "calculate_entropy": True,
                "shuffle": False,
                "data_loader_seed": seed,
                "use_torch_compile": False,
                "optim": {"lr": float(actor["learning_rate"])},
                "fsdp_config": {"param_offload": False, "optimizer_offload": False, "seed": seed},
                "checkpoint": {
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                    "save_lora_only": bool(actor.get("save_lora_only", True)),
                },
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": micro_batch, "fsdp_config": {"param_offload": False, "seed": seed}},
        },
        "reward_model": {"enable": False, "launch_reward_fn_async": False},
        "trainer": {
            "project_name": "rlalpha", "experiment_name": experiment_name,
            "logger": ["console", "file"] if metrics_path else ["console"], "nnodes": 1, "n_gpus_per_node": 1,
            "total_epochs": int(expected_global_step), "total_training_steps": max_steps,
            "save_freq": checkpoint_interval, "test_freq": -1,
            "val_before_train": False, "critic_warmup": 0,
            "balance_batch": False,
            "use_v1": False,
            "default_local_dir": str(run_dir / "checkpoints/verl"),
            "validation_data_dir": str(run_dir / "logs/verl_validation"),
            "rollout_data_dir": str(run_dir / "logs/verl_rollouts"),
            "resume_mode": "resume_path" if resume_from_path else "disable",
            "resume_from_path": str(resume_from_path) if resume_from_path else None,
            "max_actor_ckpt_to_keep": checkpoint_keep,
        },
        "ray_init": {"num_cpus": ray_cpus, "object_store_memory": object_store_bytes},
    })
    if online_dataset:
        overrides = OmegaConf.merge(overrides, OmegaConf.create({
            "data": {
                "custom_cls": {
                    "path": str(Path(__file__).with_name("online_dataset.py").resolve()),
                    "name": "OnlinePoolDataset",
                }
            }
        }))
    if current_api:
        overrides = OmegaConf.merge(overrides, OmegaConf.create({
            "reward": {
                "num_workers": 1,
                "custom_reward_function": {
                    "path": str(Path(reward_function_path).resolve()) if reward_function_path else None,
                    "name": "compute_score",
                },
                "reward_model": {"enable": False, "enable_resource_pool": False},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {
                        "default_agent_loop": "rlalpha_structured_single_turn",
                        "agent_loop_config_path": str(Path(agent_loop_config_path).resolve()) if agent_loop_config_path else None,
                    },
                },
            },
            "ray_kwargs": {"ray_init": {"num_cpus": ray_cpus, "object_store_memory": object_store_bytes}},
        }))
    config = OmegaConf.merge(base, overrides)
    OmegaConf.resolve(config)
    config._rlalpha_config_source = str(base_path)
    config._rlalpha_metrics_path = str(metrics_path) if metrics_path else None
    assert_grpo_loss_controls(config)
    return config


def assert_grpo_loss_controls(config: Any) -> None:
    actor = config.actor_rollout_ref.actor
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("advantage estimator is not GRPO")
    if not actor.use_kl_loss or float(actor.kl_loss_coef) <= 0:
        raise ValueError("reference-policy KL is not active")
    if float(actor.clip_ratio) <= 0 or float(actor.clip_ratio_low) <= 0 or float(actor.clip_ratio_high) <= 0:
        raise ValueError("PPO clipping is not active")
    if int(actor.ppo_epochs) < 2:
        raise ValueError("formal GRPO requires rollout reuse across at least two PPO epochs")
    if config.actor_rollout_ref.model.use_remove_padding:
        raise ValueError("Qwen3.5 requires use_remove_padding=false in this verified path")
