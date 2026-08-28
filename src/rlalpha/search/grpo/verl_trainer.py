from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any


def _install_torch_padding_fallback() -> bool:
    """Use Transformers' exact Torch padding helpers when flash-attn is absent.

    Verl's legacy RayPPOTrainer always converts the generated padded batch to a
    jagged TensorDict before old/reference-policy evaluation.  That conversion
    imports ``flash_attn.bert_padding`` even when the actor itself is correctly
    configured for SDPA and ``use_remove_padding=False``.  Recent Transformers
    ships API-compatible pure-Torch implementations for this bookkeeping; use
    those functions without changing the actor attention implementation.

    Returns True only when the fallback was installed.
    """
    try:
        from flash_attn.bert_padding import unpad_input as _unused  # noqa: F401

        return False
    except (ImportError, ModuleNotFoundError):
        from einops import rearrange
        from transformers.modeling_flash_attention_utils import _index_first_axis, _pad_input, _unpad_input
        from verl.utils import attention_utils

        def _torch_attention_functions():
            return _index_first_axis, _pad_input, rearrange, _unpad_input

        attention_utils._get_attention_functions = _torch_attention_functions
        return True


def run_quant_evolver_verl_trainer(
    config: Any,
    reward_fn: Any | None = None,
    validation_reward_fn: Any | None = None,
    *,
    expected_global_step: int | None = None,
) -> dict[str, Any]:
    """Run the actual QuantEvolver/Verl RayPPOTrainer core.

    RLAlpha owns only prompt/reward/pool adapters.  Old-policy log
    probabilities, importance ratios, PPO clipping, reference-policy KL,
    entropy, minibatches, optimizer/scheduler and distributed checkpoints stay
    inside Verl's ``RayPPOTrainer`` and ``ActorRolloutRefWorker``.
    """
    from ..base_llm import configure_packaged_cuda_toolchain

    configure_packaged_cuda_toolchain()
    _install_torch_padding_fallback()
    import ray
    from verl.single_controller.ray import RayWorkerGroup
    try:
        from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
    except ImportError:  # pre reward-loop Verl used by older QuantEvolver installs
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local
    try:
        from verl.workers.engine_workers import ActorRolloutRefWorker

        unified_engine = True
    except ImportError:  # pre unified-engine Verl
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        unified_engine = False

    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_USE_V1": "1",
        "RAY_DEDUP_LOGS": "0",
    }
    python_paths = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[5] / "QuantEvolver")]
    if os.getenv("PYTHONPATH"):
        python_paths.append(os.environ["PYTHONPATH"])
    env_vars["PYTHONPATH"] = os.pathsep.join(python_paths)
    metrics_path = config.get("_rlalpha_metrics_path")
    if metrics_path:
        env_vars["VERL_FILE_LOGGER_PATH"] = str(metrics_path)
        os.environ["VERL_FILE_LOGGER_PATH"] = str(metrics_path)
    try:
        if not ray.is_initialized():
            ray_cfg = config.get("ray_kwargs", {}).get("ray_init", {})
            num_cpus = ray_cfg.get("num_cpus") if ray_cfg else config.get("ray_init", {}).get("num_cpus")
            object_store_memory = ray_cfg.get("object_store_memory") if ray_cfg else config.get("ray_init", {}).get("object_store_memory")
            ray.init(runtime_env={"env_vars": env_vars}, num_cpus=num_cpus, object_store_memory=object_store_memory)
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)
        lora_rank = int(config.actor_rollout_ref.model.get("lora_rank", 0))
        needs_reference = bool(config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward)
        actor_role = Role.ActorRolloutRef if unified_engine and needs_reference and lora_rank <= 0 else Role.ActorRollout
        role_worker_mapping = {actor_role: ray.remote(ActorRolloutRefWorker)}
        mapping = {actor_role: "global_pool"}
        if not unified_engine and needs_reference and lora_rank <= 0:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = "global_pool"
        resources = ResourcePoolManager(resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes}, mapping=mapping)
        dataset_parameters = inspect.signature(create_rl_dataset).parameters
        train_kwargs = {"is_train": True} if "is_train" in dataset_parameters else {}
        validation_kwargs = {"is_train": False} if "is_train" in dataset_parameters else {}
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, **train_kwargs)
        validation_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, **validation_kwargs)
        kwargs = {
            "config": config,
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": role_worker_mapping,
            "resource_pool_manager": resources,
            "ray_worker_group_cls": RayWorkerGroup,
            "train_dataset": train_dataset,
            "val_dataset": validation_dataset,
            "collate_fn": collate_fn,
            "train_sampler": create_rl_sampler(config.data, train_dataset),
            "device_name": config.trainer.device,
        }
        parameters = inspect.signature(RayPPOTrainer).parameters
        if "reward_fn" in parameters:
            if reward_fn is None:
                raise RuntimeError("installed Verl requires the legacy reward_fn callback")
            kwargs["reward_fn"] = reward_fn
            kwargs["val_reward_fn"] = validation_reward_fn or reward_fn
        elif reward_fn is not None:
            raise RuntimeError("installed Verl uses the configured custom reward hook; do not pass reward_fn")
        trainer = RayPPOTrainer(**kwargs)
        trainer.init_workers()
        stopped_online = False
        try:
            trainer.fit()
        except Exception as exc:
            from .online_dataset import OnlineTrainingComplete

            if not isinstance(exc, OnlineTrainingComplete):
                raise
            stopped_online = True
            # The domain callback runs only after the optimizer step.  Verl may
            # already have saved this step when it lands on ``save_freq``.  A
            # duplicate save of the same global step advances Verl's retention
            # queue a second time and can incorrectly evict the preceding
            # checkpoint, so force the terminal save only when it is absent.
            completed_step = int(trainer.global_steps) - 1
            completed_actor = Path(config.trainer.default_local_dir) / f"global_step_{completed_step}" / "actor"
            if not completed_actor.is_dir():
                trainer.global_steps -= 1
                try:
                    trainer._save_checkpoint()
                finally:
                    trainer.global_steps += 1
        completed_step = int(trainer.global_steps) - 1
        if expected_global_step is not None and completed_step != int(expected_global_step):
            raise RuntimeError(f"Verl stopped at optimizer step {completed_step}, expected {expected_global_step}")
        checkpoint = Path(config.trainer.default_local_dir) / f"global_step_{completed_step}"
        if not checkpoint.is_dir() or not (checkpoint / "actor").is_dir():
            raise RuntimeError(f"Verl did not commit the expected actor checkpoint: {checkpoint}")
        return {"global_step": completed_step, "checkpoint": str(checkpoint), "stopped_online": stopped_online}
    finally:
        if ray.is_initialized():
            ray.shutdown()
