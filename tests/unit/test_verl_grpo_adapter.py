from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rlalpha.search.grpo.verl_config import assert_grpo_loss_controls, build_verl_grpo_config
from rlalpha.search.grpo.verl_reward_adapter import VerlRLAlphaRewardBridge
from rlalpha.search.grpo import verl_reward_function
from rlalpha.search.grpo.verl_trainer import _install_torch_padding_fallback
from rlalpha.utils.hashing import stable_hash


def test_verl_reward_adapter_places_finite_rewards_and_preserves_frozen_state(tmp_path):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    state = {"hash": "pool-v3"}
    seen = []

    def score(texts, info):
        seen.extend(texts)
        return [{"valid": True, "reason_code": "ok", "shaped_reward": 0.5}, {"valid": False, "reason_code": "bad", "shaped_reward": float("nan")}]

    data = SimpleNamespace(
        batch={"responses": torch.tensor([[4, 5, 0], [7, 0, 0]]), "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])},
        non_tensor_batch={"extra_info": np.asarray([{"pool_version": 3}, {"pool_version": 3}], dtype=object)},
    )
    bridge = VerlRLAlphaRewardBridge(Tokenizer(), score, lambda: state["hash"], archive_path=tmp_path / "archive.jsonl")
    result = bridge(data, return_dict=True)
    assert torch.equal(result["reward_tensor"], torch.tensor([[0.0, 0.5, 0.0], [-1.0, 0.0, 0.0]]))
    assert result["reward_extra_info"]["reason_code"] == ["ok", "non_finite_reward"]
    assert len((tmp_path / "archive.jsonl").read_text().splitlines()) == 2
    assert seen == ["4 5", "7"]


def test_verl_config_consumes_ratio_clip_reference_kl_and_qwen_padding(tmp_path):
    pytest.importorskip("omegaconf")
    root = tmp_path / "qe"
    (root / "configs").mkdir(parents=True)
    source = __import__("pathlib").Path(__file__).parents[3] / "QuantEvolver/configs/verl_ppo_trainer_base.yaml"
    (root / "configs/verl_ppo_trainer_base.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    no_think = root / "quant_evolver/rft/no_think_dataset.py"
    no_think.parent.mkdir(parents=True)
    no_think.write_text("class NoThinkRLHFDataset: pass\n", encoding="utf-8")
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    train.touch(); validation.touch()
    effective = {
        "model": {"path": "/model", "use_remove_padding": False, "enable_gradient_checkpointing": True, "trust_remote_code": True},
        "rollout": {"name": "vllm", "n": 8, "temperature": 1.0, "response_length": 128, "max_model_len": 4096},
        "actor": {"lora_rank": 16, "lora_alpha": 32, "lora_target_modules": "all-linear", "use_dynamic_bsz": False, "use_kl_loss": True, "kl_loss_coef": 0.001, "kl_loss_type": "low_var_kl", "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "ppo_epochs": 2, "entropy_coeff": 0.01, "learning_rate": 1e-6},
        "search": {"group_size": 8},
    }
    config = build_verl_grpo_config(root, effective, train, validation, tmp_path / "run", experiment_name="smoke")
    assert config.algorithm.adv_estimator == "grpo"
    assert config.actor_rollout_ref.actor.clip_ratio == 0.2
    assert config.actor_rollout_ref.actor.use_kl_loss
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert config.actor_rollout_ref.actor.ppo_epochs == 2
    assert config.actor_rollout_ref.actor.entropy_coeff == 0.01
    assert config.actor_rollout_ref.model.use_remove_padding is False
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.rollout.load_format == "safetensors"
    assert config.data.train_batch_size == 1
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 1
    config.actor_rollout_ref.actor.ppo_epochs = 1
    with pytest.raises(ValueError, match="rollout reuse"):
        assert_grpo_loss_controls(config)


def test_verl_old_log_prob_padding_conversion_has_torch_fallback():
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    from verl.workers.utils.padding import left_right_2_no_padding

    assert _install_torch_padding_fallback()
    data = TensorDict(
        {
            "input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
            "response_mask": torch.ones((2, 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]]),
        },
        batch_size=[2],
    )
    converted = left_right_2_no_padding(data)
    assert converted["input_ids"].is_nested
    assert converted["input_ids"].values().tolist() == [1, 2, 3, 4, 5, 6]


def test_installed_verl_loss_graph_consumes_clip_and_reference_kl(monkeypatch):
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils import losses as verl_losses

    old_log_prob = torch.zeros((2, 3))
    log_prob = torch.full((2, 3), math.log(2.0))
    advantages = torch.ones((2, 3))
    response_mask = torch.ones((2, 3), dtype=torch.bool)
    clipped = OmegaConf.create(
        {"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0, "global_batch_info": {}}
    )
    loose = OmegaConf.create(
        {"clip_ratio": 10.0, "clip_ratio_low": 10.0, "clip_ratio_high": 10.0, "clip_ratio_c": 3.0, "global_batch_info": {}}
    )
    clipped_loss, clipped_metrics = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, config=clipped
    )
    loose_loss, loose_metrics = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, config=loose
    )
    assert clipped_metrics["actor/pg_clipfrac"] == 1.0
    assert loose_metrics["actor/pg_clipfrac"] == 0.0
    assert clipped_loss.item() == pytest.approx(-1.2)
    assert loose_loss.item() == pytest.approx(-2.0)

    # Exercise the same composite PPO loss used by ActorRolloutRefWorker.  The
    # tensors here are already padded, so bypass only its jagged-to-padded
    # representation conversion and leave the policy/KL implementation intact.
    monkeypatch.setattr(verl_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    data = TensorDict(
        {
            "response_mask": response_mask,
            "old_log_probs": old_log_prob,
            "advantages": advantages,
            "ref_log_prob": torch.zeros((2, 3)),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    config = OmegaConf.create(
        {
            "clip_ratio": 10.0,
            "clip_ratio_low": 10.0,
            "clip_ratio_high": 10.0,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
            "loss_scale_factor": None,
            "loss_agg_mode": "token-mean",
            "policy_loss": {"loss_mode": "vanilla"},
            "entropy_coeff": 0.0,
            "use_kl_loss": True,
            "kl_loss_type": "mse",
            "kl_loss_coef": 0.5,
        }
    )
    model_output = {"log_probs": log_prob}
    with_kl, with_metrics = verl_losses.ppo_loss(config, model_output, data.clone())
    config.kl_loss_coef = 0.0
    without_kl, _ = verl_losses.ppo_loss(config, model_output, data.clone())
    assert with_metrics["kl_loss"].aggregate() > 0
    assert with_kl.item() > without_kl.item()


def test_current_verl_reward_batch_is_frozen_and_marks_all_stage_duplicates(monkeypatch, tmp_path):
    rng = np.random.default_rng(17)

    class Panel:
        label = rng.normal(size=(300, 120))
        exposures = np.ones((300, 120, 1))
        common_mask = np.ones((300, 120), dtype=bool)

        @staticmethod
        def target(value):
            return value

        @staticmethod
        def evaluate(node):
            local = np.random.default_rng(int(node.expr_hash[:8], 16))
            return local.normal(size=(300, 120))

    class Store:
        def __init__(self, root):
            assert root == "/processed"

        @staticmethod
        def load_split(name, start=None, end=None):
            assert name == "train"
            return Panel()

    monkeypatch.setattr(verl_reward_function, "PanelStore", Store)
    archive = tmp_path / "rollouts.jsonl"
    spec_payload = {
        "schema_version": 1,
        "stage": 0,
        "expected_samples": 3,
        "remaining_budget": 10,
        "invalid_penalty": -1.0,
        "reward": "r0",
        "processed_root": "/processed",
        "train_start": None,
        "train_end": None,
        "pool_version": 0,
        "pool_capacity": 20,
        "min_delta": 1e-5,
        "pool_objective": 0.0,
        "pool_weights": [],
        "pool": [],
        "seen_hashes": [],
        "archive_path": str(archive),
    }
    spec = {
        **spec_payload,
        "spec_hash": stable_hash({key: value for key, value in spec_payload.items() if key != "archive_path"}),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(__import__("json").dumps(spec), encoding="utf-8")

    def request(text, group):
        return {
            "solution_str": text,
            "extra_info": {
                "stage": 0,
                "prompt_group": group,
                "split": "train",
                "pool_version": 0,
                "expected_stage_samples": 3,
                "stage_spec_path": str(spec_path),
                "frozen_state_hash": spec["spec_hash"],
            },
        }

    records = verl_reward_function._score_batch_sync(
        [request("<expr>CSRank($return)</expr>", 0), request("<expr>CSRank($return)</expr>", 0), request("<expr>Mean($return,5)</expr>", 1)]
    )
    assert [item["reason_code"] for item in records[:2]] == ["stage_duplicate", "stage_duplicate"]
    assert records[2]["market_evaluated"]
    assert len(archive.read_text(encoding="utf-8").splitlines()) == 3

    validation_requests = [request("<expr>CSRank($return)</expr>", group) for group in range(3)]
    for item in validation_requests:
        item["extra_info"]["split"] = "validation"
    with pytest.raises(RuntimeError, match="train split"):
        verl_reward_function._score_batch_sync(validation_requests)
