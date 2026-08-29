from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlalpha.dsl.parser import parse_expression
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolScore
from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator


class _Objective:
    @staticmethod
    def score_pool(signals):
        value = float(sum(float(np.nanmean(signal)) for signal in signals)) if signals else 0.0
        return PoolScore(value, value, tuple(), tuple(1.0 for _ in signals))


def _coordinator(path: Path, budget: int = 3) -> VerlGRPOStageCoordinator:
    effective = {
        "search": {"group_size": 8},
        "rollout": {"n": 8},
        "actor": {"checkpoint_interval": 50, "checkpoint_keep": 2, "save_lora_only": True},
        "reward": {"invalid_penalty": -1.0},
        "experiment": {"grpo_ray_cpus": 8, "grpo_object_store_gib": 8},
    }
    return VerlGRPOStageCoordinator(
        PoolManager(_Objective()),
        lambda node: np.ones((300, 120)),
        np.ones((300, 120), dtype=bool),
        budget,
        path,
        effective,
        "/qe",
        "/processed",
        "r0",
        4,
    )


def _install_fake_persistent_verl(monkeypatch, coordinator):
    calls = []

    def build(root, effective, train, validation, run_dir, **kwargs):
        assert kwargs["online_dataset"] is True
        assert kwargs["total_training_steps"] > coordinator.ledger.limit
        return OmegaConf.create({
            "data": {"train_files": [str(train)]},
            "trainer": {"default_local_dir": str(Path(run_dir) / "checkpoints/verl")},
            "_rlalpha_metrics_path": str(kwargs["metrics_path"]),
        })

    def commit(self, row, session_dir):
        self.ledger.valid_unique_evaluations += 1
        self.updates += 1
        self.stage += 1
        self.pool_snapshots.append({
            "snapshot_id": f"snapshot_{self.updates}",
            "pool_version": self.pool.version,
            "expressions": [],
            "factors": [],
            "train": {},
            "valid_unique_evaluations": self.ledger.valid_unique_evaluations,
            "optimizer_update": self.updates,
        })
        if self.ledger.exhausted:
            return None
        return self._prepare_online_stage(session_dir)

    coordinator._commit_online_stage = MethodType(commit, coordinator)

    def run(config, expected_global_step=None):
        from rlalpha.search.grpo.online_dataset import _CALLBACKS

        calls.append("trainer")
        key = str(Path(config.data.train_files[0]).resolve())
        callback = _CALLBACKS[key]
        while callback({"batch": object()}) is not None:
            pass
        checkpoint = Path(config.trainer.default_local_dir) / f"global_step_{coordinator.updates}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "actor/adapter_model.safetensors").write_bytes(b"lora-only")
        return {"global_step": coordinator.updates, "checkpoint": str(checkpoint), "stopped_online": True}

    monkeypatch.setattr("rlalpha.search.grpo.stage_coordinator.build_verl_grpo_config", build)
    monkeypatch.setattr("rlalpha.search.grpo.stage_coordinator.run_quant_evolver_verl_trainer", run)
    return calls


def test_multiple_optimizer_updates_use_one_persistent_trainer(monkeypatch, tmp_path):
    coordinator = _coordinator(tmp_path / "run")
    calls = _install_fake_persistent_verl(monkeypatch, coordinator)
    result = coordinator.run_cell()
    assert calls == ["trainer"]
    assert coordinator.updates == 3
    assert coordinator.ledger.valid_unique_evaluations == 3
    assert len(result["pool_snapshots"]) == 3
    checkpoint = Path(result["checkpoint"])
    assert (checkpoint / "actor/adapter_model.safetensors").stat().st_size < 1024
    assert coordinator.checkpoint_fingerprint == {"global_step": 3, "save_lora_only": True}


def test_dynamically_loaded_online_dataset_shares_callback_and_stop_signal():
    import importlib.util
    from rlalpha.search.grpo import online_dataset

    spec = importlib.util.spec_from_file_location("verl_external_online_dataset", online_dataset.__file__)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._CALLBACKS is online_dataset._CALLBACKS
    assert module.OnlineTrainingComplete is online_dataset.OnlineTrainingComplete
    messages = [{"role": "user", "content": "factor"}]
    assert module.OnlinePoolDataset._build_messages({"prompt": messages}, "prompt") == messages
    assert module.OnlinePoolDataset._process_multi_modal_info(messages, 14, {}) == (None, None, None)


def test_reward_worker_signal_cache_prevents_parent_reevaluation(tmp_path):
    coordinator = _coordinator(tmp_path / "run")
    node = parse_expression("CSRank($return)")
    signal = np.arange(300 * 120, dtype=float).reshape(300, 120)
    coordinator.signal_cache.put(node.expr_hash, signal, permanent=True)

    def unexpected_evaluation(_):
        raise AssertionError("shared reward-worker signal should be reused")

    coordinator.evaluator = unexpected_evaluation
    record = {
        "stage": 0,
        "prompt_group": 0,
        "rollout_index": 0,
        "raw_text": f"<expr>{node.canonical()}</expr>",
        "expression": node.canonical(),
        "expr_hash": node.expr_hash,
        "valid": True,
        "reason_code": "ok",
        "market_evaluated": True,
        "delta_objective": 0.01,
        "shaped_reward": 1.0,
        "delta_add": 0.01,
        "replaced_hash": None,
        "saliency": [],
        "eviction_candidates": [],
        "post_prune_delta": 0.01,
        "self_evicted": False,
        "formally_rechecked": True,
        "pool_score": {
            "objective": 0.01,
            "mean_ic": 0.01,
            "daily_ic": [],
            "weights": [1.0],
            "standard_error": 0.0,
        },
        "reward_valid_days": 300,
        "reward_valid_observations": int(signal.size),
        "reward_valid_day_rate": 1.0,
        "reward_observation_rate": 1.0,
    }
    entries, scores = coordinator._consume_records([record])
    assert len(entries) == len(scores) == 1
    assert np.array_equal(entries[0].signal, signal)


def test_stage_io_paths_do_not_change_frozen_semantic_hash(tmp_path):
    first = _coordinator(tmp_path / "first")
    second = _coordinator(tmp_path / "second")
    left = first._stage_spec(tmp_path / "left.jsonl", 8)
    right = second._stage_spec(tmp_path / "right.jsonl", 8)
    assert left["archive_path"] != right["archive_path"]
    assert left["signal_cache_root"] != right["signal_cache_root"]
    assert left["spec_hash"] == right["spec_hash"]


def test_resume_ignores_unpaired_journal_and_rejects_old_semantics(monkeypatch, tmp_path):
    coordinator = _coordinator(tmp_path / "run", budget=2)
    _install_fake_persistent_verl(monkeypatch, coordinator)
    coordinator.run_cell()
    orphan_hash = "reward_worker_orphan"
    coordinator.signal_cache.put(orphan_hash, np.ones((3, 2)), permanent=True)
    orphan_path = coordinator.signal_cache.root / f"{orphan_hash}.npy"
    assert orphan_path.exists()
    journal = coordinator.run_dir / "checkpoints/domain_journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"optimizer_update": 999, "unpaired": True}) + "\n", encoding="utf-8")

    resumed = _coordinator(tmp_path / "run", budget=2)
    resumed.load_checkpoint()
    assert resumed.updates == 2
    assert resumed.ledger.valid_unique_evaluations == 2
    assert not orphan_path.exists()

    commit = json.loads((coordinator.run_dir / "checkpoint_commit.json").read_text(encoding="utf-8"))
    commit["reward_pool_semantics"] = "legacy"
    (coordinator.run_dir / "checkpoint_commit.json").write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        _coordinator(tmp_path / "run", budget=2).load_checkpoint()
