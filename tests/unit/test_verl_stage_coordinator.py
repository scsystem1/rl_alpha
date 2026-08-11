from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolScore
from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator


class _Objective:
    @staticmethod
    def score_pool(signals):
        value = float(sum(float(np.nanmean(signal)) for signal in signals)) if signals else 0.0
        return PoolScore(value, value, tuple(), tuple(1.0 for _ in signals))


def _coordinator(path: Path) -> VerlGRPOStageCoordinator:
    effective = {
        "search": {"group_size": 8},
        "rollout": {"n": 8},
        "reward": {"invalid_penalty": -1.0},
    }
    return VerlGRPOStageCoordinator(
        PoolManager(_Objective()),
        lambda node: np.ones((300, 120)),
        np.ones((300, 120), dtype=bool),
        8,
        path,
        effective,
        "/qe",
        "/processed",
        "r0",
        4,
    )


def _install_fake_verl(monkeypatch):
    resume_paths = []

    def build(root, effective, train, validation, run_dir, **kwargs):
        resume_paths.append(kwargs.get("resume_from_path"))
        return OmegaConf.create(
            {
                "data": {"train_files": [str(train)]},
                "trainer": {
                    "default_local_dir": str(Path(run_dir) / "checkpoints/verl"),
                    "resume_from_path": str(kwargs["resume_from_path"]) if kwargs.get("resume_from_path") else None,
                },
                "_rlalpha_metrics_path": str(kwargs["metrics_path"]),
            }
        )

    def run(config, expected_global_step=None):
        train = __import__("pandas").read_parquet(config.data.train_files[0])
        assert len(train) == 1
        spec_path = Path(train.iloc[0]["extra_info"]["stage_spec_path"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert spec["expected_samples"] == 8
        prompt = str(train.iloc[0]["prompt"]).lower()
        assert "momentum" in prompt and "mean reversion" in prompt and "price-volume" in prompt
        records = []
        for group in range(1):
            for rollout in range(8):
                records.append(
                    {
                        "stage": spec["stage"],
                        "prompt_group": group,
                        "rollout_index": rollout,
                        "raw_text": "invalid",
                        "expression": None,
                        "expr_hash": None,
                        "valid": False,
                        "reason_code": "parse_or_type_error",
                        "market_evaluated": False,
                        "shaped_reward": -1.0,
                        "delta_objective": None,
                        "pool_objective_before": 0.0,
                        "pool_objective_after": None,
                        "replaced_hash": None,
                        "frozen_state_hash": spec["spec_hash"],
                    }
                )
        Path(spec["archive_path"]).write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        metrics = {
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/kl_loss": 0.01,
            "actor/grad_norm": 0.2,
            "actor/lr": 1e-6,
            "critic/advantages/mean": 0.0,
            "actor/entropy": 1.0,
            "response_length/mean": 5.0,
            "timing_s/step": 0.1,
        }
        Path(config._rlalpha_metrics_path).write_text(json.dumps({"step": expected_global_step, "data": metrics}) + "\n", encoding="utf-8")
        checkpoint = Path(config.trainer.default_local_dir) / f"global_step_{expected_global_step}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "actor/model.bin").write_bytes(f"step-{expected_global_step}".encode())
        return {"global_step": expected_global_step, "checkpoint": str(checkpoint)}

    monkeypatch.setattr("rlalpha.search.grpo.stage_coordinator.build_verl_grpo_config", build)
    monkeypatch.setattr("rlalpha.search.grpo.stage_coordinator.run_quant_evolver_verl_trainer", run)
    return resume_paths


def test_stage_ends_without_admission_and_resume_uses_committed_optimizer_state(monkeypatch, tmp_path):
    resume_paths = _install_fake_verl(monkeypatch)
    first = _coordinator(tmp_path / "resumed")
    stage0 = first.run_stage()
    assert first.stage == 1 and first.updates == 1
    assert not stage0["admission"]["admitted"]

    resumed = _coordinator(tmp_path / "resumed")
    resumed.load_checkpoint()
    resumed.run_stage()
    assert resumed.stage == 2 and resumed.updates == 2
    assert resume_paths[0] is None
    assert Path(resume_paths[1]).name == "global_step_1"
    assert [event["kind"] for event in resumed.events].count("stage_end") == 2
    assert [event["event_id"] for event in resumed.events] == list(range(1, len(resumed.events) + 1))
    assert resumed.checkpoint_fingerprint["sha256"] != stage0["checkpoint_fingerprint"]["sha256"]
    assert resumed.zero_group_variance == 2


def test_fresh_and_boundary_resume_have_equivalent_outer_state(monkeypatch, tmp_path):
    """A process boundary must not change stages, lineage, or frozen samples."""
    _install_fake_verl(monkeypatch)
    continuous = _coordinator(tmp_path / "continuous")
    continuous_stage0 = continuous.run_stage()
    continuous_stage1 = continuous.run_stage()

    resumed_first = _coordinator(tmp_path / "resumed")
    resumed_stage0 = resumed_first.run_stage()
    resumed = _coordinator(tmp_path / "resumed")
    resumed.load_checkpoint()
    resumed_stage1 = resumed.run_stage()

    continuous_state = continuous.state_dict()
    resumed_state = resumed.state_dict()
    continuous_state.pop("checkpoint")
    resumed_state.pop("checkpoint")
    assert continuous_state == resumed_state
    assert continuous.ledger.state_dict() == resumed.ledger.state_dict()
    assert continuous.records == resumed.records
    assert [event["kind"] for event in continuous.events] == [event["kind"] for event in resumed.events]
    assert [event["event_id"] for event in continuous.events] == [event["event_id"] for event in resumed.events]
    assert continuous_stage0["checkpoint_fingerprint"]["sha256"] == resumed_stage0["checkpoint_fingerprint"]["sha256"]
    assert continuous_stage1["checkpoint_fingerprint"]["sha256"] == resumed_stage1["checkpoint_fingerprint"]["sha256"]
    assert Path(continuous_stage0["attempt_dir"]).name == Path(resumed_stage0["attempt_dir"]).name
    assert Path(continuous_stage1["attempt_dir"]).name == Path(resumed_stage1["attempt_dir"]).name
    for uninterrupted, after_restart in (
        (continuous_stage0, resumed_stage0),
        (continuous_stage1, resumed_stage1),
    ):
        left = Path(uninterrupted["attempt_dir"])
        right = Path(after_restart["attempt_dir"])
        left_spec = json.loads((left / "frozen_stage_spec.json").read_text(encoding="utf-8"))
        right_spec = json.loads((right / "frozen_stage_spec.json").read_text(encoding="utf-8"))
        assert left_spec["spec_hash"] == right_spec["spec_hash"]
        assert (left / "rollouts.jsonl").read_text(encoding="utf-8") == (right / "rollouts.jsonl").read_text(encoding="utf-8")
        assert uninterrupted["admission"] == after_restart["admission"]
