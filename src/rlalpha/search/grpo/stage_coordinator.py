from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...dsl.parser import parse_expression
from ...factors.cache import SignalCache
from ...factors.pool import PoolManager
from ...factors.records import CandidateScore, PoolEntry, PoolScore
from ...utils.hashing import file_fingerprint, stable_hash
from ...utils.io import atomic_write_text, write_json
from ..models import BudgetLedger, SearchContext
from ..prompts import build_messages
from .verl_config import build_verl_grpo_config
from .verl_trainer import run_quant_evolver_verl_trainer


class VerlGRPOStageCoordinator:
    """Persistent online-pool state machine around one QuantEvolver/Verl run.

    Verl owns rollout, old/reference log probabilities, GRPO advantages, the
    PPO clipped loss, KL/entropy terms, optimizer and distributed model state.
    This coordinator owns only domain prompts, train-only scoring, one admission
    per update and paired domain commits linking pool state to Verl checkpoints.
    """

    def __init__(
        self,
        pool: PoolManager,
        evaluator: Callable[[object], np.ndarray],
        membership: np.ndarray,
        budget: int,
        run_dir: str | Path,
        effective_config: dict[str, Any],
        quantevolver_root: str | Path,
        processed_root: str | Path,
        reward: str,
        seed: int,
        *,
        train_start: str | None = None,
        train_end: str | None = None,
        max_training_steps: int = 100_000,
    ):
        self.pool = pool
        self.evaluator = evaluator
        self.membership = np.asarray(membership, dtype=bool)
        self.ledger = BudgetLedger(int(budget))
        self.run_dir = Path(run_dir)
        self.effective_config = effective_config
        self.quantevolver_root = Path(quantevolver_root)
        self.processed_root = Path(processed_root)
        self.reward = reward
        self.seed = int(seed)
        self.train_start = train_start
        self.train_end = train_end
        self.max_training_steps = int(max_training_steps)
        self.signal_cache = SignalCache(self.run_dir / "cache/signals")
        self.seen: set[str] = set()
        self.signals: dict[str, np.ndarray] = {}
        self.records: list[dict[str, Any]] = []
        self.stage = 0
        self.updates = 0
        self.event_id = 0
        self.events: list[dict[str, Any]] = []
        self.zero_group_variance = 0
        self.checkpoint: Path | None = None
        self.checkpoint_fingerprint: dict[str, Any] | None = None
        self.total_tokens = 0
        self.gpu_seconds = 0.0
        self.searcher = self
        self.pool_snapshots: list[dict[str, Any]] = []
        self._persisted_record_count = 0

    def context(self) -> SearchContext:
        score = self.pool.score
        return SearchContext(
            self.pool.version,
            tuple(item.expression for item in self.pool.entries),
            tuple(score.weights),
            score.objective,
            self.ledger.valid_unique_evaluations,
            self.ledger.limit,
            tuple(self.records[-64:]),
        )

    def _event(self, kind: str, **payload: Any) -> dict[str, Any]:
        self.event_id += 1
        event = {"event_id": self.event_id, "kind": kind, "stage": self.stage, "optimizer_update": self.updates, **payload}
        self.events.append(event)
        return event

    def _stage_spec(self, archive_path: Path, expected_samples: int) -> dict[str, Any]:
        score = self.pool.score
        spec = {
            "schema_version": 4,
            "reward_pool_semantics": "independent-availability-same-support-admission-v4",
            "stage": self.stage,
            "expected_samples": int(expected_samples),
            "remaining_budget": self.ledger.remaining,
            "invalid_penalty": float(self.effective_config.get("reward", {}).get("invalid_penalty", -1.0)),
            "reward": self.reward,
            "reward_config": dict(self.effective_config.get("reward", {})),
            "processed_root": str(self.processed_root.resolve()),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "pool_version": self.pool.version,
            "pool_capacity": self.pool.capacity,
            "min_delta": self.pool.min_delta,
            "replacement_top_k": self.pool.replacement_top_k,
            "admission_recheck_top_k": self.pool.admission_recheck_top_k,
            "candidate_workers": int(
                self.effective_config.get("experiment", {}).get(
                    "grpo_reward_candidate_workers", 4
                )
            ),
            "pool_objective": float(score.objective),
            "pool_weights": list(map(float, score.weights)),
            "pool": [
                {"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata}
                for item in self.pool.entries
            ],
            "seen_hashes": sorted(self.seen),
        }
        # The frozen-state identity is semantic and therefore independent of
        # the checkout/run directory.  The archive destination is an I/O
        # address, not part of the pool/reward state.
        return {
            **spec,
            "archive_path": str(archive_path.resolve()),
            "spec_hash": stable_hash(spec),
        }

    def _attempt_id(self) -> str:
        """Return a reproducible stage-attempt ID, preserving failed retries."""
        base = "attempt_" + stable_hash(
            {
                "stage": self.stage,
                "optimizer_update": self.updates,
                "resume_checkpoint_hash": (
                    self.checkpoint_fingerprint["sha256"] if self.checkpoint_fingerprint else None
                ),
                "pool_version": self.pool.version,
                "pool_hashes": [entry.expr_hash for entry in self.pool.entries],
                "seen_hashes": sorted(self.seen),
                "seed": self.seed,
            }
        )[:16]
        stage_root = self.run_dir / "stages" / f"stage_{self.stage:05d}"
        candidate = base
        retry = 0
        while (stage_root / candidate).exists():
            retry += 1
            candidate = f"{base}_retry_{retry:02d}"
        return candidate

    @staticmethod
    def _write_prompts(
        path: Path,
        context: SearchContext,
        stage: int,
        groups: int,
        spec_path: Path,
        spec_hash: str,
        expected_samples: int,
        split: str,
    ) -> None:
        if groups != 1:
            raise ValueError(f"GRPO fairness protocol requires one prompt group per round, got {groups}")
        rows = []
        for group in range(groups):
            rows.append(
                {
                    "data_source": f"rlalpha/grpo/{split}",
                    "prompt": build_messages(context),
                    "reward_model": {"ground_truth": [], "style": "rule"},
                    "extra_info": {
                        "index": stage * groups + group,
                        "stage": stage,
                        "prompt_group": group,
                        "split": split,
                        "pool_version": context.pool_version,
                        "stage_spec_path": str(spec_path.resolve()),
                        "frozen_state_hash": spec_hash,
                        "expected_stage_samples": expected_samples,
                    },
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    @staticmethod
    def _read_verl_metrics(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError(f"Verl file logger did not produce metrics: {path}")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Verl metrics log is empty")
        return dict(lines[-1].get("data") or {})

    @staticmethod
    def _domain_metrics(records: list[dict[str, Any]], groups: int) -> dict[str, Any]:
        rewards = np.asarray([float(item["shaped_reward"]) for item in records], dtype=float)
        grouped = [[float(item["shaped_reward"]) for item in records if int(item["prompt_group"]) == group] for group in range(groups)]
        normalized = []
        zero = 0
        for values in grouped:
            array = np.asarray(values, dtype=float)
            if len(array) == 0 or float(np.std(array)) <= 1e-12:
                zero += 1
                normalized.extend([0.0] * len(array))
            else:
                normalized.extend(((array - array.mean()) / (array.std() + 1e-6)).tolist())
        valid = sum(bool(item["valid"]) for item in records)
        unique = sum(bool(item["market_evaluated"]) for item in records)
        return {
            "domain/invalid_rate": 1.0 - valid / max(1, len(records)),
            "domain/unique_rate": unique / max(1, len(records)),
            "domain/reward_mean": float(rewards.mean()),
            "domain/reward_std": float(rewards.std()),
            "domain/reward_min": float(rewards.min()),
            "domain/reward_max": float(rewards.max()),
            "domain/advantage_mean": float(np.mean(normalized)) if normalized else 0.0,
            "domain/advantage_std": float(np.std(normalized)) if normalized else 0.0,
            "domain/zero_variance_groups": zero,
        }

    def _consume_records(self, records: list[dict[str, Any]]) -> tuple[list[PoolEntry], list[Any]]:
        pre_hash = stable_hash({"pool_version": self.pool.version, "factors": [entry.expr_hash for entry in self.pool.entries]})
        entries: list[PoolEntry] = []
        standard_records = []
        for raw_index, item in enumerate(records):
            reason = str(item["reason_code"])
            expr_hash = item.get("expr_hash")
            expression = item.get("expression") or item.get("raw_text") or ""
            if reason in {"exact_duplicate", "intra_group_duplicate_reused"}:
                self.ledger.duplicates += 1
            elif reason not in {"ok", "budget_exhausted"}:
                self.ledger.invalid += 1
            if expr_hash and reason not in {"exact_duplicate", "intra_group_duplicate_reused", "budget_exhausted"}:
                self.seen.add(str(expr_hash))
            metadata = {
                "proposal_id": f"proposal_{stable_hash({'seed': self.seed, 'stage': self.stage, 'prompt_group': item['prompt_group'], 'rollout_index': item['rollout_index'], 'raw_text': item['raw_text']})[:24]}",
                "factor_id": expr_hash,
                "generator": "grpo_llm",
                "parents": [],
                "raw_text": item["raw_text"],
                "group_index": self.stage,
                "raw_proposal_index": self.ledger.raw_proposals + raw_index,
                "pre_group_pool_version": self.pool.version,
                "pre_group_pool_snapshot_hash": pre_hash,
                "stage": self.stage,
                "prompt_group": int(item["prompt_group"]),
                "rollout_index": int(item["rollout_index"]),
                "optimizer_update": self.updates + 1,
                "reward_diagnostics": {key: value for key, value in item.items() if key not in {"raw_text", "expression", "expr_hash"}},
            }
            if bool(item["market_evaluated"]):
                self.ledger.valid_unique_evaluations += 1
                node = parse_expression(str(item["expression"]))
                signal = self.signal_cache.get(node.expr_hash)
                if signal is None:
                    signal = self.evaluator(node)
                    self.signal_cache.put(node.expr_hash, signal, permanent=True)
                self.signals[node.expr_hash] = np.asarray(signal)
                entries.append(PoolEntry(node.canonical(), node.expr_hash, signal, metadata))
            standard_records.append(
                {
                    "expr_hash": expr_hash or stable_hash({"invalid": item["raw_text"]}),
                    "expression": expression,
                    "valid": bool(item["valid"]),
                    "reason": reason,
                    "market_evaluated": bool(item["market_evaluated"]),
                    "delta_objective": float(item["delta_objective"]) if item.get("delta_objective") is not None else 0.0,
                    "shaped_reward": float(item["shaped_reward"]),
                    "metadata": metadata,
                }
            )
        self.ledger.raw_proposals += len(records)
        archived_scored = [item for item in records if bool(item.get("market_evaluated"))]
        if len(archived_scored) != len(entries):
            raise RuntimeError("reward archive and parent candidate counts differ")
        scored = []
        for entry, archived in zip(entries, archived_scored, strict=True):
            raw_pool_score = dict(archived["pool_score"])
            pool_score = PoolScore(
                float(raw_pool_score["objective"]),
                float(raw_pool_score["mean_ic"]),
                tuple(map(float, raw_pool_score["daily_ic"])),
                tuple(map(float, raw_pool_score["weights"])),
                float(raw_pool_score.get("standard_error", float("nan"))),
            )
            scored.append(CandidateScore(
                entry.expr_hash,
                pool_score,
                float(archived["delta_objective"]),
                float(archived["shaped_reward"]),
                archived.get("replaced_hash"),
                bool(archived["valid"]),
                str(archived["reason_code"]),
                float(archived["delta_add"]),
                tuple(map(float, archived.get("saliency") or ())),
                tuple(map(str, archived.get("eviction_candidates") or ())),
                float(archived["post_prune_delta"]),
                bool(archived["self_evicted"]),
                bool(archived["formally_rechecked"]),
                bool(float(archived["delta_add"]) > 0 and float(archived["post_prune_delta"]) <= self.pool.min_delta),
                int(archived.get("reward_valid_days") or 0),
                int(archived.get("reward_valid_observations") or 0),
                float(archived.get("reward_valid_day_rate") or 0.0),
                float(archived.get("reward_observation_rate") or 0.0),
            ))
        self.records.extend(standard_records)
        return entries, scored

    def _prepare_online_stage(self, session_dir: Path) -> dict[str, Any]:
        base = session_dir / "stages" / f"stage_{self.stage:05d}"
        attempt = base
        retry = 0
        while attempt.exists():
            retry += 1
            attempt = base.with_name(f"{base.name}_retry_{retry:02d}")
        attempt.mkdir(parents=True, exist_ok=False)
        archive_path = attempt / "rollouts.jsonl"
        spec_path = attempt / "frozen_stage_spec.json"
        spec = self._stage_spec(archive_path, int(self.effective_config["rollout"]["n"]))
        write_json(spec_path, spec)
        context = self.context()
        return {
            "data_source": "rlalpha/grpo/train",
            "prompt": build_messages(context),
            "reward_model": {"ground_truth": [], "style": "rule"},
            "extra_info": {
                "index": self.stage,
                "stage": self.stage,
                "prompt_group": 0,
                "split": "train",
                "pool_version": context.pool_version,
                "stage_spec_path": str(spec_path.resolve()),
                "frozen_state_hash": spec["spec_hash"],
                "expected_stage_samples": int(self.effective_config["rollout"]["n"]),
            },
        }

    def _commit_online_stage(self, row: dict[str, Any], session_dir: Path) -> dict[str, Any] | None:
        extra = row["extra_info"]
        spec_path = Path(extra["stage_spec_path"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        archive_path = Path(spec["archive_path"])
        if not archive_path.exists():
            raise RuntimeError("optimizer completed without a pending domain transition")
        records = [json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = int(spec["expected_samples"])
        if len(records) != expected:
            raise RuntimeError(f"pending domain transition has {len(records)} records, expected {expected}")
        records.sort(key=lambda item: (int(item["prompt_group"]), int(item["rollout_index"]), str(item.get("expression") or "")))
        context = self.context()
        entries, scores = self._consume_records(records)
        admission = self.pool.consider_group(entries, scores)
        for record in self.records[-len(records) :]:
            metadata = dict(record.get("metadata") or {})
            admitted = bool(admission.admitted and admission.candidate_hash == record.get("expr_hash"))
            metadata["admitted"] = admitted
            metadata["positive_not_admitted"] = bool(
                float(record.get("delta_objective") or 0.0) > 0 and not admitted
            )
            record["metadata"] = metadata
        self.updates += 1
        self.stage += 1
        domain_metrics = self._domain_metrics(records, 1)
        self.zero_group_variance += int(domain_metrics["domain/zero_variance_groups"])
        self.total_tokens += int(sum(len(str(item.get("raw_text") or "").split()) for item in records))
        self.ledger.tokens = self.total_tokens
        prepared = self.pool.prepared_state()
        train_score = asdict(self.pool.score)
        support_diagnostics = getattr(self.pool.objective, "support_diagnostics", None)
        if prepared is not None and callable(support_diagnostics):
            train_score["support"] = support_diagnostics(prepared)
        snapshot_factors = [
            {
                "factor_id": entry.expr_hash,
                "proposal_id": entry.metadata.get("proposal_id"),
                "expression": entry.expression,
                "generator": entry.metadata.get("generator"),
                "parents": entry.metadata.get("parents", []),
                "search_weight": train_score.get("weights", [])[index] if index < len(train_score.get("weights", [])) else None,
            }
            for index, entry in enumerate(self.pool.entries)
        ]
        pool_hash = stable_hash({"pool_version": self.pool.version, "factor_ids": [item["factor_id"] for item in snapshot_factors]})
        self.pool_snapshots.append({
            "snapshot_id": f"snapshot_{stable_hash({'pool_hash': pool_hash, 'valid_unique_evaluations': self.ledger.valid_unique_evaluations})[:20]}",
            "stage": self.stage - 1,
            "optimizer_update": self.updates,
            "admission": asdict(admission),
            "expressions": [entry.expression for entry in self.pool.entries],
            "factors": snapshot_factors,
            "pool_version": self.pool.version,
            "pool_snapshot_hash": pool_hash,
            "train": train_score,
            "valid_unique_evaluations": self.ledger.valid_unique_evaluations,
            "checkpoint": None,
        })
        journal = self.run_dir / "checkpoints/domain_journal.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "optimizer_update": self.updates,
                "pending_spec_hash": spec["spec_hash"],
                "pool_version_before": context.pool_version,
                "pool_version_after": self.pool.version,
                "admission": asdict(admission),
                "ledger": self.ledger.state_dict(),
            }, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        retained = self.pool.hashes
        self.signals = {key: value for key, value in self.signals.items() if key in retained}
        for key, signal in self.signals.items():
            self.signal_cache.put(key, signal, permanent=True)
        self.signal_cache.prune(retained)
        self._event("optimizer_update_committed", admission=asdict(admission), metrics=domain_metrics)
        checkpoint_interval = int(self.effective_config["actor"].get("checkpoint_interval", 50))
        if self.updates % checkpoint_interval == 0:
            paired = session_dir / "checkpoints/verl" / f"global_step_{self.updates}"
            if not (paired / "actor").is_dir():
                raise RuntimeError(f"Verl did not save the expected paired checkpoint {paired}")
            self.checkpoint = paired
            self.checkpoint_fingerprint = {
                "global_step": self.updates,
                "save_lora_only": bool(self.effective_config["actor"].get("save_lora_only", True)),
            }
            self.pool_snapshots[-1]["checkpoint"] = str(paired)
            self.save_checkpoint()
        if self.ledger.exhausted:
            return None
        return self._prepare_online_stage(session_dir)

    def run_cell(self) -> dict[str, Any]:
        """Run all GRPO updates with one Ray/Verl/vLLM lifetime."""
        if self.ledger.exhausted:
            raise RuntimeError("cannot run a completed GRPO cell")
        from .online_dataset import register_online_callback, unregister_online_callback

        session_dir = self.run_dir / "grpo_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        initial_row = self._prepare_online_stage(session_dir)
        # One valid representative per update is the conservative upper bound.
        # Keep dataset length stable across paired-checkpoint resume so Verl's
        # StatefulDataLoader sampler offset remains valid.
        max_updates = min(self.max_training_steps, max(2, self.ledger.limit + 1))
        train_file = session_dir / "online_train.parquet"
        validation_file = session_dir / "validation.parquet"
        pd.DataFrame([initial_row for _ in range(max_updates)]).to_parquet(train_file, index=False)
        pd.DataFrame([initial_row]).to_parquet(validation_file, index=False)
        current_row = initial_row

        def on_batch_end(_: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal current_row
            next_row = self._commit_online_stage(current_row, session_dir)
            if next_row is not None:
                current_row = next_row
            return next_row

        register_online_callback(train_file, on_batch_end)
        metrics_path = session_dir / "verl_metrics.jsonl"
        config = build_verl_grpo_config(
            self.quantevolver_root,
            self.effective_config,
            train_file,
            validation_file,
            session_dir,
            experiment_name=f"cell_seed_{self.seed}",
            resume_from_path=self.checkpoint,
            prompt_groups=1,
            expected_global_step=max(1, self.updates + 1),
            total_training_steps=max_updates,
            reward_function_path=Path(__file__).with_name("verl_reward_function.py"),
            agent_loop_config_path=Path(__file__).with_name("agent_loop_config.yaml"),
            metrics_path=metrics_path,
            online_dataset=True,
        )
        from omegaconf import OmegaConf

        atomic_write_text(session_dir / "effective_verl_config.yaml", OmegaConf.to_yaml(config, resolve=True))
        try:
            trainer_result = run_quant_evolver_verl_trainer(config, expected_global_step=None)
        finally:
            unregister_online_callback(train_file)
        if not self.ledger.exhausted:
            raise RuntimeError("GRPO online dataset reached its safety limit before exhausting the valid-unique budget")
        self.checkpoint = Path(trainer_result["checkpoint"])
        self.checkpoint_fingerprint = {
            "global_step": int(trainer_result["global_step"]),
            "save_lora_only": bool(self.effective_config["actor"].get("save_lora_only", True)),
        }
        if self.pool_snapshots:
            self.pool_snapshots[-1]["checkpoint"] = str(self.checkpoint)
        self.save_checkpoint()
        return {
            "updates": self.updates,
            "checkpoint": str(self.checkpoint),
            "pool_snapshots": list(self.pool_snapshots),
            "metrics_path": str(metrics_path),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "updates": self.updates,
            "groups_in_stage": 0,
            "pool_version": self.pool.version,
            "zero_group_variance": self.zero_group_variance,
            "total_tokens": self.total_tokens,
            "gpu_seconds": self.gpu_seconds,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_hash": self.checkpoint_fingerprint["sha256"] if self.checkpoint_fingerprint else None,
        }

    def save_checkpoint(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint is not None and not (self.checkpoint / "actor").is_dir():
            raise RuntimeError("paired Verl actor checkpoint is missing")
        state = {
            "schema_version": 4,
            "reward_pool_semantics": "independent-availability-same-support-admission-v4",
            "paired_optimizer_step": self.updates,
            "ledger": self.ledger.state_dict(),
            "seen": sorted(self.seen),
            "stage": self.stage,
            "updates": self.updates,
            "event_id": self.event_id,
            "events": self.events,
            "zero_group_variance": self.zero_group_variance,
            "total_tokens": self.total_tokens,
            "gpu_seconds": self.gpu_seconds,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "pool_version": self.pool.version,
            "pool": [{"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata} for item in self.pool.entries],
            "pool_history": self.pool.history,
            "pool_snapshots": self.pool_snapshots,
            "record_count": len(self.records),
        }
        candidates_path = self.run_dir / "candidates.jsonl"
        if self._persisted_record_count < len(self.records):
            with candidates_path.open("a", encoding="utf-8") as handle:
                for record in self.records[self._persisted_record_count :]:
                    handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._persisted_record_count = len(self.records)
        checkpoint_path = self.run_dir / "checkpoint.json"
        write_json(checkpoint_path, state)
        write_json(
            self.run_dir / "checkpoint_commit.json",
            {
                "schema_version": 4,
                "reward_pool_semantics": "independent-availability-same-support-admission-v4",
                "paired_optimizer_step": self.updates,
                "checkpoint": file_fingerprint(checkpoint_path),
                "verl_checkpoint": self.checkpoint_fingerprint,
            },
        )

    def load_checkpoint(self) -> None:
        commit_path = self.run_dir / "checkpoint_commit.json"
        state_path = self.run_dir / "checkpoint.json"
        candidates_path = self.run_dir / "candidates.jsonl"
        if not commit_path.exists():
            raise RuntimeError("GRPO checkpoint is uncommitted and cannot be resumed")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if commit.get("schema_version") != 4 or commit.get("reward_pool_semantics") != "independent-availability-same-support-admission-v4":
            raise RuntimeError("GRPO checkpoint uses incompatible reward/pool semantics")
        if file_fingerprint(state_path)["sha256"] != commit["checkpoint"]["sha256"]:
            raise RuntimeError("GRPO checkpoint hash mismatch")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 4 or state.get("reward_pool_semantics") != "independent-availability-same-support-admission-v4":
            raise RuntimeError("GRPO state uses incompatible reward/pool semantics")
        if int(state["paired_optimizer_step"]) != int(commit["paired_optimizer_step"]):
            raise RuntimeError("model/domain checkpoint step mismatch")
        checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
        fingerprint = state.get("checkpoint_fingerprint")
        if checkpoint is not None and not (checkpoint / "actor").is_dir():
            raise RuntimeError("paired GRPO Verl checkpoint is missing")
        self.ledger = BudgetLedger.from_state_dict(state["ledger"])
        self.seen = set(state["seen"])
        self.stage = int(state["stage"])
        self.updates = int(state["updates"])
        self.event_id = int(state["event_id"])
        self.events = list(state["events"])
        self.zero_group_variance = int(state.get("zero_group_variance", 0))
        self.total_tokens = int(state.get("total_tokens", 0))
        self.gpu_seconds = float(state.get("gpu_seconds", 0.0))
        self.checkpoint = checkpoint
        self.checkpoint_fingerprint = fingerprint
        self.pool_snapshots = list(state.get("pool_snapshots", []))
        self.pool.entries = []
        for item in state["pool"]:
            node = parse_expression(str(item["expression"]))
            signal = self.signal_cache.get(node.expr_hash)
            if signal is None:
                signal = self.evaluator(node)
                self.signal_cache.put(node.expr_hash, signal, permanent=True)
            self.signals[node.expr_hash] = np.asarray(signal)
            self.pool.entries.append(PoolEntry(node.canonical(), node.expr_hash, signal, dict(item.get("metadata") or {})))
        self.pool.version = int(state["pool_version"])
        self.pool.history = list(state["pool_history"])
        self.pool.invalidate_cache()
        all_records = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()] if candidates_path.exists() else []
        self.records = all_records[: int(state.get("record_count", len(all_records)))]
        self._persisted_record_count = len(all_records)
