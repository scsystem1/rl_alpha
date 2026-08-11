from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...dsl.parser import parse_expression
from ...dsl.validity import validate_signal
from ...factors.cache import SignalCache
from ...factors.pool import PoolManager
from ...factors.records import PoolEntry
from ...utils.hashing import directory_fingerprint, file_fingerprint, stable_hash
from ...utils.io import atomic_write_text, write_json
from ..models import BudgetLedger, SearchContext
from ..prompts import build_messages
from .verl_config import build_verl_grpo_config
from .verl_trainer import run_quant_evolver_verl_trainer


class VerlGRPOStageCoordinator:
    """Frozen-pool stage state machine around QuantEvolver/Verl.

    Verl owns rollout, old/reference log probabilities, GRPO advantages, the
    PPO clipped loss, KL/entropy terms, optimizer and distributed model state.
    This coordinator owns only domain prompts, train-only scoring, one admission
    per stage and an outer atomic commit linking pool state to a Verl checkpoint.
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

    def context(self) -> SearchContext:
        score = self.pool._score(self.pool.entries)
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
        score = self.pool._score(self.pool.entries)
        spec = {
            "schema_version": 1,
            "stage": self.stage,
            "expected_samples": int(expected_samples),
            "remaining_budget": self.ledger.remaining,
            "invalid_penalty": float(self.effective_config.get("reward", {}).get("invalid_penalty", -1.0)),
            "reward": self.reward,
            "processed_root": str(self.processed_root.resolve()),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "pool_version": self.pool.version,
            "pool_capacity": self.pool.capacity,
            "min_delta": self.pool.min_delta,
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
            if reason in {"exact_duplicate", "stage_duplicate"}:
                self.ledger.duplicates += 1
            elif reason not in {"ok", "budget_exhausted"}:
                self.ledger.invalid += 1
            if expr_hash and reason not in {"exact_duplicate", "stage_duplicate", "budget_exhausted"}:
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
                validity = validate_signal(signal, self.membership, [entry.signal for entry in self.pool.entries])
                if not validity.valid:
                    raise RuntimeError(f"parent/reward-worker signal validity disagreement for {node.canonical()}")
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
        scored = self.pool.score_candidates(entries)
        archive_by_hash = {str(item["expr_hash"]): item for item in records if item.get("expr_hash")}
        for score in scored:
            archived = archive_by_hash[score.candidate_hash]
            if not math.isclose(float(score.shaped_reward), float(archived["shaped_reward"]), rel_tol=1e-7, abs_tol=1e-8):
                raise RuntimeError(f"parent/reward-worker reward disagreement for {score.candidate_hash}")
            archived_delta = float(archived["delta_objective"])
            if not math.isclose(float(score.delta_objective), archived_delta, rel_tol=1e-7, abs_tol=1e-10):
                raise RuntimeError(f"parent/reward-worker objective disagreement for {score.candidate_hash}")
        self.records.extend(standard_records)
        return entries, scored

    def run_stage(self) -> dict[str, Any]:
        if self.ledger.exhausted:
            raise RuntimeError("cannot run a GRPO stage after the valid-unique budget is exhausted")
        groups = 1
        rollout_n = int(self.effective_config["rollout"]["n"])
        if rollout_n != 8:
            raise ValueError(f"GRPO fairness protocol requires rollout.n=8, got {rollout_n}")
        expected_samples = groups * rollout_n
        attempt_id = self._attempt_id()
        attempt = self.run_dir / "stages" / f"stage_{self.stage:05d}" / attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        archive_path = attempt / "rollouts.jsonl"
        spec_path = attempt / "frozen_stage_spec.json"
        spec = self._stage_spec(archive_path, expected_samples)
        write_json(spec_path, spec)
        context = self.context()
        train_file, validation_file = attempt / "train.parquet", attempt / "validation.parquet"
        self._write_prompts(train_file, context, self.stage, groups, spec_path, spec["spec_hash"], expected_samples, "train")
        self._write_prompts(validation_file, context, self.stage, groups, spec_path, spec["spec_hash"], expected_samples, "validation")
        metrics_path = attempt / "verl_metrics.jsonl"
        expected_step = self.updates + 1
        config = build_verl_grpo_config(
            self.quantevolver_root,
            self.effective_config,
            train_file,
            validation_file,
            attempt,
            experiment_name=f"stage_{self.stage:05d}_{attempt_id}",
            resume_from_path=self.checkpoint,
            prompt_groups=groups,
            expected_global_step=expected_step,
            total_training_steps=self.max_training_steps,
            reward_function_path=Path(__file__).with_name("verl_reward_function.py"),
            # This is a Verl agent-loop registry (a YAML list), not an
            # RLAlpha ProjectConfig fragment, so keep it beside the adapter
            # instead of under the strictly typed project config tree.
            agent_loop_config_path=Path(__file__).with_name("agent_loop_config.yaml"),
            metrics_path=metrics_path,
        )
        from omegaconf import OmegaConf

        atomic_write_text(attempt / "effective_verl_config.yaml", OmegaConf.to_yaml(config, resolve=True))
        started = time.monotonic()
        trainer_result = run_quant_evolver_verl_trainer(config, expected_global_step=expected_step)
        elapsed = time.monotonic() - started
        pending_checkpoint = Path(trainer_result["checkpoint"])
        pending_fingerprint = directory_fingerprint(pending_checkpoint)
        committed_parent = self.run_dir / "checkpoints/verl_committed" / f"stage_{self.stage:05d}_{attempt_id}"
        committed_parent.mkdir(parents=True, exist_ok=False)
        committed_checkpoint = committed_parent / pending_checkpoint.name
        os.replace(pending_checkpoint, committed_checkpoint)
        committed_fingerprint = directory_fingerprint(committed_checkpoint)
        if pending_fingerprint["sha256"] != committed_fingerprint["sha256"]:
            raise RuntimeError("Verl checkpoint changed during atomic commit")
        if not archive_path.exists():
            raise RuntimeError("reward hook did not commit the frozen-stage rollout archive")
        records = [json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != expected_samples:
            raise RuntimeError(f"reward archive has {len(records)} records, expected {expected_samples}")
        records.sort(key=lambda item: (int(item["prompt_group"]), int(item["rollout_index"]), str(item.get("expression") or "")))
        for group in range(groups):
            group_records = [item for item in records if int(item["prompt_group"]) == group]
            self._event("group_complete", prompt_group=group, sample_count=len(group_records), frozen_state_hash=spec["spec_hash"])
        self.updates = expected_step
        self.checkpoint = committed_checkpoint
        self.checkpoint_fingerprint = committed_fingerprint
        self._event("optimizer_update", checkpoint=str(committed_checkpoint), checkpoint_sha256=committed_fingerprint["sha256"])
        entries, scores = self._consume_records(records)
        admission = self.pool.consider_group(entries, scores)
        self._event("pool_update", admission=asdict(admission), pre_stage_pool_version=context.pool_version, post_stage_pool_version=self.pool.version)
        domain_metrics = self._domain_metrics(records, groups)
        self.zero_group_variance += int(domain_metrics["domain/zero_variance_groups"])
        verl_metrics = self._read_verl_metrics(metrics_path)
        required = {"actor/pg_clipfrac", "actor/ppo_kl", "actor/kl_loss", "actor/grad_norm", "actor/lr", "critic/advantages/mean", "actor/entropy"}
        missing = sorted(required - set(verl_metrics))
        if missing:
            raise RuntimeError(f"Verl omitted required formal-GRPO metrics: {missing}")
        stage_metrics = {
            **verl_metrics,
            **domain_metrics,
            "domain/stage_wall_seconds": elapsed,
            "domain/optimizer_update": self.updates,
            "domain/checkpoint_sha256": committed_fingerprint["sha256"],
        }
        write_json(attempt / "stage_metrics.json", stage_metrics)
        self.total_tokens += int(round(float(verl_metrics.get("response_length/mean", 0.0)) * expected_samples))
        self.gpu_seconds += float(verl_metrics.get("timing_s/step", elapsed))
        self.ledger.tokens = self.total_tokens
        self.ledger.gpu_seconds = self.gpu_seconds
        completed_stage = self.stage
        self._event("stage_end", completed_stage=completed_stage, admitted=admission.admitted, metrics_path=str(attempt / "stage_metrics.json"))
        self.stage += 1
        self.save_checkpoint()
        return {"stage": completed_stage, "admission": asdict(admission), "checkpoint": str(committed_checkpoint), "checkpoint_fingerprint": committed_fingerprint, "metrics": stage_metrics, "attempt_dir": str(attempt)}

    def record_validation_event(self, snapshot_id: str, objective: float) -> None:
        self._event("validation_selection", snapshot_id=snapshot_id, validation_objective=float(objective))
        self.save_checkpoint()

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
        if self.checkpoint is not None:
            actual = directory_fingerprint(self.checkpoint)
            if self.checkpoint_fingerprint is None or actual["sha256"] != self.checkpoint_fingerprint["sha256"]:
                raise RuntimeError("outer state points to a changed Verl checkpoint")
        state = {
            "schema_version": 1,
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
        }
        candidates_path = self.run_dir / "candidates.jsonl"
        atomic_write_text(candidates_path, "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in self.records))
        checkpoint_path = self.run_dir / "checkpoint.json"
        write_json(checkpoint_path, state)
        write_json(
            self.run_dir / "checkpoint_commit.json",
            {
                "schema_version": 1,
                "state_hash": stable_hash(state),
                "checkpoint": file_fingerprint(checkpoint_path),
                "candidates": file_fingerprint(candidates_path),
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
        for key, path in (("checkpoint", state_path), ("candidates", candidates_path)):
            if file_fingerprint(path)["sha256"] != commit[key]["sha256"]:
                raise RuntimeError(f"GRPO {key} hash mismatch")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if stable_hash(state) != commit["state_hash"]:
            raise RuntimeError("GRPO outer state hash mismatch")
        checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
        fingerprint = state.get("checkpoint_fingerprint")
        if checkpoint is not None:
            actual = directory_fingerprint(checkpoint)
            if not fingerprint or actual["sha256"] != fingerprint["sha256"] or actual["sha256"] != commit["verl_checkpoint"]["sha256"]:
                raise RuntimeError("GRPO Verl checkpoint hash mismatch")
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
        self.records = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
