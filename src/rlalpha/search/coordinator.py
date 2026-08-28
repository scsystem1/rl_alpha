from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..dsl.validity import validate_signal
from ..dsl.parser import parse_expression
from ..factors.pool import PoolManager
from ..factors.cache import SignalCache
from ..factors.records import CandidateScore, PoolEntry
from ..leakage.guards import assert_train_only_context
from ..utils.io import write_json
from ..utils.hashing import file_fingerprint, stable_hash
from .base import Searcher
from .models import BudgetLedger, CandidateOutcome, SearchContext


class SearchCoordinator:
    CHECKPOINT_SCHEMA_VERSION = 6
    REWARD_POOL_SEMANTICS = "fixed-universe-zero-fill-psd-gram-v6"

    def __init__(self, searcher: Searcher, pool: PoolManager, evaluator: Callable[[object], np.ndarray], membership: np.ndarray, budget: int, run_dir: str | Path | None = None):
        self.searcher = searcher
        self.pool = pool
        self.evaluator = evaluator
        self.membership = np.asarray(membership, dtype=bool)
        self.ledger = BudgetLedger(budget)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.signal_cache = SignalCache(None if self.run_dir is None else self.run_dir / "cache/signals")
        self.seen: set[str] = set()
        self.signals: dict[str, np.ndarray] = {}
        self.records: list[dict[str, object]] = []
        self.pending_entries: list[PoolEntry] = []
        self.pending_scores: list[CandidateScore] = []
        self.groups_since_admission = 0
        self.group_index = 0
        self._persisted_record_count = 0

    def context(self) -> SearchContext:
        score = self.pool.score
        context = SearchContext(self.pool.version, tuple(item.expression for item in self.pool.entries), tuple(score.weights), score.objective, self.ledger.valid_unique_evaluations, self.ledger.limit, tuple(self.records[-64:]))
        assert_train_only_context(context.to_prompt_dict())
        return context

    def run_group(self, group_size: int = 8) -> list[CandidateOutcome]:
        context = self.context()
        pre_group_pool_hash = stable_hash({"pool_version": self.pool.version, "factors": [entry.expr_hash for entry in self.pool.entries]})
        candidates = self.searcher.propose(context, group_size)
        group_index = self.group_index
        self.group_index += 1
        raw_start = self.ledger.raw_proposals
        self.ledger.raw_proposals += len(candidates)
        outcomes: list[CandidateOutcome] = []
        entries: list[PoolEntry] = []
        entry_candidates = []
        for candidate_index, candidate in enumerate(candidates):
            raw_proposal_index = raw_start + candidate_index
            proposal_id = f"proposal_{stable_hash({'run_dir': str(self.run_dir), 'group_index': group_index, 'raw_proposal_index': raw_proposal_index, 'generator': candidate.generator, 'raw_text': candidate.raw_text})[:24]}"
            base_metadata = {
                "proposal_id": proposal_id,
                "factor_id": candidate.expr_hash if candidate.node is not None else None,
                "generator": candidate.generator,
                "parents": list(candidate.parents),
                "raw_text": candidate.raw_text,
                "group_index": group_index,
                "raw_proposal_index": raw_proposal_index,
                "pre_group_pool_version": context.pool_version,
                "pre_group_pool_snapshot_hash": pre_group_pool_hash,
            }
            if candidate.node is None:
                self.ledger.invalid += 1
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "parse_or_type_error", False, shaped_reward=-1.0, metadata=base_metadata))
                continue
            if candidate.expr_hash in self.seen or candidate.expr_hash in self.pool.hashes:
                self.ledger.duplicates += 1
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "exact_duplicate", False, shaped_reward=-0.5, metadata=base_metadata))
                continue
            if self.ledger.exhausted:
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "budget_exhausted", False, metadata=base_metadata))
                continue
            self.seen.add(candidate.expr_hash)
            started = time.monotonic()
            try:
                signal = self.signals.get(candidate.expr_hash)
                if signal is None:
                    signal = self.signal_cache.get(candidate.expr_hash)
                if signal is None:
                    signal = self.evaluator(candidate.node)
                validity = validate_signal(signal, self.membership, [entry.signal for entry in self.pool.entries])
            except Exception as exc:
                self.ledger.invalid += 1
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "evaluation_error", False, metadata={**base_metadata, "error": str(exc)}))
                continue
            if not validity.valid:
                self.ledger.invalid += 1
                penalty = -0.5 if validity.reason == "near_duplicate_signal" else -0.75
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, validity.reason, False, shaped_reward=penalty, metadata={**base_metadata, "coverage": validity.coverage, "redundancy": {"mean_abs_daily_corr": validity.mean_abs_daily_corr, "pooled_correlation": validity.pooled_correlation, "mean_abs_daily_rank_corr": validity.mean_abs_daily_rank_corr, "correlation_coverage": validity.correlation_coverage}}))
                continue
            self.ledger.valid_unique_evaluations += 1
            self.signals[candidate.expr_hash] = signal
            self.signal_cache.put(candidate.expr_hash, signal)
            entry = PoolEntry(candidate.expression, candidate.expr_hash, signal, base_metadata)
            entries.append(entry)
            entry_candidates.append(candidate)
            outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, True, "ok", True, metadata={**base_metadata, "coverage": validity.coverage, "wall_seconds": time.monotonic() - started, "redundancy": {"mean_abs_daily_corr": validity.mean_abs_daily_corr, "pooled_correlation": validity.pooled_correlation, "mean_abs_daily_rank_corr": validity.mean_abs_daily_rank_corr, "correlation_coverage": validity.correlation_coverage}}))
        scored_entries = self.pool.score_candidates(entries)
        scores = {score.candidate_hash: score for score in scored_entries}
        outcomes = [
            CandidateOutcome(
                item.expr_hash,
                item.expression,
                item.valid and scores[item.expr_hash].valid,
                item.reason if scores[item.expr_hash].valid else scores[item.expr_hash].reason,
                item.market_evaluated,
                scores[item.expr_hash].delta_objective,
                scores[item.expr_hash].shaped_reward,
                {
                    **item.metadata,
                    "delta_add": scores[item.expr_hash].delta_add,
                    "post_prune_delta": scores[item.expr_hash].post_prune_delta,
                    "saliency": list(scores[item.expr_hash].saliency),
                    "eviction_candidates": list(scores[item.expr_hash].eviction_candidates),
                    "replaced_hash": scores[item.expr_hash].replaced_hash,
                    "self_evicted": scores[item.expr_hash].self_evicted,
                    "formally_rechecked": scores[item.expr_hash].formally_rechecked,
                    "positive_not_admitted": scores[item.expr_hash].positive_not_admitted,
                    "reward_valid_days": scores[item.expr_hash].reward_valid_days,
                    "reward_valid_observations": scores[item.expr_hash].reward_valid_observations,
                    "reward_valid_day_rate": scores[item.expr_hash].reward_valid_day_rate,
                    "reward_observation_rate": scores[item.expr_hash].reward_observation_rate,
                },
            )
            if item.expr_hash in scores
            else item
            for item in outcomes
        ]
        self.pending_entries.extend(entries)
        self.pending_scores.extend(scored_entries)
        self.groups_since_admission += 1
        interval = int(getattr(self.searcher, "admission_group_interval", 1))
        admission = None
        if self.groups_since_admission >= interval:
            admission = self.flush_admission()
        if admission is not None:
            outcomes = [
                CandidateOutcome(
                    item.expr_hash,
                    item.expression,
                    item.valid,
                    item.reason,
                    item.market_evaluated,
                    item.delta_objective,
                    item.shaped_reward,
                    {
                        **item.metadata,
                        "admitted": bool(admission.admitted and admission.candidate_hash == item.expr_hash),
                        "positive_not_admitted": bool(
                            item.delta_objective > 0
                            and not (admission.admitted and admission.candidate_hash == item.expr_hash)
                        ),
                    },
                )
                for item in outcomes
            ]
        self.searcher.observe(outcomes)
        self.ledger.tokens = int(getattr(self.searcher, "total_tokens", self.ledger.tokens))
        self.ledger.gpu_seconds = float(getattr(self.searcher, "gpu_seconds", self.ledger.gpu_seconds))
        retained = self.pool.hashes | {item.expr_hash for item in self.pending_entries}
        self.signals = {key: value for key, value in self.signals.items() if key in retained}
        for key, signal in self.signals.items():
            self.signal_cache.put(key, signal, permanent=True)
        self.records.extend(item.to_dict() for item in outcomes)
        if self.run_dir is not None:
            self.save_checkpoint()
        return outcomes

    def flush_admission(self):
        admission = None
        if self.pending_entries:
            admission = self.pool.consider_group(self.pending_entries, self.pending_scores)
        self.pending_entries = []
        self.pending_scores = []
        self.groups_since_admission = 0
        return admission

    def save_checkpoint(self) -> None:
        assert self.run_dir is not None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = {"schema_version": self.CHECKPOINT_SCHEMA_VERSION, "reward_pool_semantics": self.REWARD_POOL_SEMANTICS, "ledger": self.ledger.state_dict(), "seen": sorted(self.seen), "searcher": self.searcher.state_dict(), "pool_version": self.pool.version, "pool": [{"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata} for item in self.pool.entries], "pool_history": self.pool.history, "pending_entries": [{"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata} for item in self.pending_entries], "groups_since_admission": self.groups_since_admission, "group_index": self.group_index}
        write_json(self.run_dir / "checkpoint.json", state)
        if self._persisted_record_count < len(self.records):
            candidate_path = self.run_dir / "candidates.jsonl"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            with candidate_path.open("a", encoding="utf-8") as handle:
                for record in self.records[self._persisted_record_count :]:
                    handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()
                __import__("os").fsync(handle.fileno())
            self._persisted_record_count = len(self.records)
        checkpoint_record = file_fingerprint(self.run_dir / "checkpoint.json")
        write_json(self.run_dir / "checkpoint_commit.json", {"schema_version": self.CHECKPOINT_SCHEMA_VERSION, "reward_pool_semantics": self.REWARD_POOL_SEMANTICS, "checkpoint": checkpoint_record})
        retained = self.pool.hashes | {item.expr_hash for item in self.pending_entries}
        self.signal_cache.prune(retained)

    def load_checkpoint(self) -> None:
        if self.run_dir is None:
            raise ValueError("run_dir is required")
        commit_path = self.run_dir / "checkpoint_commit.json"
        if not commit_path.exists():
            raise RuntimeError("checkpoint is legacy/uncommitted and cannot be resumed")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if commit.get("schema_version") != self.CHECKPOINT_SCHEMA_VERSION or commit.get("reward_pool_semantics") != self.REWARD_POOL_SEMANTICS:
            raise RuntimeError("checkpoint uses incompatible reward/pool semantics; start a new experiment ID")
        actual = file_fingerprint(self.run_dir / "checkpoint.json")
        if actual["sha256"] != commit.get("checkpoint", {}).get("sha256"):
            raise RuntimeError("checkpoint hash mismatch")
        with (self.run_dir / "checkpoint.json").open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("schema_version") != self.CHECKPOINT_SCHEMA_VERSION or state.get("reward_pool_semantics") != self.REWARD_POOL_SEMANTICS:
            raise RuntimeError("checkpoint state uses incompatible reward/pool semantics")
        self.ledger = BudgetLedger.from_state_dict(state["ledger"])
        self.seen = set(state["seen"])
        self.searcher.load_state_dict(state["searcher"])
        def restored_entry(item: dict[str, object]) -> PoolEntry:
            expression, expr_hash = str(item["expression"]), str(item["expr_hash"])
            signal = self.signal_cache.get(expr_hash)
            if signal is None:
                signal = self.evaluator(parse_expression(expression))
                self.signal_cache.put(expr_hash, signal, permanent=True)
            self.signals[expr_hash] = np.asarray(signal)
            return PoolEntry(expression, expr_hash, signal, item.get("metadata", {}))

        self.pool.entries = [restored_entry(item) for item in state["pool"]]
        self.pool.version = int(state["pool_version"])
        self.pool.invalidate_cache()
        self.pool.history = list(state["pool_history"])
        self.pending_entries = [restored_entry(item) for item in state.get("pending_entries", [])]
        self.pending_scores = self.pool.score_candidates(self.pending_entries)
        self.groups_since_admission = int(state.get("groups_since_admission", 0))
        self.group_index = int(state.get("group_index", 0))
        candidate_path = self.run_dir / "candidates.jsonl"
        self.records = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()] if candidate_path.exists() else []
        self._persisted_record_count = len(self.records)
