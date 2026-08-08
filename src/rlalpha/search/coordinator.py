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
from ..utils.io import atomic_write_text, write_json
from .base import Searcher
from .models import BudgetLedger, CandidateOutcome, SearchContext


class SearchCoordinator:
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

    def context(self) -> SearchContext:
        score = self.pool._score(self.pool.entries)
        context = SearchContext(self.pool.version, tuple(item.expression for item in self.pool.entries), tuple(score.weights), score.objective, self.ledger.valid_unique_evaluations, self.ledger.limit, tuple(self.records[-64:]))
        assert_train_only_context(context.to_prompt_dict())
        return context

    def run_group(self, group_size: int = 8) -> list[CandidateOutcome]:
        candidates = self.searcher.propose(self.context(), group_size)
        self.ledger.raw_proposals += len(candidates)
        outcomes: list[CandidateOutcome] = []
        entries: list[PoolEntry] = []
        entry_candidates = []
        for candidate in candidates:
            if candidate.node is None:
                self.ledger.invalid += 1
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "parse_or_type_error", False, shaped_reward=-1.0))
                continue
            rescore = candidate.generator == "gp_rescore" and candidate.expr_hash in self.seen
            if (candidate.expr_hash in self.seen or candidate.expr_hash in self.pool.hashes) and not rescore:
                self.ledger.duplicates += 1
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "exact_duplicate", False, shaped_reward=-0.5))
                continue
            if self.ledger.exhausted and not rescore:
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "budget_exhausted", False))
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
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, "evaluation_error", False, metadata={"error": str(exc)}))
                continue
            if not validity.valid:
                self.ledger.invalid += 1
                penalty = -0.5 if validity.reason == "near_duplicate_signal" else -0.75
                outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, False, validity.reason, False, shaped_reward=penalty, metadata={"coverage": validity.coverage}))
                continue
            if not rescore:
                self.ledger.valid_unique_evaluations += 1
            self.signals[candidate.expr_hash] = signal
            self.signal_cache.put(candidate.expr_hash, signal)
            entry = PoolEntry(candidate.expression, candidate.expr_hash, signal, {"generator": candidate.generator, "parents": candidate.parents})
            entries.append(entry)
            entry_candidates.append(candidate)
            outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, True, "ok", True, metadata={"coverage": validity.coverage, "wall_seconds": time.monotonic() - started, "rescore": rescore}))
        scored_entries = self.pool.score_candidates(entries)
        scores = {score.candidate_hash: score for score in scored_entries}
        outcomes = [CandidateOutcome(item.expr_hash, item.expression, item.valid, item.reason, item.market_evaluated, scores[item.expr_hash].delta_objective, scores[item.expr_hash].shaped_reward, item.metadata) if item.expr_hash in scores else item for item in outcomes]
        self.pending_entries.extend(entries)
        self.pending_scores.extend(scored_entries)
        self.groups_since_admission += 1
        interval = int(getattr(self.searcher, "admission_group_interval", 1))
        if self.groups_since_admission >= interval:
            self.flush_admission()
        self.searcher.observe(outcomes)
        self.ledger.tokens = int(getattr(self.searcher, "total_tokens", self.ledger.tokens))
        self.ledger.gpu_seconds = float(getattr(self.searcher, "gpu_seconds", self.ledger.gpu_seconds))
        retained = self.pool.hashes | {item.expr_hash for item in self.pending_entries} | set(getattr(self.searcher, "retained_hashes", set()))
        self.signals = {key: value for key, value in self.signals.items() if key in retained}
        for key, signal in self.signals.items():
            self.signal_cache.put(key, signal, permanent=True)
        self.records.extend(item.to_dict() for item in outcomes)
        if self.run_dir is not None:
            self.save_checkpoint()
        return outcomes

    def flush_admission(self) -> None:
        if self.pending_entries:
            self.pool.consider_group(self.pending_entries, self.pending_scores)
        self.pending_entries = []
        self.pending_scores = []
        self.groups_since_admission = 0

    def save_checkpoint(self) -> None:
        assert self.run_dir is not None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = {"ledger": self.ledger.state_dict(), "seen": sorted(self.seen), "searcher": self.searcher.state_dict(), "pool_version": self.pool.version, "pool": [{"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata} for item in self.pool.entries], "pool_history": self.pool.history, "pending_entries": [{"expression": item.expression, "expr_hash": item.expr_hash, "metadata": item.metadata} for item in self.pending_entries], "groups_since_admission": self.groups_since_admission}
        write_json(self.run_dir / "checkpoint.json", state)
        text = "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in self.records)
        atomic_write_text(self.run_dir / "candidates.jsonl", text)

    def load_checkpoint(self) -> None:
        if self.run_dir is None:
            raise ValueError("run_dir is required")
        with (self.run_dir / "checkpoint.json").open(encoding="utf-8") as handle:
            state = json.load(handle)
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
        self.pool.history = list(state["pool_history"])
        self.pending_entries = [restored_entry(item) for item in state.get("pending_entries", [])]
        self.pending_scores = self.pool.score_candidates(self.pending_entries)
        self.groups_since_admission = int(state.get("groups_since_admission", 0))
        candidate_path = self.run_dir / "candidates.jsonl"
        self.records = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()] if candidate_path.exists() else []
