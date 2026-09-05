from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlalpha.data.store import PanelStore
from rlalpha.dsl.parser import parse_expression, parse_llm_response
from rlalpha.dsl.validity import validate_signal, validate_signals
from rlalpha.factors.cache import SignalCache
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.factory import (objective_for as _objective, objective_contract, CHECKPOINT_SCHEMA_VERSION, REWARD_POOL_SEMANTICS)
from rlalpha.search.prompts import prompt_contract
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import atomic_write_text


_BATCHES: dict[str, dict[str, Any]] = {}
_LOCKS: dict[int, asyncio.Lock] = {}
_PANELS: dict[tuple[str, str | None, str | None], Any] = {}
_SIGNALS: dict[tuple[tuple[str, str | None, str | None], str], np.ndarray] = {}
_OBJECTIVES: dict[tuple[tuple[str, str | None, str | None], str, str], Any] = {}
_POOLS: dict[tuple[Any, ...], PoolManager] = {}


def _single_blas_thread():
    """Prevent candidate threads from each spawning a full BLAS team."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def _loop_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _LOCKS.setdefault(id(loop), asyncio.Lock())


def _empty_record(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": int(request["extra_info"]["stage"]),
        "prompt_group": int(request["extra_info"]["prompt_group"]),
        "rollout_index": -1,
        "raw_text": str(request["solution_str"]).strip(),
        "expression": None,
        "expr_hash": None,
        "valid": False,
        "reason_code": "unscored",
        "market_evaluated": False,
        "shaped_reward": -1.0,
        "delta_objective": None,
        "delta_add": None,
        "pool_objective_before": None,
        "pool_objective_after": None,
        "replaced_hash": None,
        "saliency": [],
        "eviction_candidates": [],
        "post_prune_delta": None,
        "self_evicted": False,
        "formally_rechecked": False,
        "pool_score": None,
        "add_increment": None,
        "post_prune_increment": None,
        "duplicate_representative_index": None,
        "coverage": None,
        "valid_days": None,
        "variable_day_rate": None,
        "max_pool_correlation": None,
        "mean_abs_daily_corr": None,
        "pooled_correlation": None,
        "mean_abs_daily_rank_corr": None,
        "correlation_coverage": None,
        "reward_valid_days": None,
        "reward_valid_observations": None,
        "reward_valid_day_rate": None,
        "reward_observation_rate": None,
        "reward_scale": None,
        "evaluation_error": None,
        "frozen_state_hash": str(request["extra_info"]["frozen_state_hash"]),
    }


def _load_spec(requests: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {str(item["extra_info"]["stage_spec_path"]) for item in requests}
    if len(paths) != 1:
        raise RuntimeError(f"one reward batch referenced multiple stage specs: {sorted(paths)}")
    path = Path(paths.pop())
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = spec.pop("spec_hash")
    # Process-local output/cache addresses must not make identical fresh/resume
    # frozen semantic states differ.
    hash_payload = {
        key: value for key, value in spec.items()
        if key not in {"archive_path", "signal_cache_root"}
    }
    if stable_hash(hash_payload) != expected_hash:
        raise RuntimeError("frozen GRPO stage spec hash mismatch")
    spec["spec_hash"] = expected_hash
    if spec.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or spec.get("reward_pool_semantics") != REWARD_POOL_SEMANTICS:
        raise RuntimeError("GRPO stage spec uses incompatible reward/pool semantics")
    if len(requests) != int(spec["expected_samples"]):
        raise RuntimeError("reward batch size does not match the frozen stage spec")
    if {str(item["extra_info"]["frozen_state_hash"]) for item in requests} != {expected_hash}:
        raise RuntimeError("rollout metadata does not match the frozen stage spec")
    if {str(item["extra_info"].get("split")) for item in requests} != {"train"}:
        raise RuntimeError("GRPO reward hook accepts train split requests only")
    if {int(item["extra_info"]["stage"]) for item in requests} != {int(spec["stage"])}:
        raise RuntimeError("rollout stage does not match the frozen stage spec")
    if {int(item["extra_info"]["pool_version"]) for item in requests} != {int(spec["pool_version"])}:
        raise RuntimeError("rollout pool version does not match the frozen stage spec")
    if {int(item["extra_info"]["expected_stage_samples"]) for item in requests} != {
        int(spec["expected_samples"])
    }:
        raise RuntimeError("rollout expected-sample count does not match the frozen stage spec")
    return spec


def _score_batch_sync(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = _load_spec(requests)
    panel_key = (str(spec["processed_root"]), spec.get("train_start"), spec.get("train_end"))
    if panel_key not in _PANELS:
        _PANELS[panel_key] = PanelStore(spec["processed_root"]).load_split(
            "train", start=spec.get("train_start"), end=spec.get("train_end")
        )
    panel = _PANELS[panel_key]
    reward_name = str(spec["reward"])
    reward_config = dict(spec.get("reward_config") or {})
    objective_key = (panel_key, reward_name, stable_hash(reward_config))
    objective = _OBJECTIVES.get(objective_key)
    if objective is None:
        objective = _objective(reward_name, panel, reward_config)
        _OBJECTIVES[objective_key] = objective
    if (spec.get("reward_contract") != objective_contract(objective)
            or spec.get("prompt_contract_hash") != prompt_contract()["hash"]):
        raise RuntimeError("frozen reward or prompt contract changed inside the reward worker")
    objective.parallel_workers = int(spec.get("candidate_workers", 1))
    pool_key = (
        panel_key,
        reward_name,
        stable_hash(reward_config),
        int(spec["pool_version"]),
        tuple(str(item["expr_hash"]) for item in spec["pool"]),
        int(spec["pool_capacity"]),
        float(spec["min_delta"]),
        int(spec.get("replacement_top_k", 3)),
        int(spec.get("admission_recheck_top_k", 3)),
    )
    pool = _POOLS.get(pool_key)
    if pool is None:
        pool = PoolManager(
            objective,
            capacity=int(spec["pool_capacity"]),
            min_delta=float(spec["min_delta"]),
            replacement_top_k=int(spec.get("replacement_top_k", 3)),
            admission_recheck_top_k=int(spec.get("admission_recheck_top_k", 3)),
        )
        for item in spec["pool"]:
            node = parse_expression(str(item["expression"]))
            signal_key = (panel_key, node.expr_hash)
            signal = _SIGNALS.get(signal_key)
            if signal is None:
                signal = panel.evaluate(node)
                _SIGNALS[signal_key] = signal
            pool.entries.append(PoolEntry(node.canonical(), node.expr_hash, signal, dict(item.get("metadata") or {})))
        pool.version = int(spec["pool_version"])
        # Materialize the baseline now; subsequent unchanged-pool updates reuse
        # the complete prepared state, including neutralization and moments.
        _ = pool.score
        if len(_POOLS) >= 2:
            _POOLS.pop(next(iter(_POOLS)))
        _POOLS[pool_key] = pool
    baseline = pool.score
    if not math.isclose(float(baseline.objective), float(spec["pool_objective"]), rel_tol=1e-8, abs_tol=1e-10):
        raise RuntimeError("frozen pool objective changed inside the reward worker")
    if (len(baseline.weights) != len(spec["pool_weights"])
            or not np.allclose(baseline.weights, spec["pool_weights"], rtol=1e-8, atol=1e-10)):
        raise RuntimeError("frozen full-train pool weights changed inside the reward worker")

    parsed: list[tuple[Any | None, dict[str, Any]]] = []
    for request in requests:
        record = _empty_record(request)
        try:
            node = parse_llm_response(str(request["solution_str"]).strip())
            record["expression"] = node.canonical()
            record["expr_hash"] = node.expr_hash
        except (TypeError, ValueError):
            node = None
            record["reason_code"] = "parse_or_type_error"
            record["shaped_reward"] = float(spec["invalid_penalty"])
        parsed.append((node, record))

    prior_hashes = set(map(str, spec["seen_hashes"])) | pool.hashes
    order = sorted(
        range(len(parsed)),
        key=lambda index: (
            parsed[index][1]["prompt_group"],
            parsed[index][1]["expression"] or "",
            parsed[index][1]["raw_text"],
            index,
        ),
    )
    rollout_indices: dict[int, int] = {}
    per_group: Counter[int] = Counter()
    for index in order:
        group = int(parsed[index][1]["prompt_group"])
        rollout_indices[index] = per_group[group]
        per_group[group] += 1

    remaining = int(spec["remaining_budget"])
    consumed = 0
    pool_signals = [entry.signal for entry in pool.entries]
    representative: dict[str, int] = {}
    duplicate_aliases: list[tuple[int, int]] = []
    scored_entries: list[PoolEntry] = []
    scored_indices: list[int] = []
    pending_validation: list[tuple[int, Any, np.ndarray]] = []
    for index in order:
        node, record = parsed[index]
        record["rollout_index"] = rollout_indices[index]
        if node is None:
            continue
        expr_hash = str(record["expr_hash"])
        if expr_hash in prior_hashes:
            record["reason_code"] = "exact_duplicate"
            record["shaped_reward"] = -0.5
            continue
        if expr_hash in representative:
            representative_index = representative[expr_hash]
            record["reason_code"] = "intra_group_duplicate_reused"
            record["duplicate_representative_index"] = rollout_indices[representative_index]
            duplicate_aliases.append((index, representative_index))
            continue
        representative[expr_hash] = index
        try:
            signal_key = (panel_key, node.expr_hash)
            signal = _SIGNALS.get(signal_key)
            if signal is None:
                signal = panel.evaluate(node)
                _SIGNALS[signal_key] = signal
        except Exception as exc:
            record["reason_code"] = "evaluation_error"
            record["shaped_reward"] = float(spec["invalid_penalty"])
            record["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            continue
        pending_validation.append((index, node, np.asarray(signal)))

    candidate_workers = min(
        max(1, int(spec.get("candidate_workers", 1))),
        len(pending_validation) or 1,
    )
    # The usual path has enough budget for every unique completion.  Only the
    # final partial group stays serial so an invalid early candidate can hand
    # its unused budget slot to the next candidate exactly as before.
    parallel_validation = candidate_workers > 1 and remaining >= len(pending_validation)

    def validate_one(item: tuple[int, Any, np.ndarray]):
        index, node, signal = item
        try:
            validity = validate_signal(
                signal,
                panel.target(panel.common_mask),
                pool_signals,
                parallel_rank=not parallel_validation,
            )
            return index, node, signal, validity, None
        except Exception as exc:
            return index, node, signal, None, exc

    if parallel_validation:
        with _single_blas_thread(), ThreadPoolExecutor(
            max_workers=candidate_workers,
            thread_name_prefix="reward-validity",
        ) as executor:
            validation_results = list(executor.map(validate_one, pending_validation))
    else:
        try:
            batched_validity = validate_signals(
                [item[2] for item in pending_validation],
                panel.target(panel.common_mask),
                pool_signals,
            )
            validation_results = [
                (index, node, signal, validity, None)
                for (index, node, signal), validity in zip(
                    pending_validation, batched_validity, strict=True
                )
            ]
        except Exception as exc:
            # Preserve the existing per-candidate error records if a malformed
            # array reaches validation; ordinary valid batches never take this
            # diagnostic fallback.
            validation_results = [validate_one(item) for item in pending_validation]
            if validation_results and all(item[-1] is None for item in validation_results):
                raise exc

    for index, node, signal, validity, error in validation_results:
        record = parsed[index][1]
        if consumed >= remaining:
            record["reason_code"] = "budget_exhausted"
            record["shaped_reward"] = 0.0
            continue
        if error is not None:
            record["reason_code"] = "evaluation_error"
            record["shaped_reward"] = float(spec["invalid_penalty"])
            record["evaluation_error"] = f"{type(error).__name__}: {error}"
            continue
        assert validity is not None
        record.update(
            {
                "coverage": validity.coverage,
                "valid_days": validity.valid_days,
                "variable_day_rate": validity.variable_day_rate,
                "max_pool_correlation": validity.max_pool_correlation,
                "mean_abs_daily_corr": validity.mean_abs_daily_corr,
                "pooled_correlation": validity.pooled_correlation,
                "mean_abs_daily_rank_corr": validity.mean_abs_daily_rank_corr,
                "correlation_coverage": validity.correlation_coverage,
            }
        )
        if not validity.valid:
            record["reason_code"] = validity.reason
            record["shaped_reward"] = -0.5 if validity.reason == "near_duplicate_signal" else -0.75
            continue
        consumed += 1
        record["market_evaluated"] = True
        entry = PoolEntry(node.canonical(), node.expr_hash, signal)
        scored_entries.append(entry)
        scored_indices.append(index)

    score_workers = min(candidate_workers, len(scored_entries) or 1)
    with _single_blas_thread():
        candidate_scores = pool.score_candidates(
            scored_entries,
            max_workers=score_workers,
        )
    reward_scale = next(
        (score.reward_scale for score in candidate_scores if score.reward_scale is not None),
        None,
    )
    for _, record in parsed:
        record["reward_scale"] = reward_scale
    for index, candidate_score in zip(scored_indices, candidate_scores, strict=True):
        record = parsed[index][1]
        record.update(
            {
                "valid": bool(candidate_score.valid),
                "reason_code": candidate_score.reason if not candidate_score.valid else "ok",
                "shaped_reward": float(candidate_score.shaped_reward),
                "delta_objective": float(candidate_score.delta_objective),
                "delta_add": float(candidate_score.delta_add),
                "pool_objective_before": float(baseline.objective),
                "pool_objective_after": float(candidate_score.pool_score.objective),
                "replaced_hash": candidate_score.replaced_hash,
                "saliency": list(candidate_score.saliency),
                "eviction_candidates": list(candidate_score.eviction_candidates),
                "post_prune_delta": float(candidate_score.post_prune_delta),
                "self_evicted": bool(candidate_score.self_evicted),
                "formally_rechecked": bool(candidate_score.formally_rechecked),
                "pool_score": asdict(candidate_score.pool_score),
                "add_increment": asdict(candidate_score.add_increment) if candidate_score.add_increment else None,
                "post_prune_increment": asdict(candidate_score.post_prune_increment) if candidate_score.post_prune_increment else None,
                "reward_valid_days": int(candidate_score.reward_valid_days),
                "reward_valid_observations": int(candidate_score.reward_valid_observations),
                "reward_valid_day_rate": float(candidate_score.reward_valid_day_rate),
                "reward_observation_rate": float(candidate_score.reward_observation_rate),
                "reward_scale": candidate_score.reward_scale,
            }
        )

    reused_fields = (
        "valid", "shaped_reward", "delta_objective", "delta_add",
        "pool_objective_before", "pool_objective_after", "replaced_hash",
        "coverage", "valid_days", "variable_day_rate", "max_pool_correlation",
        "mean_abs_daily_corr", "pooled_correlation", "mean_abs_daily_rank_corr",
        "correlation_coverage", "saliency", "eviction_candidates",
        "post_prune_delta", "self_evicted", "formally_rechecked",
        "pool_score", "add_increment", "post_prune_increment",
        "reward_valid_days", "reward_valid_observations",
        "reward_valid_day_rate", "reward_observation_rate",
        "reward_scale",
    )
    for alias_index, representative_index in duplicate_aliases:
        alias = parsed[alias_index][1]
        source = parsed[representative_index][1]
        alias.update({key: source[key] for key in reused_fields})
        alias["market_evaluated"] = False
        alias["reason_code"] = "intra_group_duplicate_reused"

    retained = pool.hashes | {entry.expr_hash for entry in scored_entries}
    for key in list(_SIGNALS):
        if key[0] == panel_key and key[1] not in retained:
            _SIGNALS.pop(key, None)

    # The reward worker and stage coordinator are separate processes.  Commit
    # each market-evaluated signal atomically before publishing the archive so
    # the coordinator can mmap the exact evaluator output instead of evaluating
    # the same AST a second time.  Prune any failed retry first; the coordinator
    # prunes this batch again after admission, bounding storage throughout.
    cache_root = spec.get("signal_cache_root")
    if cache_root:
        shared_cache = SignalCache(cache_root)
        shared_cache.prune(pool.hashes | {entry.expr_hash for entry in scored_entries})
        for entry in scored_entries:
            shared_cache.put(entry.expr_hash, entry.signal, permanent=True)

    records = [record for _, record in parsed]
    archive = Path(spec["archive_path"])
    payload = "".join(json.dumps(records[index], sort_keys=True, default=str) + "\n" for index in order)
    atomic_write_text(archive, payload)
    return records


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Current-Verl custom reward hook with one frozen, batch-synchronous pool.

    Verl invokes the hook per completion.  A single reward worker is configured;
    this coroutine forms an explicit barrier over the complete GRPO rollout batch
    so duplicate handling and valid-unique accounting are deterministic.
    """
    del data_source, ground_truth
    key = str(extra_info["frozen_state_hash"])
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    trigger = False
    async with _loop_lock():
        batch = _BATCHES.setdefault(key, {"requests": [], "futures": [], "started": False})
        if batch["started"]:
            raise RuntimeError("received a late completion after frozen reward scoring started")
        batch["requests"].append({"solution_str": solution_str, "extra_info": dict(extra_info)})
        batch["futures"].append(future)
        expected = int(extra_info["expected_stage_samples"])
        if len(batch["requests"]) > expected:
            raise RuntimeError("received too many completions for one frozen reward batch")
        if len(batch["requests"]) == expected:
            batch["started"] = True
            trigger = True
            requests = list(batch["requests"])
            futures = list(batch["futures"])
    if trigger:
        try:
            # The barrier has already collected every completion, so there is
            # no useful event-loop work left to overlap here.  Running the
            # synchronous batch directly also avoids Python waiting forever
            # for its default executor to shut down after a reward batch (the
            # candidate-level CPU work is already parallelized internally).
            records = _score_batch_sync(requests)
            for waiter, record in zip(futures, records, strict=True):
                waiter.set_result(record)
        except Exception as exc:
            for waiter in futures:
                if not waiter.done():
                    waiter.set_exception(exc)
        finally:
            _BATCHES.pop(key, None)
    record = await future
    score = float(record["shaped_reward"])
    if not math.isfinite(score):
        score = -1.0
    # Verl turns every additional return key into a non-tensor batch column.
    # Variable-length diagnostics such as saliency (0 or pool_size + 1) and
    # daily_ic then become arrays with incompatible second dimensions when its
    # agent workers are concatenated.  The complete records are already
    # durably written to the stage archive and consumed from there after the
    # optimizer update, so the training transport must carry only the scalar
    # reward.  This also avoids sending thousands of daily-IC values through
    # Ray eight times per update.
    return {"score": float(np.clip(score, -1.0, 1.0))}
