from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlalpha.data.store import PanelStore
from rlalpha.dsl.parser import parse_expression, parse_llm_response
from rlalpha.dsl.validity import validate_signal
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.r0 import R0Objective
from rlalpha.rewards.r1 import R1Objective
from rlalpha.rewards.r2_lcb import R2LCBObjective
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import atomic_write_text


_BATCHES: dict[str, dict[str, Any]] = {}
_LOCKS: dict[int, asyncio.Lock] = {}


def _loop_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _LOCKS.setdefault(id(loop), asyncio.Lock())


def _objective(name: str, panel):
    label = panel.target(panel.label)
    mask = panel.target(panel.common_mask) & pd.notna(label)
    if name == "r0":
        return R0Objective(label, mask)
    exposures = panel.target(panel.exposures)
    if name == "r1":
        return R1Objective(label, mask, exposures)
    if name == "r2_lcb":
        return R2LCBObjective(label, mask, exposures, hac_lag=20)
    raise ValueError(f"unknown reward {name!r}")


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
        "pool_objective_before": None,
        "pool_objective_after": None,
        "replaced_hash": None,
        "coverage": None,
        "valid_days": None,
        "variable_day_rate": None,
        "max_pool_correlation": None,
        "mean_abs_daily_corr": None,
        "pooled_correlation": None,
        "mean_abs_daily_rank_corr": None,
        "correlation_coverage": None,
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
    # ``archive_path`` is deliberately excluded: it is a process-local output
    # address and must not make identical fresh/resume frozen states differ.
    hash_payload = {key: value for key, value in spec.items() if key != "archive_path"}
    if stable_hash(hash_payload) != expected_hash:
        raise RuntimeError("frozen GRPO stage spec hash mismatch")
    spec["spec_hash"] = expected_hash
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
    panel = PanelStore(spec["processed_root"]).load_split(
        "train", start=spec.get("train_start"), end=spec.get("train_end")
    )
    objective = _objective(str(spec["reward"]), panel)
    pool = PoolManager(objective, capacity=int(spec["pool_capacity"]), min_delta=float(spec["min_delta"]))
    for item in spec["pool"]:
        node = parse_expression(str(item["expression"]))
        signal = panel.evaluate(node)
        pool.entries.append(PoolEntry(node.canonical(), node.expr_hash, signal, dict(item.get("metadata") or {})))
    pool.version = int(spec["pool_version"])
    baseline = pool._score(pool.entries)
    if not math.isclose(float(baseline.objective), float(spec["pool_objective"]), rel_tol=1e-8, abs_tol=1e-10):
        raise RuntimeError("frozen pool objective changed inside the reward worker")

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

    counts = Counter(record["expr_hash"] for node, record in parsed if node is not None)
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
        if counts[expr_hash] > 1:
            record["reason_code"] = "stage_duplicate"
            record["shaped_reward"] = -0.5
            continue
        if consumed >= remaining:
            record["reason_code"] = "budget_exhausted"
            record["shaped_reward"] = 0.0
            continue
        try:
            signal = panel.evaluate(node)
            validity = validate_signal(signal, panel.target(panel.common_mask), pool_signals)
        except Exception as exc:
            record["reason_code"] = "evaluation_error"
            record["shaped_reward"] = float(spec["invalid_penalty"])
            record["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            continue
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
        candidate_score = pool.score_candidates([entry])[0]
        record.update(
            {
                "valid": bool(candidate_score.valid),
                "reason_code": candidate_score.reason if not candidate_score.valid else "ok",
                "shaped_reward": float(candidate_score.shaped_reward),
                "delta_objective": float(candidate_score.delta_objective),
                "pool_objective_before": float(baseline.objective),
                "pool_objective_after": float(candidate_score.pool_score.objective),
                "replaced_hash": candidate_score.replaced_hash,
            }
        )

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
    so duplicate handling and valid-unique budgeting are deterministic.
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
            records = await asyncio.to_thread(_score_batch_sync, requests)
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
    return {"score": float(np.clip(score, -1.0, 1.0)), **record}
