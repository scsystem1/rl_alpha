from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rlalpha.data.store import PanelStore
from rlalpha.dsl.ast import Call, Constant, Feature, Node, Window
from rlalpha.dsl.parser import parse_llm_response
from rlalpha.dsl.validity import _daily_rank_corr_exact_serial, validate_signals
from rlalpha.factors.cache import SignalCache
from rlalpha.utils.hashing import stable_hash


_BATCHES: dict[str, dict[str, Any]] = {}
_LOCKS: dict[int, asyncio.Lock] = {}
_PANELS: dict[tuple[str, str, str], Any] = {}
_SIGNALS: dict[tuple[tuple[str, str, str], str], np.ndarray] = {}
_HISTORY: dict[str, list[dict[str, Any]]] = {}


def _loop_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _LOCKS.setdefault(id(loop), asyncio.Lock())


def _family_repr(node: Node) -> str:
    if isinstance(node, Feature):
        return node.name
    if isinstance(node, Window):
        return "W"
    if isinstance(node, Constant):
        return "C"
    if isinstance(node, Call):
        return f"{node.operator}({','.join(_family_repr(arg) for arg in node.args)})"
    raise TypeError(type(node).__name__)


def _load_spec(requests: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {str(item["extra_info"]["spec_path"]) for item in requests}
    if len(paths) != 1:
        raise RuntimeError("one QuantEvolver reward batch referenced multiple specs")
    path = Path(paths.pop())
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected = str(spec.pop("spec_hash"))
    if stable_hash(spec) != expected:
        raise RuntimeError("QuantEvolver reward spec hash mismatch")
    spec["spec_hash"] = expected
    if spec.get("schema_version") != 1 or spec.get("reward_semantics") != "quantevolver-dico-rankic-v1":
        raise RuntimeError("incompatible QuantEvolver reward semantics")
    if len(requests) != int(spec["rollout_n"]):
        raise RuntimeError("QuantEvolver reward batch size mismatch")
    if len({int(item["extra_info"]["round"]) for item in requests}) != 1:
        raise RuntimeError("one reward batch mixed QuantEvolver rounds")
    return spec


def _history(spec: dict[str, Any]) -> list[dict[str, Any]]:
    key = str(spec["history_path"])
    if key not in _HISTORY:
        path = Path(key)
        _HISTORY[key] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if path.exists() else []
    return _HISTORY[key]


def _signal(panel_key: tuple[str, str, str], panel: Any, node: Node) -> np.ndarray:
    key = (panel_key, node.expr_hash)
    values = _SIGNALS.get(key)
    if values is None:
        values = np.asarray(panel.evaluate(node), dtype=float)
        _SIGNALS[key] = values
    return values


def _rankic(signal: np.ndarray, label: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    support = mask & np.isfinite(signal) & np.isfinite(label)
    valid_days = support.sum(axis=1) >= 3
    daily = _daily_rank_corr_exact_serial(signal, label, mask, valid_days)
    finite = daily[np.isfinite(daily)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    icir = float(mean / (np.std(finite) + 1e-8) * np.sqrt(len(finite)))
    return mean, icir, int(len(finite))


def _elite_signals(
    history: list[dict[str, Any]],
    panel_key: tuple[str, str, str],
    panel: Any,
    top_k: int,
) -> list[np.ndarray]:
    best: dict[str, dict[str, Any]] = {}
    for item in history:
        if not item.get("mined") or not item.get("expression"):
            continue
        expression = str(item["expression"])
        if expression not in best or float(item.get("raw_reward", -1.0)) > float(best[expression].get("raw_reward", -1.0)):
            best[expression] = item
    selected = sorted(best.values(), key=lambda item: float(item.get("raw_reward", -1.0)), reverse=True)[:top_k]
    signals = []
    for item in selected:
        try:
            node = parse_llm_response(f"<expr>{item['expression']}</expr>")
            signals.append(_signal(panel_key, panel, node))
        except Exception:
            continue
    return signals


def _score_batch_sync(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = _load_spec(requests)
    extra = requests[0]["extra_info"]
    start, end = str(extra["start_date"]), str(extra["end_date"])
    # Load the audited train panel once per persistent reward worker.  Regime
    # tasks differ only in their metric mask; reloading four overlapping Zarr
    # panels would multiply both startup latency and resident memory.
    panel_key = (str(spec["processed_root"]), "train", "full")
    panel = _PANELS.get(panel_key)
    if panel is None:
        panel = PanelStore(spec["processed_root"]).load_split("train")
        _PANELS[panel_key] = panel
    dates = panel.target_dates
    regime = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    mask = panel.target(panel.common_mask) & regime[:, None]
    label = panel.target(panel.label)
    history = _history(spec)
    prior_seen = {str(item.get("expr_hash")) for item in history if item.get("expr_hash")}
    family_counts = Counter(str(item.get("family_hash")) for item in history if item.get("family_hash"))
    elite = _elite_signals(history, panel_key, panel, int(spec["elite_top_k"]))

    parsed: list[tuple[Node | None, dict[str, Any]]] = []
    for rollout_index, request in enumerate(requests):
        record: dict[str, Any] = {
            "round": int(request["extra_info"]["round"]),
            "rollout_index": int(rollout_index),
            "task_id": str(request["extra_info"]["task_id"]),
            "seed_id": str(request["extra_info"]["seed_id"]),
            "seed_expr": str(request["extra_info"]["seed_expr"]),
            "family": str(request["extra_info"]["family"]),
            "time_split": str(request["extra_info"]["time_split"]),
            "start_date": start,
            "end_date": end,
            "raw_text": str(request["solution_str"]).strip(),
            "expression": None,
            "expr_hash": None,
            "family_hash": None,
            "valid": False,
            "reason_code": "unscored",
            "market_evaluated": False,
            "mean_rank_ic": None,
            "rank_ic_ir": None,
            "valid_times": 0,
            "coverage": 0.0,
            "behavior_max_corr": 0.0,
            "raw_reward": -1.0,
            "exact_repeat_penalty": 0.0,
            "family_repeat_penalty": 0.0,
            "family_new_bonus": 0.0,
            "behavior_penalty": 0.0,
            "behavior_bonus": 0.0,
            "shaped_reward": float(spec["invalid_penalty"]),
            "mined": False,
        }
        try:
            node = parse_llm_response(record["raw_text"])
            record["expression"] = node.canonical()
            record["expr_hash"] = node.expr_hash
            record["family_hash"] = stable_hash(_family_repr(node))
        except (TypeError, ValueError):
            node = None
            record["reason_code"] = "parse_or_type_error"
        parsed.append((node, record))

    candidate_nodes: list[Node] = []
    candidate_indices: list[int] = []
    candidate_signals: list[np.ndarray] = []
    for index, (node, record) in enumerate(parsed):
        if node is None:
            continue
        try:
            candidate_nodes.append(node)
            candidate_indices.append(index)
            candidate_signals.append(_signal(panel_key, panel, node))
        except Exception as exc:
            record["reason_code"] = "evaluation_error"
            record["evaluation_error"] = f"{type(exc).__name__}: {exc}"

    validity = validate_signals(candidate_signals, mask, elite, max_pool_corr=2.0)
    for node, index, signal, valid in zip(candidate_nodes, candidate_indices, candidate_signals, validity, strict=True):
        record = parsed[index][1]
        record["coverage"] = float(valid.coverage)
        record["behavior_max_corr"] = float(max(valid.max_pool_correlation, valid.mean_abs_daily_rank_corr))
        if not valid.valid:
            record["reason_code"] = str(valid.reason)
            continue
        mean_rank_ic, rank_ic_ir, valid_times = _rankic(signal, label, mask)
        if not np.isfinite(mean_rank_ic):
            record["reason_code"] = "non_finite_rankic"
            continue
        raw_reward = float(np.clip(mean_rank_ic * 5.0 + 0.02 * np.tanh(rank_ic_ir), -1.0, 1.0))
        exact_penalty = float(spec["exact_repeat_penalty"]) if node.expr_hash in prior_seen else 0.0
        family_count = int(family_counts.get(str(record["family_hash"]), 0))
        family_bonus = float(spec["family_new_bonus"]) if family_count == 0 and raw_reward >= float(spec["family_good_new_threshold"]) else 0.0
        family_penalty = float(spec["family_repeat_penalty"]) if family_count > int(spec["family_free_quota"]) and raw_reward < float(spec["family_low_quality_threshold"]) else 0.0
        behavior_penalty = 0.0
        behavior_bonus = 0.0
        behavior_corr = float(record["behavior_max_corr"])
        if elite and raw_reward >= float(spec["behavior_good_score_threshold"]):
            if behavior_corr > float(spec["behavior_corr_threshold"]):
                behavior_penalty = float(spec["behavior_corr_penalty"]) * (behavior_corr - float(spec["behavior_corr_threshold"]))
            elif behavior_corr < float(spec["behavior_low_corr_threshold"]):
                behavior_bonus = float(spec["behavior_low_corr_bonus"])
        shaped = float(np.clip(raw_reward - exact_penalty - family_penalty + family_bonus - behavior_penalty + behavior_bonus, -1.0, 1.0))
        mined = bool(mean_rank_ic >= float(spec["mined_rank_ic_threshold"]) and valid.coverage >= float(spec["mined_coverage_threshold"]) and node.expr_hash not in prior_seen)
        record.update({
            "valid": True,
            "reason_code": "ok",
            "market_evaluated": True,
            "mean_rank_ic": mean_rank_ic,
            "rank_ic_ir": rank_ic_ir,
            "valid_times": valid_times,
            "raw_reward": raw_reward,
            "exact_repeat_penalty": exact_penalty,
            "family_repeat_penalty": family_penalty,
            "family_new_bonus": family_bonus,
            "behavior_penalty": behavior_penalty,
            "behavior_bonus": behavior_bonus,
            "shaped_reward": shaped,
            "mined": mined,
        })

    records = [record for _, record in parsed]
    history.extend(records)
    history_path = Path(spec["history_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    cache = SignalCache(spec["signal_cache_root"])
    for node, signal in zip(candidate_nodes, candidate_signals, strict=True):
        cache.put(node.expr_hash, signal, permanent=True)
    return records


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, float]:
    del data_source, ground_truth
    key = str(extra_info["reward_batch_key"])
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    trigger = False
    async with _loop_lock():
        batch = _BATCHES.setdefault(key, {"requests": [], "futures": [], "started": False})
        if batch["started"]:
            raise RuntimeError("late QuantEvolver completion after reward scoring started")
        batch["requests"].append({"solution_str": solution_str, "extra_info": dict(extra_info)})
        batch["futures"].append(future)
        expected = int(extra_info["expected_samples"])
        if len(batch["requests"]) > expected:
            raise RuntimeError("too many completions for one QuantEvolver reward batch")
        if len(batch["requests"]) == expected:
            batch["started"] = True
            trigger = True
            requests = list(batch["requests"])
            futures = list(batch["futures"])
    if trigger:
        try:
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
    return {"score": float(np.clip(score if math.isfinite(score) else -1.0, -1.0, 1.0))}
