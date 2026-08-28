from __future__ import annotations

"""Compare the legacy and optimized validity kernels on a retained real pool."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from rlalpha.config import load_paths
from rlalpha.data.store import PanelStore
from rlalpha.dsl.validity import ValidityResult, _daily_rank_corr_exact, validate_signal
from rlalpha.factors.calculator import daily_corr
from rlalpha.utils.numerics import finite_corr


def legacy_validate(signal: np.ndarray, membership: np.ndarray, pools: list[np.ndarray]) -> ValidityResult:
    finite = np.isfinite(signal) & membership
    coverage = float(finite.sum() / max(1, membership.sum()))
    count = finite.sum(axis=1)
    valid_day_mask = count >= 100
    values = np.where(finite, signal, 0.0)
    with np.errstate(all="ignore"):
        variance = np.square(values).sum(axis=1) / count - np.square(values.sum(axis=1) / count)
    variable = valid_day_mask & (variance > 1e-12)
    valid_days = int(valid_day_mask.sum())
    variable_rate = float(variable.sum() / max(1, valid_days))
    diagnostics = []
    for pool in pools:
        pearson_daily = daily_corr(signal, pool, membership)[valid_day_mask]
        finite_pearson = pearson_daily[np.isfinite(pearson_daily)]
        mean_abs = float(np.mean(np.abs(finite_pearson))) if len(finite_pearson) else 0.0
        common = membership & np.isfinite(signal) & np.isfinite(pool)
        pooled = finite_corr(signal[common], pool[common])
        pooled = float(pooled) if np.isfinite(pooled) else 0.0
        rank_values = []
        for day in np.flatnonzero(valid_day_mask):
            day_common = common[day]
            if day_common.sum() >= 3:
                rank_values.append(finite_corr(rankdata(signal[day, day_common]), rankdata(pool[day, day_common])))
        finite_rank = np.asarray(rank_values, dtype=float)
        finite_rank = finite_rank[np.isfinite(finite_rank)]
        mean_abs_rank = float(np.mean(np.abs(finite_rank))) if len(finite_rank) else 0.0
        diagnostics.append((mean_abs, pooled, mean_abs_rank, float(len(finite_pearson) / max(1, valid_days))))
    redundancy = [max(item[0], abs(item[1]), item[2]) for item in diagnostics]
    strongest = int(np.argmax(redundancy))
    mean_abs, pooled, mean_abs_rank, correlation_coverage = diagnostics[strongest]
    maximum = redundancy[strongest]
    reason = "ok"
    if coverage < 0.80:
        reason = "coverage_failure"
    elif valid_days < 252:
        reason = "insufficient_valid_days"
    elif variable_rate < 0.80:
        reason = "near_constant"
    elif maximum > 0.95:
        reason = "near_duplicate_signal"
    return ValidityResult(
        reason == "ok", reason, coverage, valid_days, variable_rate, maximum,
        mean_abs, pooled, mean_abs_rank, correlation_coverage,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/revision_v3_full_2000_cuda3.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads((args.run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    hashes = [str(item["expr_hash"]) for item in checkpoint["pool"]]
    pools = [np.load(args.run_dir / "cache/signals" / f"{key}.npy", mmap_mode="r") for key in hashes]
    panel = PanelStore(load_paths(args.config).processed_root).load_split("train")
    membership = panel.target(panel.common_mask)
    candidate = np.asarray(pools[0])
    _daily_rank_corr_exact(candidate[:2, :8], candidate[:2, :8], membership[:2, :8], np.ones(2, dtype=bool))
    started = time.perf_counter()
    legacy = legacy_validate(candidate, membership, pools)
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized = validate_signal(candidate, membership, pools)
    optimized_seconds = time.perf_counter() - started
    numeric_fields = (
        "coverage", "valid_days", "variable_day_rate", "max_pool_correlation",
        "mean_abs_daily_corr", "pooled_correlation", "mean_abs_daily_rank_corr",
        "correlation_coverage",
    )
    equivalent = legacy.valid == optimized.valid and legacy.reason == optimized.reason and all(
        np.isclose(float(getattr(legacy, field)), float(getattr(optimized, field)), atol=1e-12)
        for field in numeric_fields
    )
    print(json.dumps({
        "pool_size": len(pools),
        "panel_shape": list(candidate.shape),
        "legacy_seconds": legacy_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup": legacy_seconds / optimized_seconds,
        "numerically_equivalent": equivalent,
        "reason": optimized.reason,
    }, indent=2))


if __name__ == "__main__":
    main()
