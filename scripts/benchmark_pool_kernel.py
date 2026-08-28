from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
import time

import numpy as np

from rlalpha.data.store import PanelStore
from rlalpha.dsl.parser import parse_expression
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.r0 import R0Objective


BASE_EXPRESSIONS = [
    *(f"CSRank(Mean({feature},{window}))" for feature in ("$open", "$high", "$low", "$close", "$volume", "$return") for window in (5, 10, 20)),
    "CSZScore($close)",
    "CSZScore($volume)",
]
CANDIDATE_EXPRESSIONS = [
    "CSRank(Std($return,20))",
    "CSRank(Delta($close,5))",
    "CSRank(TSRank($volume,20))",
    "CSZScore(Mean($return,40))",
    "CSRank(Corr($close,$volume,20))",
    "CSRank(Div($close,Mean($close,20)))",
    "CSRank(Sub($high,$low))",
    "CSRank(Mul($return,$volume))",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="/data/sunyuxiang/rl_alpha/processed")
    parser.add_argument("--start", default="2017-01-03")
    parser.add_argument("--end", default="2017-01-10")
    args = parser.parse_args()

    panel = PanelStore(args.processed_root).load_split("train", start=args.start, end=args.end)
    label = panel.target(panel.label)
    mask = panel.target(panel.common_mask) & np.isfinite(label)
    # Use real panel values while keeping the benchmark focused on the pool
    # kernel rather than DSL rolling-window evaluation.
    sources = [panel.target(panel.features[name]) for name in ("$open", "$high", "$low", "$close", "$volume", "$return")]
    generated = [
        np.asarray(sources[index % len(sources)], dtype=float)
        + (index + 1) * 1e-4 * np.asarray(sources[(index + 1) % len(sources)], dtype=float)
        for index in range(28)
    ]
    base = [PoolEntry(expression, parse_expression(expression).expr_hash, signal) for expression, signal in zip(BASE_EXPRESSIONS, generated[:20], strict=True)]
    candidates = [PoolEntry(expression, parse_expression(expression).expr_hash, signal) for expression, signal in zip(CANDIDATE_EXPRESSIONS, generated[20:], strict=True)]

    context = multiprocessing.get_context("fork")

    def measure(callback):
        parent, child = context.Pipe(duplex=False)

        def target():
            started = time.perf_counter()
            result = callback()
            child.send((time.perf_counter() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, result))
            child.close()

        process = context.Process(target=target)
        process.start()
        measured = parent.recv()
        process.join()
        if process.exitcode:
            raise RuntimeError(f"benchmark child failed with exit code {process.exitcode}")
        return measured

    def legacy_kernel():
        objective = R0Objective(label, mask)
        baseline = objective.score_pool([entry.signal for entry in base])
        calls = 1
        for candidate in candidates:
            for index in range(len(base)):
                entries = base[:index] + [candidate] + base[index + 1 :]
                objective.score_pool([entry.signal for entry in entries])
                calls += 1
        return {"calls": calls, "baseline": baseline.objective}

    def new_kernel():
        objective = R0Objective(label, mask)
        pool = PoolManager(objective, capacity=20, replacement_top_k=3, admission_recheck_top_k=3)
        pool.entries = base
        scores = pool.score_candidates(candidates)
        return {"prepare_calls": objective.prepare_calls, "formal_rechecks": sum(score.formally_rechecked for score in scores)}

    legacy_seconds, legacy_peak_kib, legacy = measure(legacy_kernel)
    new_seconds, new_peak_kib, new = measure(new_kernel)
    print(json.dumps({
        "panel_shape": list(label.shape),
        "legacy_full_score_calls": legacy["calls"],
        "new_natural_prepare_calls": new["prepare_calls"],
        "new_formal_rechecks": new["formal_rechecks"],
        "legacy_wall_seconds": legacy_seconds,
        "new_wall_seconds": new_seconds,
        "speedup": legacy_seconds / new_seconds,
        "legacy_peak_kib": legacy_peak_kib,
        "new_peak_kib": new_peak_kib,
        "baseline_objective": legacy["baseline"],
    }, indent=2))


if __name__ == "__main__":
    main()
