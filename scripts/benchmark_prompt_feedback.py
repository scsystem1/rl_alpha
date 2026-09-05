"""Offline matched prompt ablation on predetermined TRAIN snapshots; no admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from rlalpha.config import load_yaml
from rlalpha.data.store import PanelStore
from rlalpha.dsl.parser import parse_expression
from rlalpha.dsl.validity import validate_signal
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.factory import objective_for
from rlalpha.search.base_llm import BaseLLMSearcher
from rlalpha.search.models import SearchContext, TrainPoolSummary
from rlalpha.search.prompts import build_messages, prompt_contract
from rlalpha.utils.hashing import file_fingerprint, stable_hash
from rlalpha.utils.io import write_json


def select_snapshots(rows, budgets):
    selected = [{"valid_unique_evaluations": 0, "pool_version": 0, "expressions": []}]
    for budget in budgets:
        if budget <= 0:
            continue
        row = next((r for r in rows if int(r["valid_unique_evaluations"]) >= budget), None)
        if row is None:
            raise ValueError(f"snapshot archive has not reached predeclared budget {budget}")
        selected.append({k: row[k] for k in ("valid_unique_evaluations", "pool_version", "expressions")})
    return selected


def score_completions(candidates, pool, panel, seen):
    records, entries, indices = [], [], []
    for candidate in candidates:
        row = {"expression": candidate.expression, "raw_text": candidate.raw_text,
               "valid": False, "reason": "parse_or_type_error"}
        if candidate.node is not None:
            if candidate.expr_hash in seen or candidate.expr_hash in pool.hashes:
                row["reason"] = "exact_duplicate"
            else:
                seen.add(candidate.expr_hash)
                try:
                    signal = panel.evaluate(candidate.node)
                except Exception as error:
                    row.update({"reason": "evaluation_error", "error": str(error)})
                    records.append(row)
                    continue
                validity = validate_signal(signal, panel.target(panel.common_mask), [e.signal for e in pool.entries])
                row.update({"reason": validity.reason, "validity": asdict(validity)})
                if validity.valid:
                    indices.append(len(records))
                    entries.append(PoolEntry(candidate.expression, candidate.expr_hash, signal))
        records.append(row)
    for index, score in zip(indices, pool.score_candidates(entries), strict=True):
        records[index].update({"valid": score.valid, "reason": score.reason,
            "increment": asdict(score.add_increment), "post_prune_delta": score.post_prune_delta})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--snapshots", type=Path, required=True, help="checkpoints/snapshots.jsonl, never validation-selected final_pool.json")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/qwen3_5_2b.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.groups < 1:
        parser.error("groups must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a fresh ablation result path")
    source = [json.loads(line) for line in args.snapshots.read_text().splitlines() if line.strip()]
    snapshots = select_snapshots(source, args.budgets)
    panel = PanelStore(args.processed_root).load_split("train")
    reward_config = load_yaml(Path(__file__).resolve().parents[1] / "configs/reward/r1_oof.yaml")["reward"]
    objective = objective_for("r1_oof", panel, reward_config)
    model = load_yaml(args.model_config)
    generator = BaseLLMSearcher(args.seed, {**model, "temperature": model["rollout"]["temperature"],
        "response_length": model["rollout"]["response_length"]})
    output = {"contract": prompt_contract(), "snapshot_source": file_fingerprint(args.snapshots),
        "model_config": model, "panel_fingerprint": panel.panel_fingerprint,
        "seed": args.seed, "groups": args.groups, "time_folds": objective.time_folds,
        "selection": "first train snapshot reaching each predeclared budget; no validation/test metrics", "results": []}
    for snapshot in snapshots:
        pool = PoolManager(objective, capacity=20)
        pool.entries = [PoolEntry(expr, parse_expression(expr).expr_hash, panel.evaluate(parse_expression(expr)))
                        for expr in snapshot["expressions"]]
        pool.version = int(snapshot["pool_version"])
        state = pool.prepared_state()
        context = SearchContext(pool.version, tuple(snapshot["expressions"]), tuple(state.score.weights),
            state.score.mean_ic, int(snapshot["valid_unique_evaluations"]), 2000,
            prompt_summary=TrainPoolSummary(**objective.prompt_summary(state)))
        rng_state = generator.rng.getstate()
        frozen = stable_hash({"version": pool.version, "formulas": context.pool_formulas})
        for variant in ("with_summary", "without_summary"):
            generator.rng.setstate(rng_state)
            prompt_context = context if variant == "with_summary" else replace(context, prompt_summary=None)
            seen, records = set(), []
            for group in range(args.groups):
                batch = score_completions(generator.propose(prompt_context, 8), pool, panel, seen)
                records.extend({"group": group, **record} for record in batch)
            valid = [r for r in records if r["valid"]]
            deltas = [r["increment"]["mean_delta"] for r in valid]
            correlations = [r["validity"]["max_pool_correlation"] for r in valid]
            output["results"].append({"snapshot": snapshot, "variant": variant,
                "prompt_hash": stable_hash(build_messages(prompt_context)),
                "raw_proposals": len(records), "valid_rate": len(valid)/len(records),
                "duplicate_rate": sum(r["reason"] == "exact_duplicate" for r in records)/len(records),
                "oof_delta_quantiles": np.quantile(deltas, [0, .25, .5, .75, 1]).tolist() if deltas else [],
                "mean_abs_pool_correlation": float(np.mean(correlations)) if correlations else None,
                "records": records})
            assert frozen == stable_hash({"version": pool.version, "formulas": tuple(e.expression for e in pool.entries)})
        write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "comparisons": len(snapshots)}))


if __name__ == "__main__":
    main()
