from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rlalpha.search.base_llm import BaseLLMSearcher
from rlalpha.search.models import SearchContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    searcher = BaseLLMSearcher(args.seed, {"model": {"path": None}, "rollout": {"max_model_len": 4096}})
    context = SearchContext(0, (), (), 0.0, 0, args.n)
    candidates = searcher.propose(context, args.n)
    valid = [item for item in candidates if item.node is not None]
    report = {"completions": len(candidates), "valid": len(valid), "parse_validity": len(valid) / max(1, len(candidates)), "unique": len({item.expr_hash for item in valid}), "unique_rate": len({item.expr_hash for item in valid}) / max(1, len(valid)), "mean_length": statistics.fmean(len(item.raw_text or "") for item in candidates), "expressions": [item.expression for item in valid], "raw_completions": [item.raw_text for item in candidates]}
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("completions", "valid", "parse_validity", "unique", "unique_rate", "mean_length")}, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
