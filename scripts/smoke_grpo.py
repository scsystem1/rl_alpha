from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlalpha.search.grpo.staged_controller import StagedGRPOSearcher
from rlalpha.search.models import CandidateOutcome, SearchContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/rlalpha_grpo_smoke"))
    args = parser.parse_args()
    config = {"run_dir": str(args.run_dir), "model": {"path": "/data/shared/huggingface/Qwen3.5-2B"}}
    searcher = StagedGRPOSearcher(0, config)
    context = SearchContext(0, (), (), 0.0, 0, 64)
    reports = []
    for update in range(args.updates):
        candidates = searcher.propose(context, 8)
        outcomes = []
        for index, candidate in enumerate(candidates):
            reward = (index - 3.5) / 3.5
            outcomes.append(CandidateOutcome(candidate.expr_hash, candidate.expression, candidate.node is not None, "synthetic", False, reward / 100, reward))
        searcher.observe(outcomes)
        reports.append({"update": update + 1, "valid": sum(item.node is not None for item in candidates), "pool_version": searcher.pool_version})
    print(json.dumps({"updates": searcher.updates, "group": searcher.rollout_group, "zero_group_variance": searcher.zero_group_variance, "reports": reports, "state": searcher.state_dict()}, indent=2))


if __name__ == "__main__":
    main()
