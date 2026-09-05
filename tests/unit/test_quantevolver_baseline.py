from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rlalpha.dsl.parser import parse_expression
from rlalpha.search.quantevolver import reward_function as reward
from rlalpha.search.quantevolver.prompts import SEED_LIBRARY, task_for_round
from rlalpha.utils.hashing import stable_hash
from rlalpha.utils.io import write_json


def test_public_seed_adaptations_obey_shared_dsl_limits() -> None:
    for seed in SEED_LIBRARY:
        node = parse_expression(seed["expression"])
        assert node.nodes <= 21
        assert node.depth <= 6
        assert node.lookback <= 252
    assert task_for_round(0, 0)["seed_expr"] == SEED_LIBRARY[0]["expression"]


def test_dico_reward_persists_history_and_penalizes_later_repeats(tmp_path, monkeypatch) -> None:
    class FakePanel:
        def __init__(self):
            self.target_dates = pd.date_range("2010-01-01", periods=260, freq="B")
            self.common_mask = np.ones((260, 110), dtype=bool)
            self.label = np.random.default_rng(7).normal(size=self.common_mask.shape)

        @staticmethod
        def target(values):
            return np.asarray(values)

        def evaluate(self, node):
            generator = np.random.default_rng(int(node.expr_hash[:16], 16))
            return 0.08 * self.label + generator.normal(size=self.label.shape)

    class FakeStore:
        def __init__(self, _root):
            pass

        @staticmethod
        def load_split(_name):
            return FakePanel()

    monkeypatch.setattr(reward, "PanelStore", FakeStore)
    reward._PANELS.clear()
    reward._SIGNALS.clear()
    reward._HISTORY.clear()
    spec = {
        "schema_version": 1,
        "reward_semantics": "quantevolver-dico-rankic-v1",
        "processed_root": "/fake",
        "history_path": str(tmp_path / "history.jsonl"),
        "signal_cache_root": str(tmp_path / "signals"),
        "rollout_n": 8,
        "invalid_penalty": -1.0,
        "exact_repeat_penalty": 0.10,
        "family_free_quota": 8,
        "family_low_quality_threshold": 0.08,
        "family_repeat_penalty": 0.02,
        "family_good_new_threshold": 0.10,
        "family_new_bonus": 0.02,
        "elite_top_k": 20,
        "behavior_good_score_threshold": 0.12,
        "behavior_corr_threshold": 0.85,
        "behavior_corr_penalty": 0.08,
        "behavior_low_corr_threshold": 0.50,
        "behavior_low_corr_bonus": 0.02,
        "mined_rank_ic_threshold": 0.01,
        "mined_coverage_threshold": 0.60,
    }
    spec["spec_hash"] = stable_hash(spec)
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, spec)
    texts = [
        "<expr>Div(Mean($return,20),Std($return,120))</expr>",
        "<expr>Mul(-1,CSRank(Mean($return,20)))</expr>",
    ] * 4

    def requests(round_index: int):
        return [
            {
                "solution_str": text,
                "extra_info": {
                    "spec_path": str(spec_path),
                    "round": round_index,
                    "task_id": "smoke",
                    "seed_id": "seed",
                    "seed_expr": "Mean($return,20)",
                    "family": "smoke",
                    "time_split": "train",
                    "start_date": "2010-01-01",
                    "end_date": "2012-12-31",
                },
            }
            for text in texts
        ]

    first = reward._score_batch_sync(requests(0))
    second = reward._score_batch_sync(requests(1))
    assert all(item["valid"] for item in first + second)
    assert all(item["exact_repeat_penalty"] == 0.0 for item in first)
    assert all(item["exact_repeat_penalty"] == 0.10 for item in second)
    assert sum(bool(item["mined"]) for item in second) == 0
    persisted = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert len(persisted) == 16

