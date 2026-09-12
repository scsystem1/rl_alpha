import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rlalpha.config import PathsConfig
from rlalpha.dsl.parser import parse_expression
from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry
from rlalpha.rewards.factory import objective_for
from rlalpha.search.models import Candidate
from rlalpha.search.random_search import RandomSearcher
from rlalpha.search.run import run_search
from rlalpha.utils.io import write_yaml


def _panel_config():
    dates = pd.bdate_range("2018-07-01", "2020-06-30")
    rng = np.random.default_rng(135)
    signals = {expression: rng.normal(size=(len(dates), 120)) for expression in ("$return", "$close", "$volume")}
    label = .4 * signals["$return"] + .25 * signals["$close"] + .1 * signals["$volume"] + rng.normal(size=(len(dates), 120))
    panel = SimpleNamespace(target_dates=dates, label=label, common_mask=np.ones_like(label, bool),
        exposures=np.stack([np.ones_like(label), rng.normal(size=label.shape)], axis=-1), target=lambda x: x,
        evaluate=lambda node: signals[node.canonical()])
    config = {"protocol": "recent_alpha_v1", "data": {"warmup_start": "2009-01-01", "horizon_trading_days": 20,
        "execution": "next_close", "fundamental_lag_months": 6, "fundamental_max_age_months": 18,
        "train": ["2018-07-01", "2020-06-30"],
        "validation": ["2020-07-01", "2020-12-31"], "test": ["2021-01-01", "2021-12-31"]},
        "experiment": {"proposal_group_size": 8, "pool_capacity": 2, "methods": ["random"], "rewards": ["r1_oof"], "seeds": [0]},
        "reward": {"name": "r1_oof", "neutralized": True, "min_pool_valid_days": 80, "time_folds": [
            {"fit": ["2018-07-01", "2018-12-31"], "score": ["2019-01-01", "2019-06-30"]},
            {"fit": ["2019-01-01", "2019-06-30"], "score": ["2019-07-01", "2019-12-31"]},
            {"fit": ["2019-07-01", "2019-12-31"], "score": ["2020-01-01", "2020-06-30"]}]}}
    return panel, config


@pytest.mark.parametrize("empty", [False, True])
def test_recent_search_never_loads_or_scores_calibration_and_selects_terminal(monkeypatch, tmp_path, empty):
    import rlalpha.search.run as run_module

    panel, config = _panel_config()
    paths = PathsConfig(code_root=Path(__file__).parents[2], runs_root=tmp_path / "runs", processed_root=tmp_path / "processed")
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, config)
    monkeypatch.setattr(run_module, "load_paths", lambda _: paths)
    monkeypatch.setattr(run_module, "git_info", lambda _: {})
    monkeypatch.setattr(run_module, "discover_data_files", lambda _: {})
    monkeypatch.setattr(run_module, "build_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_module, "_record_gpu_environment", lambda _: None)
    monkeypatch.setattr(run_module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""))
    loaded = []

    def load_interval(name, start, end):
        assert (name, start, end) == ("train", "2018-07-01", "2020-06-30")
        loaded.append(name)
        return panel

    monkeypatch.setattr(run_module, "PanelStore", lambda _: SimpleNamespace(load_interval=load_interval))
    monkeypatch.setattr(run_module, "_score_validation", lambda *a, **k: pytest.fail("calibration entered search feedback"))
    monkeypatch.setattr(run_module, "_select_snapshot", lambda *a: pytest.fail("historical snapshots were selected"))

    class DeterministicSearch(RandomSearcher):
        def propose(self, context, n):
            if empty:
                return [Candidate(None, "random", raw_text="invalid") for _ in range(n)]
            expression = ("$return", "$close", "$volume")[min(self.observed // n, 2)]
            return [Candidate(parse_expression(expression), "random") for _ in range(n)]

    monkeypatch.setattr(run_module, "searcher_for", lambda *a, **k: DeterministicSearch(0))
    if empty:
        with pytest.raises(RuntimeError, match="empty terminal pool"):
            run_search(config_path, "random", "r1_oof", 0, 3, "recent/test_2021")
        folder = paths.runs_root / "recent/test_2021/random/r1_oof/seed_0"
        assert json.loads((folder / "final_pool.json").read_text())["status"] == "empty_pool"
        assert json.loads((folder / "result.json").read_text())["status"] == "failed_empty_pool"
        assert json.loads((folder / "train_metrics.json").read_text())["raw_proposals"] == 24
        return
    result = run_search(config_path, "random", "r1_oof", 0, 3, "recent/test_2021")
    folder = Path(result["run_dir"])
    terminal = json.loads((folder / "final_pool.json").read_text())
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    assert terminal["selection_rule"] == "fixed_budget_terminal_pool"
    assert terminal["expressions"] == [entry["expression"] for entry in checkpoint["pool"]]
    assert terminal["validation"] == {}
    assert result["completed_steps"] == 3 and result["raw_proposals"] == 24
    assert loaded == ["train"]
    resumed = run_search(config_path, "random", "r1_oof", 0, 3, "recent/test_2021")
    assert resumed["raw_proposals"] == 24
    assert resumed["wall_seconds"] >= result["wall_seconds"]
    effective_before = (folder / "effective_config.yaml").read_text()
    # Even policy/date changes with an identical train panel are not resumable.
    config["data"]["validation"] = ["2020-08-01", "2020-12-31"]
    write_yaml(config_path, config)
    with pytest.raises(RuntimeError, match="identity changed"):
        run_search(config_path, "random", "r1_oof", 0, 3, "recent/test_2021")
    assert (folder / "effective_config.yaml").read_text() == effective_before


def test_grpo_recent_worker_receives_interval_and_matches_main_score(monkeypatch, tmp_path):
    from rlalpha.search.grpo import verl_reward_function as worker
    from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator
    from rlalpha.utils.io import write_json

    panel, config = _panel_config()
    objective = objective_for("r1_oof", panel, config["reward"])
    pool = PoolManager(objective, capacity=2)
    initial = parse_expression("$return")
    pool.entries = [PoolEntry(initial.canonical(), initial.expr_hash, panel.evaluate(initial))]
    pool.version = 1
    candidate = parse_expression("$close")
    expected = pool.score_candidates([PoolEntry(candidate.canonical(), candidate.expr_hash, panel.evaluate(candidate))])[0]
    coordinator = VerlGRPOStageCoordinator(pool, panel.evaluate, panel.common_mask, 8, tmp_path, config,
        "/qe", "/processed", "r1_oof", 0, train_start="2018-07-01", train_end="2020-06-30", max_training_steps=1)
    spec = coordinator._stage_spec(tmp_path / "archive.jsonl", 1)
    write_json(tmp_path / "spec.json", spec)
    for cache in ("_PANELS", "_SIGNALS", "_OBJECTIVES", "_POOLS"):
        monkeypatch.setattr(worker, cache, {})
    intervals = []

    def load_interval(*args):
        intervals.append(args)
        return panel

    monkeypatch.setattr(worker, "PanelStore", lambda _: SimpleNamespace(load_interval=load_interval))
    requests = [{"solution_str": "<expr>$close</expr>", "extra_info": {"stage": 0, "prompt_group": 0,
        "split": "train", "pool_version": 1, "expected_stage_samples": 1,
        "stage_spec_path": str(tmp_path / "spec.json"), "frozen_state_hash": spec["spec_hash"]}}]
    record = worker._score_batch_sync(requests)[0]
    assert intervals == [("train", "2018-07-01", "2020-06-30")]
    assert record["delta_add"] == pytest.approx(expected.delta_add)
    assert record["shaped_reward"] == pytest.approx(expected.shaped_reward)
    coordinator.save_checkpoint()
    coordinator.load_checkpoint()
    config["data"]["test"] = ["2022-01-01", "2022-12-31"]
    with pytest.raises(RuntimeError, match="window or calibration policy"):
        coordinator.load_checkpoint()


def test_real_alphagen_gp_recent_oof_when_checkout_is_available(tmp_path):
    from rlalpha.dsl.evaluator import evaluate
    from rlalpha.search.coordinator import SearchCoordinator
    from rlalpha.search.gp import GPSearcher

    root = Path(os.getenv("RLALPHA_ALPHAGEN_ROOT", str(Path(__file__).parents[3] / "alphagen")))
    if not (root / "gplearn/__init__.py").exists():
        pytest.skip(f"real AlphaGen engine unavailable: {root / 'gplearn'}")
    panel, config = _panel_config()
    returns = panel.evaluate(parse_expression("$return")) * .01
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    features = {"$return": returns, "$close": close, "$open": close * 1.001, "$high": close * 1.01,
        "$low": close * .99, "$volume": np.exp(panel.evaluate(parse_expression("$volume"))) * 1000}
    pool = PoolManager(objective_for("r1_oof", panel, config["reward"]), capacity=2)
    gp = GPSearcher(8, root, population_size=8)
    evaluator = lambda node: evaluate(node, features, eligibility_mask=panel.common_mask)
    coordinator = SearchCoordinator(gp, pool, evaluator, panel.common_mask, 16, tmp_path)
    outcomes = coordinator.run_group(8)
    assert len(outcomes) == 8 and gp.generation == 1
    assert {record.metadata["pre_group_pool_version"] for record in outcomes} == {0}
    assert any(record.market_evaluated for record in outcomes)
    restored_gp = GPSearcher(8, root, population_size=8)
    restored = SearchCoordinator(restored_gp, PoolManager(objective_for("r1_oof", panel, config["reward"]), capacity=2),
        evaluator, panel.common_mask, 16, tmp_path)
    restored.load_checkpoint()
    assert restored.group_index == 1
    assert [candidate.expression for candidate in gp.propose(coordinator.context(), 8)] == [
        candidate.expression for candidate in restored_gp.propose(restored.context(), 8)]
