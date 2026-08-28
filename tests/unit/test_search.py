from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolScore
from rlalpha.search.gp import GPSearcher
from rlalpha.search.coordinator import SearchCoordinator
from rlalpha.search.models import CandidateOutcome, SearchContext
from rlalpha.search.random_search import RandomSearcher
from rlalpha.search.run import _select_snapshot, _snapshot_record, _write_lineage
from rlalpha.matrix.runner import _contains_cuda_oom, _gpu_candidates, _gpu_for, _pid_alive, run_matrix
from rlalpha.config import load_yaml


ALPHAGEN_ROOT = Path(__file__).parents[3] / "alphagen"


class _Objective:
    def score_pool(self, signals):
        score = float(sum(np.asarray(signal).mean() for signal in signals))
        return PoolScore(score, score, (), tuple(1.0 for _ in signals))


def _context(version=0):
    return SearchContext(version, (), (), 0.0, 0, 100)


def test_random_resume_reproduces_proposal_sequence():
    first = RandomSearcher(17)
    first.propose(_context(), 11)
    state = first.state_dict()
    expected = [item.expression for item in first.propose(_context(), 16)]
    resumed = RandomSearcher(999)
    resumed.load_state_dict(state)
    assert [item.expression for item in resumed.propose(_context(), 16)] == expected


def _gp_outcomes(candidates, pool_version, scale=1e-3):
    return [
        CandidateOutcome(
            candidate.expr_hash,
            candidate.expression,
            True,
            "ok",
            True,
            (index + 1) * scale,
            metadata={"pre_group_pool_version": pool_version},
        )
        for index, candidate in enumerate(candidates)
    ]


def test_alphagen_gp_resume_reproduces_next_generation_and_pool_delta_fitness():
    gp = GPSearcher(5, ALPHAGEN_ROOT, population_size=8)
    first = gp.propose(_context(0), 8)
    gp.observe(_gp_outcomes(first, 0))
    state = gp.state_dict()
    assert state["engine"] == "alphagen_modified_gplearn"
    assert Path(state["engine_source"]).resolve() == (ALPHAGEN_ROOT / "gplearn/genetic.py").resolve()
    assert [item["fitness"] for item in state["population"]] == pytest.approx(
        [(index + 1) * 1e-3 for index in range(8)]
    )

    expected = gp.propose(_context(1), 8)
    resumed = GPSearcher(999, ALPHAGEN_ROOT, population_size=8)
    resumed.load_state_dict(state)
    actual = resumed.propose(_context(1), 8)
    assert [(item.expression, item.parents) for item in actual] == [
        (item.expression, item.parents) for item in expected
    ]
    assert all(item.parents for item in actual)


def test_alphagen_gp_rejects_mixed_frozen_pool_versions():
    gp = GPSearcher(7, ALPHAGEN_ROOT, population_size=8)
    candidates = gp.propose(_context(3), 8)
    outcomes = _gp_outcomes(candidates, 3)
    outcomes[-1].metadata["pre_group_pool_version"] = 4
    with pytest.raises(ValueError, match="mixed frozen pool versions"):
        gp.observe(outcomes)


def test_random_and_alphagen_gp_each_propose_eight_typed_candidates():
    for searcher in (RandomSearcher(91), GPSearcher(91, ALPHAGEN_ROOT, population_size=8)):
        proposed = searcher.propose(_context(), 8)
        assert len(proposed) == 8
        assert all(item.node is not None and item.node.depth <= 6 and item.node.lookback <= 252 for item in proposed)
        if isinstance(searcher, GPSearcher):
            assert len({item.expr_hash for item in proposed}) == 8


def test_alphagen_gp_generation_uses_one_frozen_pool_and_one_admission(tmp_path):
    searcher = GPSearcher(13, ALPHAGEN_ROOT, population_size=8)
    pool = PoolManager(_Objective(), capacity=3, min_delta=-1.0)
    membership = np.ones((300, 120), dtype=bool)

    def evaluator(node):
        rng = np.random.default_rng(int(node.expr_hash[:16], 16))
        return rng.normal(size=membership.shape) + int(node.expr_hash[:4], 16) / 65536

    coordinator = SearchCoordinator(searcher, pool, evaluator, membership, 8, tmp_path)
    outcomes = coordinator.run_group(8)
    assert searcher.generation == 1
    assert searcher.fitness_pool_version == 0
    assert pool.version == 1
    assert len(pool.entries) == 1
    assert {item.metadata["pre_group_pool_version"] for item in outcomes} == {0}
    expected = [
        item.delta_objective if item.valid and item.market_evaluated and np.isfinite(item.delta_objective) else -np.inf
        for item in outcomes
    ]
    np.testing.assert_allclose([program.fitness_ for program in searcher.population], expected)


def test_pool_candidate_reward_is_add_only_and_pruning_is_separate():
    pool = PoolManager(_Objective(), capacity=2)
    from rlalpha.factors.records import PoolEntry

    pool.entries = [PoolEntry("a", "a", np.array([1.0])), PoolEntry("b", "b", np.array([2.0]))]
    candidate = PoolEntry("c", "c", np.array([5.0]))
    score = pool.score_candidates([candidate])[0]
    alternatives = [_Objective().score_pool([entry.signal for entry in entries]).objective for entries in ([candidate, pool.entries[1]], [pool.entries[0], candidate])]
    baseline = _Objective().score_pool([entry.signal for entry in pool.entries]).objective
    assert score.delta_add == 5.0
    assert score.delta_objective == score.delta_add
    assert score.post_prune_delta == max(alternatives) - baseline


def test_snapshot_selection_rejects_high_objective_with_low_support():
    sparse = {
        "pool_version": 2,
        "expressions": ["sparse"],
        "train": {"support": {"valid": False}},
        "validation": {"objective": 0.50, "support": {"valid": False}},
    }
    broad = {
        "pool_version": 1,
        "expressions": ["broad"],
        "train": {"support": {"valid": True}},
        "validation": {"objective": 0.03, "support": {"valid": True}},
    }
    assert _select_snapshot([sparse, broad]) is broad
    with pytest.raises(RuntimeError, match="support requirements"):
        _select_snapshot([sparse])


def test_staged_searcher_updates_pool_only_at_stage_boundary(tmp_path):
    class Staged(RandomSearcher):
        admission_group_interval = 2

    # Admission applies the same minimum panel dimensions as production.
    base = np.arange(300 * 120, dtype=float).reshape(300, 120)
    evaluator = lambda node: base + int(node.expr_hash[:4], 16) / 65536
    coordinator = SearchCoordinator(Staged(5), PoolManager(_Objective()), evaluator, np.ones_like(base, dtype=bool), 16, tmp_path)
    coordinator.run_group(8)
    assert coordinator.pool.version == 0
    assert list((tmp_path / "cache/signals").glob("*.npy"))
    coordinator.run_group(8)
    assert coordinator.pool.version == 1


def test_checkpoint_hash_mismatch_refuses_resume(tmp_path):
    base = np.arange(300 * 120, dtype=float).reshape(300, 120)
    coordinator = SearchCoordinator(RandomSearcher(5), PoolManager(_Objective()), lambda node: base, np.ones_like(base, dtype=bool), 8, tmp_path)
    coordinator.run_group(8)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(checkpoint.read_text(encoding="utf-8") + " ", encoding="utf-8")
    resumed = SearchCoordinator(RandomSearcher(5), PoolManager(_Objective()), lambda node: base, np.ones_like(base, dtype=bool), 8, tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resumed.load_checkpoint()


def test_lineage_artifacts_preserve_proposal_admission_and_final_factor_ids(tmp_path):
    base = np.arange(300 * 120, dtype=float).reshape(300, 120)
    searcher = RandomSearcher(7)
    pool = PoolManager(_Objective(), capacity=2, min_delta=-1.0)
    coordinator = SearchCoordinator(searcher, pool, lambda node: base + int(node.expr_hash[:4], 16) / 65536, np.ones_like(base, dtype=bool), 8, tmp_path)
    coordinator.run_group(8)
    snapshot = _snapshot_record(pool, {"objective": 1.0, "mean_ic": 1.0, "weights": [1.0]}, 8, searcher)
    final = _write_lineage(tmp_path, coordinator, [snapshot], snapshot, "random", "r0", 7, "lineage_test")
    assert final["factors"]
    assert all(item["lineage_status"] == "verified" for item in final["factors"])
    proposals = __import__("pandas").read_parquet(tmp_path / "lineage/proposals.parquet")
    admissions = __import__("pandas").read_parquet(tmp_path / "lineage/admission_events.parquet")
    final_rows = __import__("pandas").read_parquet(tmp_path / "lineage/final_pool_lineage.parquet")
    assert set(final_rows["factor_id"]).issubset(set(proposals["factor_id"]))
    assert set(final_rows["admission_event_id"]).issubset(set(admissions["admission_event_id"]))


def test_matrix_oom_detection_is_specific_to_cuda_allocation_failures():
    assert _contains_cuda_oom("torch.OutOfMemoryError: CUDA out of memory")
    assert _contains_cuda_oom("CUBLAS_STATUS_ALLOC_FAILED")
    assert not _contains_cuda_oom("ValueError: malformed expression")


def test_matrix_gpu_mapping_can_pin_all_llm_cells_to_one_physical_device():
    experiment = {"gpu_devices": {"base_llm": [3], "grpo_llm": [3]}}
    assert _gpu_for("base_llm", "r0", 0, experiment) == 3
    assert _gpu_for("grpo_llm", "r0", 0, experiment) == 3
    assert _gpu_for("grpo_llm", "r2_lcb", 7, experiment) == 3
    assert _gpu_for("random", "r0", 0, experiment) is None


def test_matrix_gpu_candidates_rotate_and_retain_fallbacks():
    experiment = {"gpu_devices": {"base_llm": [1, 2, 4], "grpo_llm": [3]}}
    assert _gpu_candidates("base_llm", "r0", 0, experiment) == [1, 2, 4]
    assert _gpu_candidates("base_llm", "r1", 0, experiment) == [2, 4, 1]
    assert _gpu_candidates("base_llm", "r2_lcb", 0, experiment) == [4, 1, 2]
    assert _gpu_candidates("grpo_llm", "r1", 0, experiment) == [3]


def test_matrix_pid_liveness_uses_the_operating_system(monkeypatch):
    calls = []
    monkeypatch.setattr("rlalpha.matrix.runner.os.kill", lambda pid, signal: calls.append((pid, signal)))
    assert _pid_alive(1234)
    assert calls == [(1234, 0)]
    assert not _pid_alive(None)


def test_unknown_config_fields_fail_instead_of_being_ignored(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("search:\n  method: random\n  silently_ignored: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_yaml(config)


def test_formal_gp_config_requires_eight_candidates_and_complete_probabilities(tmp_path):
    wrong_population = tmp_path / "wrong_population.yaml"
    wrong_population.write_text("search:\n  method: gp\n  population_size: 16\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="population_size=8"):
        load_yaml(wrong_population)

    partial_probabilities = tmp_path / "partial_probabilities.yaml"
    partial_probabilities.write_text("search:\n  method: gp\n  population_size: 8\n  p_crossover: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="specified together"):
        load_yaml(partial_probabilities)


def test_all_repository_yaml_configs_are_typed():
    root = __import__("pathlib").Path(__file__).parents[2]
    for path in sorted((root / "configs").glob("**/*.yaml")):
        load_yaml(path)


def test_matrix_failure_isolated_and_resume_only_restarts_failed_cell(tmp_path, monkeypatch):
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "paths": {"code_root": str(tmp_path), "raw_data_root": str(tmp_path), "processed_root": str(tmp_path), "cache_root": str(tmp_path), "runs_root": str(tmp_path / "runs"), "model_search_root": str(tmp_path), "alphagen_root": str(tmp_path), "quantevolver_root": str(tmp_path)},
        "experiment": {"cells": [["random", "r0"], ["gp", "r0"]], "seeds": [0], "valid_unique_budget": 8, "max_cpu_jobs": 2},
    }), encoding="utf-8")
    failed_methods = {"gp"}
    calls = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            method = command[command.index("--method") + 1]
            calls.append(method)
            self.returncode = 1 if method in failed_methods else 0
            self.pid = 999999

        def poll(self):
            return self.returncode

    monkeypatch.setattr("rlalpha.matrix.runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("rlalpha.matrix.runner._gpu_free_mib", lambda: {})
    monkeypatch.setattr("rlalpha.matrix.runner._cell_acceptance", lambda directory, budget: (True, None))
    first = run_matrix(config, "matrix_smoke", poll_seconds=0)
    assert first["cells"]["random/r0/seed_0"]["status"] == "complete"
    assert first["cells"]["gp/r0/seed_0"]["status"] == "failed"
    failed_methods.clear()
    second = run_matrix(config, "matrix_smoke", poll_seconds=0)
    assert all(state["status"] == "complete" for state in second["cells"].values())
    assert calls.count("random") == 1
    assert calls.count("gp") == 2
    root = tmp_path / "runs/matrix_smoke"
    assert (root / "progress.json").exists()
    assert (root / "experiment.log").exists()
    assert not list(root.rglob("*.lock"))


def test_matrix_method_filter_never_launches_other_methods(tmp_path, monkeypatch):
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "paths": {"code_root": str(tmp_path), "raw_data_root": str(tmp_path), "processed_root": str(tmp_path), "cache_root": str(tmp_path), "runs_root": str(tmp_path / "runs"), "model_search_root": str(tmp_path), "alphagen_root": str(tmp_path), "quantevolver_root": str(tmp_path)},
        "experiment": {"cells": [["random", "r0"], ["gp", "r0"]], "seeds": [0], "valid_unique_budget": 8, "max_cpu_jobs": 2},
    }), encoding="utf-8")
    calls = []

    class FakeProcess:
        pid = 999999
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command[command.index("--method") + 1])

        def poll(self):
            return 0

    monkeypatch.setattr("rlalpha.matrix.runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("rlalpha.matrix.runner._gpu_free_mib", lambda: {})
    monkeypatch.setattr("rlalpha.matrix.runner._cell_acceptance", lambda directory, budget: (True, None))
    result = run_matrix(config, "method_only", poll_seconds=0, methods=["random"])
    assert calls == ["random"]
    assert set(result["cells"]) == {"random/r0/seed_0"}


def test_matrix_reward_filter_never_launches_other_rewards(tmp_path, monkeypatch):
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "paths": {"code_root": str(tmp_path), "raw_data_root": str(tmp_path), "processed_root": str(tmp_path), "cache_root": str(tmp_path), "runs_root": str(tmp_path / "runs"), "model_search_root": str(tmp_path), "alphagen_root": str(tmp_path), "quantevolver_root": str(tmp_path)},
        "experiment": {"cells": [["random", "r1"], ["random", "r0"]], "seeds": [0], "valid_unique_budget": 8, "max_cpu_jobs": 2},
    }), encoding="utf-8")
    calls = []

    class FakeProcess:
        pid = 999999
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command[command.index("--reward") + 1])

        def poll(self):
            return 0

    monkeypatch.setattr("rlalpha.matrix.runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("rlalpha.matrix.runner._gpu_free_mib", lambda: {})
    monkeypatch.setattr("rlalpha.matrix.runner._cell_acceptance", lambda directory, budget: (True, None))
    result = run_matrix(config, "reward_only", poll_seconds=0, rewards=["r1"])
    assert calls == ["r1"]
    assert set(result["cells"]) == {"random/r1/seed_0"}
