from __future__ import annotations

import numpy as np
import yaml

from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolScore
from rlalpha.search.gp import GPSearcher
from rlalpha.search.coordinator import SearchCoordinator
from rlalpha.search.models import SearchContext
from rlalpha.search.random_search import RandomSearcher
from rlalpha.matrix.runner import _contains_cuda_oom, run_matrix


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


def test_gp_resume_and_pool_change_invalidate_fitness():
    gp = GPSearcher(5, population_size=16)
    proposed = gp.propose(_context(0), 8)
    state = gp.state_dict()
    resumed = GPSearcher(8, population_size=16)
    resumed.load_state_dict(state)
    assert [item.expression for item in resumed.propose(_context(0), 8)] == [item.expression for item in gp.propose(_context(0), 8)]
    resumed.propose(_context(1), 1)
    assert all(item.fitness == float("-inf") for item in resumed.population)


def test_random_and_gp_each_propose_32_typed_candidates():
    for searcher in (RandomSearcher(91), GPSearcher(91, population_size=32)):
        proposed = searcher.propose(_context(), 32)
        assert len(proposed) == 32
        assert all(item.node is not None and item.node.depth <= 6 and item.node.lookback <= 252 for item in proposed)


def test_pool_candidate_delta_matches_complete_recomputation():
    pool = PoolManager(_Objective(), capacity=2)
    from rlalpha.factors.records import PoolEntry

    pool.entries = [PoolEntry("a", "a", np.array([1.0])), PoolEntry("b", "b", np.array([2.0]))]
    candidate = PoolEntry("c", "c", np.array([5.0]))
    score = pool.score_candidates([candidate])[0]
    alternatives = [_Objective().score_pool([entry.signal for entry in entries]).objective for entries in ([candidate, pool.entries[1]], [pool.entries[0], candidate])]
    baseline = _Objective().score_pool([entry.signal for entry in pool.entries]).objective
    assert score.delta_objective == max(alternatives) - baseline


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


def test_matrix_oom_detection_is_specific_to_cuda_allocation_failures():
    assert _contains_cuda_oom("torch.OutOfMemoryError: CUDA out of memory")
    assert _contains_cuda_oom("CUBLAS_STATUS_ALLOC_FAILED")
    assert not _contains_cuda_oom("ValueError: malformed expression")


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
    first = run_matrix(config, "matrix_smoke", poll_seconds=0)
    assert first["cells"]["random/r0/seed_0"]["status"] == "complete"
    assert first["cells"]["gp/r0/seed_0"]["status"] == "failed"
    failed_methods.clear()
    second = run_matrix(config, "matrix_smoke", poll_seconds=0)
    assert all(state["status"] == "complete" for state in second["cells"].values())
    assert calls.count("random") == 1
    assert calls.count("gp") == 2
