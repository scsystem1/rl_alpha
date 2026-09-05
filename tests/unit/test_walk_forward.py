from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rlalpha.factors.pool import PoolManager
from rlalpha.factors.records import PoolEntry, PoolScore
from rlalpha.rewards.factory import objective_for
from rlalpha.rewards.r1 import R1Objective
from rlalpha.rewards.statistics import gap_aware_mean_se, newey_west_mean_se
from rlalpha.rewards.walk_forward import WalkForwardObjective


def sample(assets=30):
    rng = np.random.default_rng(415)
    dates = pd.bdate_range("2010-01-01", periods=420)
    signals = [rng.normal(size=(420, assets)) for _ in range(4)]
    risk = np.stack([np.ones((420, assets)), rng.normal(size=(420, assets))], axis=-1)
    label = .3 * signals[0] - .15 * signals[1] + rng.normal(size=(420, assets))
    folds = [{"fit": [str(dates[s].date()), str(dates[s+119].date())],
              "score": [str(dates[s+120].date()), str(dates[s+219].date())]} for s in (0, 100, 200)]
    return signals, {"label": label, "mask": np.ones_like(label, bool), "exposures": risk,
        "dates": dates, "time_folds": folds, "min_pool_valid_days": 40}


@pytest.mark.parametrize("critical", [0., 1.645])
def test_cached_fold_fits_match_independent_reference_and_full_train_weights(critical):
    signals, kwargs = sample()
    signals[1][::9, :8] = np.nan
    signals[2][::13] = 1.0
    objective = WalkForwardObjective(**kwargs, critical_value=critical)
    state = objective.prepare_pool(signals[:3])
    expected = np.full(len(kwargs["dates"]), np.nan)
    for k, (a, b) in enumerate(zip(objective.fit_rows, objective.score_rows)):
        fit = R1Objective(kwargs["label"][a], kwargs["mask"][a], kwargs["exposures"][a])
        fitted = fit.prepare_pool([s[a] for s in signals[:3]])
        scoring = R1Objective(kwargs["label"][b], kwargs["mask"][b], kwargs["exposures"][b])
        prepared = scoring.prepare_pool([s[b] for s in signals[:3]])
        expected[b] = scoring.score_prepared_with_weights(prepared, fitted.score.weights).daily_ic
        np.testing.assert_allclose(state.fold_fits[k].weights, fitted.score.weights, atol=1e-12)
    np.testing.assert_allclose(state.score.daily_ic, expected, equal_nan=True, atol=1e-12)
    full = R1Objective(kwargs["label"], kwargs["mask"], kwargs["exposures"]).prepare_pool(signals[:3])
    np.testing.assert_allclose(state.score.weights, full.score.weights, atol=1e-12)
    assert state.score.objective == pytest.approx(np.nanmean(expected))
    assert objective.snapshot_diagnostics(state)["weights_source"] == "full_train"


def test_label_exit_purge_and_current_score_labels_cannot_change_current_fit():
    signals, kwargs = sample()
    obj = WalkForwardObjective(**kwargs)
    assert np.flatnonzero(obj.fit_rows[0])[-1] == 98  # t+21 <= 119
    assert np.flatnonzero(obj.score_rows[0])[-1] == 198  # t+21 <= 219
    base = obj.prepare_pool(signals[:2])
    for k, score_rows in enumerate(obj.score_rows):
        label = kwargs["label"].copy()
        label[score_rows] *= -1
        changed = WalkForwardObjective(**{**kwargs, "label": label}).prepare_pool(signals[:2])
        np.testing.assert_allclose(changed.fold_fits[k].weights, base.fold_fits[k].weights)
        assert not np.allclose(np.asarray(changed.score.daily_ic)[score_rows], np.asarray(base.score.daily_ic)[score_rows])


@pytest.mark.parametrize("critical", [0., 1.645])
def test_add_subset_empty_and_parallel_scores_preserve_fixed_support(critical):
    signals, kwargs = sample()
    signals[3][::3, :12] = np.nan
    obj = WalkForwardObjective(**kwargs, critical_value=critical)
    base = obj.prepare_pool(signals[:2])
    additions = obj.prepare_add_many(base, signals[2:])
    for signal, added in zip(signals[2:], additions):
        ref = obj.prepare_pool(signals[:2] + [signal])
        np.testing.assert_allclose(added.score.daily_ic, ref.score.daily_ic, equal_nan=True)
        assert np.array_equal(base.common_mask, added.common_mask)
    subset = obj.prepare_subset(additions[1], [2, 0])
    ref = obj.prepare_pool([signals[3], signals[0]])
    np.testing.assert_allclose(subset.score.daily_ic, ref.score.daily_ic, equal_nan=True)
    empty = obj.prepare_subset(base, [])
    assert np.nanmax(np.abs(empty.score.daily_ic)) == 0
    assert obj.compare_scores(base.score, base.score).reward == 0
    batch = obj.score_subsets(additions[1], [[2, 0], []])
    np.testing.assert_allclose(batch[0].daily_ic, subset.score.daily_ic, equal_nan=True)
    np.testing.assert_allclose(batch[1].daily_ic, empty.score.daily_ic, equal_nan=True)

    pool = PoolManager(obj, capacity=2)
    pool.entries = [PoolEntry(f"base{i}", f"base{i}", s) for i, s in enumerate(signals[:2])]
    candidates = [PoolEntry(f"new{i}", f"new{i}", s) for i, s in enumerate(signals[2:])]
    serial, parallel = pool.score_candidates(candidates), pool.score_candidates(candidates, max_workers=2)
    for x, y in zip(serial, parallel):
        assert x.delta_add == pytest.approx(y.delta_add)
        assert x.post_prune_delta == pytest.approx(y.post_prune_delta)
        assert x.delta_add == pytest.approx(x.add_increment.mean_delta - x.add_increment.penalty)
        assert x.post_prune_delta == pytest.approx(x.post_prune_increment.reward)
        assert x.post_prune_delta == pytest.approx(obj.compare_scores(base.score, x.pool_score).reward)


def test_paired_lcb_does_not_subtract_pool_standard_errors():
    _, kwargs = sample()
    obj = WalkForwardObjective(**kwargs, critical_value=1.645)
    rng = np.random.default_rng(5)
    old = np.where(obj.scoring_rows, rng.normal(0, .15, 420), np.nan)
    delta = np.where(obj.scoring_rows, .001 + rng.normal(0, .04, 420), np.nan)
    new = old + delta
    score = lambda a: PoolScore(float(np.nanmean(a)), float(np.nanmean(a)), tuple(a), (), gap_aware_mean_se(a))
    result = obj.compare_scores(score(old), score(new))
    assert result.standard_error == pytest.approx(gap_aware_mean_se(delta))
    assert result.standard_error != pytest.approx(gap_aware_mean_se(new) - gap_aware_mean_se(old))
    shifted = obj.compare_scores(score(old), score(old + .02))
    assert shifted.reward == pytest.approx(.02)
    assert shifted.standard_error == pytest.approx(0., abs=1e-15)


def test_gap_aware_hac_matches_reference_without_joining_purged_tails():
    rng = np.random.default_rng(12)
    continuous = rng.normal(size=70)
    assert gap_aware_mean_se(continuous) == pytest.approx(newey_west_mean_se(continuous))
    values = continuous.copy()
    values[25:46] = np.nan
    finite = np.isfinite(values)
    centered = np.where(finite, values - np.nanmean(values), 0.)
    distance = np.abs(np.arange(70)[:, None] - np.arange(70)[None, :])
    kernel = np.maximum(1 - distance / 21, 0.)
    reference = np.sqrt(centered @ kernel @ centered) / finite.sum()
    assert gap_aware_mean_se(values) == pytest.approx(reference)
    assert gap_aware_mean_se(values) != pytest.approx(newey_west_mean_se(values))


def test_insufficient_or_overlapping_folds_fail_instead_of_falling_back():
    signals, kwargs = sample()
    with pytest.raises(ValueError, match="insufficient OOF"):
        WalkForwardObjective(**{**kwargs, "min_pool_valid_days": 252}).prepare_pool(signals[:1])
    with pytest.raises(ValueError, match="disjoint"):
        WalkForwardObjective(**{**kwargs, "time_folds": kwargs["time_folds"][:1] * 2})
    with pytest.raises(ValueError, match="cover"):
        WalkForwardObjective(**{**kwargs, "dates": kwargs["dates"] + pd.Timedelta(days=365)})


def test_factory_worker_and_validation_use_correct_estimators():
    from rlalpha.search.grpo.verl_reward_function import _objective
    from rlalpha.search.run import _score_validation
    signals, kwargs = sample()
    panel = SimpleNamespace(label=kwargs["label"], common_mask=kwargs["mask"], exposures=kwargs["exposures"],
        target=lambda a: a, target_dates=kwargs["dates"])
    config = {"time_folds": kwargs["time_folds"], "min_pool_valid_days": 40, "critical_value": 1.645}
    a, b = objective_for("r2_paired_oof", panel, config), _objective("r2_paired_oof", panel, config)
    sa, sb = a.prepare_pool(signals[:2]), b.prepare_pool(signals[:2])
    np.testing.assert_allclose(sa.score.daily_ic, sb.score.daily_ic, equal_nan=True)
    outer = objective_for("r2_paired_oof", panel, config, evaluation=True)
    assert type(outer) is R1Objective
    np.testing.assert_allclose(sa.score.weights, outer.prepare_pool(signals[:2]).score.weights)
    panel.evaluate = lambda node: signals[0]
    trained = a.prepare_pool(signals[:1])
    metrics = _score_validation(["$return"], panel, "r2_paired_oof", trained.score.weights, reward_config=config)
    assert metrics["objective"] == metrics["mean_ic"]
    np.testing.assert_allclose(metrics["weights"], trained.score.weights)
    assert metrics["ridge_weight_source"] == "train"


def test_regime_flip_is_not_rewarded_as_future_predictive_signal():
    signals, kwargs = sample()
    rng = np.random.default_rng(42)
    kwargs["time_folds"] = kwargs["time_folds"][:1]
    label = .7 * signals[0] + rng.normal(size=signals[0].shape)
    label[120:] = -.7 * signals[0][120:] + rng.normal(size=signals[0][120:].shape)
    obj = WalkForwardObjective(**{**kwargs, "label": label})
    empty, new = obj.prepare_pool([]), obj.prepare_pool(signals[:1])
    assert obj.compare_scores(empty.score, new.score).reward < -.3


@pytest.mark.parametrize("critical", [0., 1.645])
def test_full_pool_admission_uses_paired_replacement_increment(critical):
    signals, kwargs = sample()
    objective = WalkForwardObjective(**kwargs, critical_value=critical)
    pool = PoolManager(objective, capacity=1)
    pool.entries = [PoolEntry("noise", "noise", signals[2])]
    baseline = pool.score
    candidate = PoolEntry("predictor", "predictor", signals[0])
    scored = pool.score_candidates([candidate])
    assert scored[0].replaced_hash == "noise" and scored[0].formally_rechecked
    admission = pool.consider_group([candidate], precomputed=scored)
    assert admission.admitted and admission.replaced_hash == "noise"
    increment = objective.compare_scores(baseline, pool.score)
    assert scored[0].post_prune_delta == pytest.approx(increment.reward)
    assert scored[0].post_prune_increment.penalty == pytest.approx(increment.penalty)


def test_prompt_diagnostics_are_reward_independent_and_cached():
    from rlalpha.rewards.factory import prompt_objective_for
    from rlalpha.rewards.r0 import R0Objective
    from rlalpha.search.prompt_diagnostics import PoolPromptDiagnostics
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2010-01-01", "2018-12-31")
    label = rng.normal(size=(len(dates), 8))
    mask = np.ones_like(label, bool)
    risk = np.ones((*label.shape, 1))
    signal = rng.normal(size=label.shape)
    panel = SimpleNamespace(label=label, common_mask=mask, exposures=risk, target=lambda x: x, target_dates=dates)
    raw = PoolManager(R0Objective(label, mask))
    oof = PoolManager(prompt_objective_for(panel))
    raw.entries = [PoolEntry("$return", "factor", signal)]
    oof.entries = list(raw.entries)
    left, right = PoolPromptDiagnostics(panel, raw), PoolPromptDiagnostics(panel, oof)
    assert left() == right()
    calls = left.objective.prepare_calls
    first = left()
    assert left() is first and left.objective.prepare_calls == calls
    assert right.objective is oof.objective


def test_oof_checkpoint_reproduces_scores_and_rejects_changed_folds(tmp_path):
    from rlalpha.search.coordinator import SearchCoordinator
    from rlalpha.search.random_search import RandomSearcher
    from rlalpha.dsl.parser import parse_expression
    from rlalpha.utils.io import write_json
    import json

    signals, kwargs = sample()
    expression = "$return"
    entry = PoolEntry(expression, parse_expression(expression).expr_hash, signals[0])

    def coordinator(options):
        return SearchCoordinator(RandomSearcher(12), PoolManager(WalkForwardObjective(**options)),
            lambda node: signals[0], kwargs["mask"], 8, tmp_path)

    first = coordinator(kwargs)
    first.pool.entries = [entry]
    first.pool.version = 1
    first.save_checkpoint()
    restored = coordinator(kwargs)
    restored.load_checkpoint()
    np.testing.assert_allclose(first.pool.score.daily_ic, restored.pool.score.daily_ic, equal_nan=True)
    candidate = PoolEntry("new", "new", signals[1])
    assert first.pool.score_candidates([candidate])[0].delta_add == pytest.approx(restored.pool.score_candidates([candidate])[0].delta_add)
    changed = coordinator({**kwargs, "critical_value": 1.645})
    with pytest.raises(RuntimeError, match="contract changed"):
        changed.load_checkpoint()
    folds = [{"fit": list(f["fit"]), "score": list(f["score"])} for f in kwargs["time_folds"]]
    folds[0]["fit"][0] = str(kwargs["dates"][1].date())
    with pytest.raises(RuntimeError, match="contract changed"):
        coordinator({**kwargs, "time_folds": folds}).load_checkpoint()
    commit_path = tmp_path / "checkpoint_commit.json"
    commit = json.loads(commit_path.read_text())
    commit["schema_version"] = 7
    write_json(commit_path, commit)
    with pytest.raises(RuntimeError, match="incompatible"):
        coordinator(kwargs).load_checkpoint()


def test_grpo_archive_round_trip_preserves_paired_increment_and_weight_source(tmp_path):
    from rlalpha.dsl.parser import parse_expression
    from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator

    signals, kwargs = sample()
    obj = WalkForwardObjective(**kwargs, critical_value=1.645)
    pool = PoolManager(obj)
    node = parse_expression("$return")
    score = pool.score_candidates([PoolEntry(node.canonical(), node.expr_hash, signals[0])])[0]
    coordinator = VerlGRPOStageCoordinator(pool, lambda _: signals[0], kwargs["mask"], 8, tmp_path,
        {"reward": {"time_folds": kwargs["time_folds"]}}, "/qe", "/processed", "r2_paired_oof", 12)
    record = {**asdict(score), "expr_hash": node.expr_hash, "expression": node.canonical(),
        "raw_text": "<expr>$return</expr>", "market_evaluated": True, "reason_code": "ok",
        "prompt_group": 0, "rollout_index": 0}
    entries, scores = coordinator._consume_records([record])
    assert entries[0].expr_hash == node.expr_hash
    assert scores[0].add_increment == score.add_increment
    assert scores[0].post_prune_increment == score.post_prune_increment
    np.testing.assert_allclose(scores[0].pool_score.weights, score.pool_score.weights)
    pool.entries = entries
    coordinator.save_checkpoint()
    coordinator.load_checkpoint()
    spec = coordinator._stage_spec(tmp_path / "archive.jsonl", 8)
    assert spec["reward_config"]["time_folds"] == kwargs["time_folds"]
    assert len(spec["pool_weights"]) == len(entries)
    pool.objective.critical_value = .5
    with pytest.raises(RuntimeError, match="contract changed"):
        coordinator.load_checkpoint()


def test_both_new_reward_configs_and_experiment_matrix_validate():
    from pathlib import Path
    from rlalpha.config import ProjectConfig, load_yaml
    root = Path(__file__).parents[2]
    a = ProjectConfig.model_validate(load_yaml(root / "configs/reward/r1_oof.yaml"))
    b = ProjectConfig.model_validate(load_yaml(root / "configs/reward/r2_paired_oof.yaml"))
    assert a.reward.time_folds == b.reward.time_folds
    matrix = ProjectConfig.model_validate(load_yaml(root / "configs/experiment/rolling_oof.yaml"))
    assert matrix.experiment.seeds == [0, 1, 2]
    assert matrix.experiment.rewards == ["r1_oof", "r2_paired_oof", "r1"]


@pytest.mark.parametrize("reward", ["r1_oof", "r2_paired_oof"])
def test_full_worker_batch_matches_main_search_including_pruning_and_replay(reward, monkeypatch, tmp_path):
    from rlalpha.dsl.parser import parse_expression
    from rlalpha.search.grpo import verl_reward_function as worker
    from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator
    from rlalpha.utils.io import write_json
    from rlalpha.utils.hashing import stable_hash

    signals, kwargs = sample(120)
    formulas = ["$return", "$close", "$volume"]
    nodes = [parse_expression(expr) for expr in formulas]
    by_hash = {node.expr_hash: signal for node, signal in zip(nodes, signals)}
    panel = SimpleNamespace(label=kwargs["label"], common_mask=kwargs["mask"], exposures=kwargs["exposures"],
        target=lambda x: x, target_dates=kwargs["dates"], evaluate=lambda node: by_hash[node.expr_hash])
    config = {"time_folds": kwargs["time_folds"], "min_pool_valid_days": 40, "critical_value": 1.645}
    pool = PoolManager(objective_for(reward, panel, config), capacity=1)
    pool.entries = [PoolEntry(formulas[0], nodes[0].expr_hash, signals[0])]
    pool.version = 1
    candidates = [PoolEntry(node.canonical(), node.expr_hash, by_hash[node.expr_hash]) for node in nodes[1:]]
    expected = pool.score_candidates(candidates)
    coordinator = VerlGRPOStageCoordinator(pool, panel.evaluate, kwargs["mask"], 8, tmp_path,
        {"reward": config, "experiment": {"grpo_reward_candidate_workers": 2}}, "/qe", "/processed", reward, 12)
    spec = coordinator._stage_spec(tmp_path / "archive.jsonl", 3)
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, spec)
    for cache in ("_PANELS", "_SIGNALS", "_OBJECTIVES", "_POOLS"):
        monkeypatch.setattr(worker, cache, {})

    def load_split(name, **kwargs):
        assert name == "train"
        return panel

    monkeypatch.setattr(worker, "PanelStore", lambda _: SimpleNamespace(load_split=load_split))
    requests = [{"solution_str": f"<expr>{expr}</expr>", "extra_info": {
        "stage": 0, "prompt_group": 0, "split": "train", "pool_version": 1,
        "expected_stage_samples": 3, "stage_spec_path": str(spec_path),
        "frozen_state_hash": spec["spec_hash"]}} for expr in (formulas[1], formulas[2], formulas[1])]
    records = worker._score_batch_sync(requests)
    for record, reference in zip(records[:2], expected, strict=True):
        assert record["valid"]
        assert record["delta_add"] == pytest.approx(reference.delta_add)
        assert record["post_prune_delta"] == pytest.approx(reference.post_prune_delta)
        assert record["shaped_reward"] == pytest.approx(reference.shaped_reward)
        assert record["add_increment"]["standard_error"] == pytest.approx(reference.add_increment.standard_error)
        assert record["replaced_hash"] == reference.replaced_hash
    assert records[2]["reason_code"] == "intra_group_duplicate_reused"
    assert records[2]["add_increment"] == records[0]["add_increment"]
    replay = worker._score_batch_sync(requests)
    assert [r["delta_add"] for r in replay] == [r["delta_add"] for r in records]
    assert pool.version == 1 and len(pool.entries) == 1

    spec["reward_contract"] = "changed"
    spec["spec_hash"] = stable_hash({k: v for k, v in spec.items() if k not in {"spec_hash", "archive_path", "signal_cache_root"}})
    write_json(spec_path, spec)
    for request in requests:
        request["extra_info"]["frozen_state_hash"] = spec["spec_hash"]
    with pytest.raises(RuntimeError, match="contract changed"):
        worker._score_batch_sync(requests)


def test_prompt_ablation_selects_train_budgets_and_never_admits(tmp_path):
    from pathlib import Path
    import runpy
    from rlalpha.dsl.parser import parse_expression
    from rlalpha.search.models import Candidate

    benchmark = runpy.run_path(str(Path(__file__).parents[2] / "scripts/benchmark_prompt_feedback.py"))
    rows = [{"valid_unique_evaluations": 8, "pool_version": 1, "expressions": ["$return"], "validation": {"mean_ic": .5}},
            {"valid_unique_evaluations": 16, "pool_version": 2, "expressions": ["$close"], "validation": {"mean_ic": -.5}}]
    selected = benchmark["select_snapshots"](rows, [12])
    assert selected[0]["expressions"] == [] and selected[1]["expressions"] == ["$close"]
    assert "validation" not in selected[1]
    with pytest.raises(ValueError, match="predeclared budget"):
        benchmark["select_snapshots"](rows, [20])
    signals, kwargs = sample(120)
    pool = PoolManager(WalkForwardObjective(**kwargs))
    node = parse_expression("$return")
    candidates = [Candidate(node, "base_llm"), Candidate(node, "base_llm"), Candidate(None, "base_llm", raw_text="bad")]
    panel = SimpleNamespace(evaluate=lambda _: signals[0], common_mask=kwargs["mask"], target=lambda x: x)
    scored = benchmark["score_completions"](candidates, pool, panel, set())
    assert scored[0]["valid"] and scored[0]["increment"]["mean_delta"] > 0
    assert [r["reason"] for r in scored[1:]] == ["exact_duplicate", "parse_or_type_error"]
    assert pool.entries == [] and pool.version == 0
