from pathlib import Path
import json

import pytest

from rlalpha.config import load_yaml, merge_reward_config
from rlalpha.rolling import expected_cells, prepare_rolling_windows, window_config
from rlalpha.utils.io import write_json, write_yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("RLALPHA_CODE_ROOT", str(ROOT))
    monkeypatch.setenv("RLALPHA_RUNS_ROOT", str(tmp_path / "runs"))
    return ROOT / "configs/experiment/recent_alpha_rolling.yaml"


def test_five_calendars_and_rolling_oof(config):
    raw = load_yaml(config)
    assert len(expected_cells(raw)) * len(raw["rolling"]["test_years"]) == 60
    assert raw["experiment"]["search_steps"] == 100
    assert raw["experiment"]["rewards"] == ["r1_oof"]
    for year in range(2021, 2026):
        child = window_config(raw, year, ROOT)
        assert child["data"]["train"] == [f"{year-3}-07-01", f"{year-1}-06-30"]
        assert child["data"]["validation"] == [f"{year-1}-07-01", f"{year-1}-12-31"]
        assert child["data"]["test"] == [f"{year}-01-01", f"{year}-12-31"]
        folds = child["reward"]["time_folds"]
        assert len(folds) == 3
        assert folds[0]["fit"] == [f"{year-3}-07-01", f"{year-3}-12-31"]
        assert folds[-1]["score"] == [f"{year-1}-01-01", f"{year-1}-06-30"]
        assert all(left["score"] == right["fit"] for left, right in zip(folds, folds[1:]))
        r1 = merge_reward_config(child, ROOT, "r1_oof")
        r2 = merge_reward_config(child, ROOT, "r2_paired_oof")
        assert r2.pop("critical_value") == 0.5
        r1.pop("name")
        r2.pop("name")
        assert r1 == r2
        assert r1["min_pool_valid_days"] == 80
        assert r1["ridge"] == 0.01
    assert "time_folds" not in raw.get("reward", {})


def test_two_calendar_year_train_fit_scheme(config):
    raw = load_yaml(config)
    raw["rolling"]["window_scheme"] = "two_calendar_years_then_test_year"
    raw["evaluation"]["ridge_fit_period"] = "train"
    raw["experiment"]["search_steps"] = 200
    raw["experiment"]["cells"] = [
        ["random", "r1_oof"],
        ["gp", "r1_oof"],
        ["base_llm", "r1_oof"],
        ["grpo_llm", "r2_paired_oof"],
    ]
    raw["experiment"].pop("methods")
    raw["experiment"].pop("rewards")
    child = window_config(raw, 2021, ROOT)
    assert child["data"]["train"] == ["2019-01-01", "2020-12-31"]
    assert child["data"]["validation"] == child["data"]["train"]
    assert child["data"]["test"] == ["2021-01-01", "2021-12-31"]
    assert child["evaluation"]["ridge_fit_period"] == "train"
    assert child["experiment"]["search_steps"] == 200
    assert child["reward"]["time_folds"] == [
        {"fit": ["2019-01-01", "2019-06-30"], "score": ["2019-07-01", "2019-12-31"]},
        {"fit": ["2019-07-01", "2019-12-31"], "score": ["2020-01-01", "2020-06-30"]},
        {"fit": ["2020-01-01", "2020-06-30"], "score": ["2020-07-01", "2020-12-31"]},
    ]
    assert ("grpo_llm", "r2_paired_oof", 0) in expected_cells(raw)
    assert merge_reward_config(child, ROOT, "r2_paired_oof")["critical_value"] == 0.5


def test_freeze_resume_and_config_change_rejected(config, tmp_path):
    windows = prepare_rolling_windows(config, "demo")
    assert [year for year, _, _ in windows] == list(range(2021, 2026))
    assert [name for _, _, name in windows] == [f"demo/test_{year}" for year in range(2021, 2026)]
    times = [path.stat().st_mtime_ns for _, path, _ in windows]
    assert prepare_rolling_windows(config, "demo") == windows
    assert times == [path.stat().st_mtime_ns for _, path, _ in windows]
    changed = load_yaml(config)
    changed["experiment"]["search_steps"] = 101
    changed_path = tmp_path / "changed.yaml"
    write_yaml(changed_path, changed)
    with pytest.raises(RuntimeError, match="configuration changed"):
        prepare_rolling_windows(changed_path, "demo")
    # Even editing only one frozen child is rejected before any work launches.
    child = load_yaml(windows[0][1])
    child["data"]["test"] = ["2022-01-01", "2022-12-31"]
    write_yaml(windows[0][1], child)
    with pytest.raises(RuntimeError, match="window configuration changed"):
        prepare_rolling_windows(config, "demo")


def test_matrix_entry_reuses_existing_scheduler_per_year(config, monkeypatch):
    from rlalpha.matrix import runner

    calls = []
    def execute(path, experiment_id, resume, poll_seconds, methods, rewards):
        child = load_yaml(path)
        assert "rolling" not in child
        calls.append((experiment_id, child["data"]["test"], methods, rewards))
        return {"cells": {}}
    monkeypatch.setattr(runner, "_run_matrix_unlocked", execute)
    result = runner.run_matrix(config, "demo", methods=["random"], rewards=["r1_oof"])
    assert len(calls) == 5
    assert calls[0] == ("demo/test_2021", ["2021-01-01", "2021-12-31"], ["random"], ["r1_oof"])
    assert list(result["windows"]) == [str(y) for y in range(2021, 2026)]


def test_evaluate_entry_marks_missing_and_failed_cells(config, monkeypatch, tmp_path):
    from rlalpha.evaluation import finalize

    real_entry = finalize.finalize_experiment
    def child_entry(experiment_id, path, methods=None):
        assert not load_yaml(path).get("rolling")
        if experiment_id.endswith("2023"):
            raise RuntimeError("empty pool")
        return {}
    monkeypatch.setattr(finalize, "finalize_experiment", child_entry)
    result = real_entry("demo", config)
    assert not result["complete"]
    assert len(result["missing_cells"]) == 60
    assert result["failures"] == {"2023": "empty pool"}
    assert json.loads((tmp_path / "runs/demo/rolling_evaluation.json").read_text())["complete"] is False


def test_evaluate_complete_requires_all_cell_markers(config, monkeypatch, tmp_path):
    from rlalpha.evaluation import finalize
    from rlalpha.rolling import evaluate_rolling

    def child_entry(experiment_id, path, methods=None):
        cells = expected_cells(load_yaml(path))
        for method, reward, seed in cells:
            write_json(tmp_path / "runs" / experiment_id / method / reward / f"seed_{seed}/test/finalization.json", {"status": "complete"})
            write_json(tmp_path / "runs" / experiment_id / method / reward / f"seed_{seed}/test/metrics.json", {})
        return {f"{method}/{reward}/{seed}": {"status": "complete"} for method, reward, seed in cells}
    monkeypatch.setattr(finalize, "finalize_experiment", child_entry)
    assert evaluate_rolling("demo", config)["complete"]
    assert not evaluate_rolling("demo", config, methods=["random"])["complete"]


def test_disabled_expensive_matrix_does_not_freeze_failed_configuration(config, tmp_path):
    from rlalpha.matrix.runner import run_matrix

    with pytest.raises(RuntimeError, match="enable before freezing"):
        run_matrix(config, "disabled")
    assert not (tmp_path / "runs/disabled/rolling_manifest.json").exists()
    assert not (tmp_path / "runs/disabled/window_configs").exists()


def test_recent_protocol_accepts_native_baselines_and_rejects_wrong_pairings(config, tmp_path):
    from rlalpha.search.run import run_search

    raw = load_yaml(config)
    raw["experiment"]["methods"] = ["quantevolver"]
    with pytest.raises(ValueError, match="requires quantevolver reward"):
        window_config(raw, 2021, ROOT)
    raw["experiment"]["methods"] = ["random"]
    raw["experiment"]["rewards"] = ["r2_lcb"]
    with pytest.raises(ValueError, match="not r2_lcb"):
        window_config(raw, 2021, ROOT)
    child = window_config(load_yaml(config), 2021, ROOT)
    path = tmp_path / "child.yaml"
    write_yaml(path, child)
    with pytest.raises(ValueError, match="does not support cell"):
        run_search(path, "quantevolver", "r1_oof", 0, 100, "wrong_method")

    for filename, expected_cell in (
        ("recent_alpha_quantevolver_rolling.yaml", ("quantevolver", "qe_native", 0)),
        ("recent_alpha_alphasage_rolling.yaml", ("alphasage", "r0", 0)),
    ):
        baseline = load_yaml(ROOT / "configs/experiment" / filename)
        assert baseline["experiment"]["search_steps"] == 100
        assert baseline["experiment"]["proposal_group_size"] == 8
        assert len(expected_cells(baseline)) == 3
        assert expected_cells(baseline)[0] == expected_cell
        resolved = window_config(baseline, 2025, ROOT)
        assert resolved["protocol"] == "recent_alpha_v1"
        assert resolved["data"]["train"] == ["2022-07-01", "2024-06-30"]


@pytest.mark.parametrize("field,value", [("sleeves", 3), ("holding_days", 10), ("one_way_cost_bps", [0, 5])])
def test_invalid_portfolio_contract_rejected_before_search(config, field, value):
    raw = load_yaml(config)
    raw["evaluation"][field] = value
    with pytest.raises(ValueError, match="recent_alpha_v1"):
        window_config(raw, 2021, ROOT)


@pytest.mark.parametrize("seeds", [[], [0, 0, 1]])
def test_empty_or_duplicate_seed_cells_rejected(config, seeds):
    raw = load_yaml(config)
    raw["experiment"]["seeds"] = seeds
    with pytest.raises(ValueError, match="unique"):
        window_config(raw, 2021, ROOT)
