from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rlalpha.evaluation.portfolio import return_metrics
from rlalpha.config import load_yaml, resolve_data_evaluation
from rlalpha.reporting.build import build_report
from rlalpha.reporting.rolling import _load_cells, _portfolio_summary, _statistics
from rlalpha.utils.hashing import stable_hash


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def rolling_report_data(tmp_path, monkeypatch):
    code_root = Path(__file__).parents[2]
    runs = tmp_path / "runs"
    root = runs / "recent"
    methods = ["random", "gp", "base_llm", "grpo_llm"]
    years = list(range(2021, 2026))
    raw = {"paths": {"code_root": str(code_root), "runs_root": str(runs)},
           "experiment": {"methods": methods, "rewards": ["r1_oof"], "seeds": [0, 1, 2], "search_steps": 100},
           "rolling": {"test_years": years}, "evaluation": {"bootstrap_samples": 24}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from rlalpha.rolling import prepare_rolling_windows
    children = {year: load_yaml(path) for year, path, _ in prepare_rolling_windows(config, "recent")}
    # Actual matrix/panel identities require external data; all report-level
    # completeness gates still run against the synthetic artifacts below.
    monkeypatch.setattr("rlalpha.evaluation.finalize._assert_experiment_frozen", lambda *args: [])
    monkeypatch.setattr("rlalpha.evaluation.finalize._panel_fingerprints", lambda *args: [{"path": "/data/panel", "sha256": "common-panel"}])
    from rlalpha.evaluation.finalize import cell_input_hash, cell_finalization_scope_hash
    for year in years:
        _json(root / f"test_{year}/test_finalization.json", {"status": "complete"})
        dates = pd.bdate_range(f"{year}-01-01", periods=48)
        for m, method in enumerate(methods):
            for seed in range(3):
                cell = root / f"test_{year}" / method / "r1_oof" / f"seed_{seed}"
                _json(cell / "progress.json", {"status": "complete", "search_steps": 100})
                _json(cell / "train_metrics.json", {"completed_steps": 100, "raw_proposals": 800, "valid_unique_evaluations": 600,
                                                    "tokens": 40, "gpu_seconds": 5, "wall_seconds": 10})
                _json(cell / "final_pool.json", {"train": {"objective": .02}, "expressions": ["$return"]})
                data, evaluation = resolve_data_evaluation(children[year], code_root)
                digest = cell_input_hash(cell, tmp_path, evaluation, data, children[year]["protocol"],
                                         cell_finalization_scope_hash(f"{method}/r1_oof/seed_{seed}"))
                metrics = {"input_hash": digest, "pool_size": 1, "test_ic_valid_days": 27, "test_trade_valid_days": 48}
                _json(cell / "test/finalization.json", {"status": "complete", "input_hash": digest, "metrics_hash": stable_hash(metrics)})
                _json(cell / "test/metrics.json", metrics)
                (cell / "manifest.yaml").write_text(yaml.safe_dump({"panel_artifacts": [{"path": "/data/build_manifest.yaml", "sha256": "common-panel"}], "evaluator_version": "test-v1"}))
                signal = .005 + .003 * m + .001 * seed + .012 * np.sin(np.arange(48) / 3)
                signal[-21:] = np.nan
                pd.DataFrame({"date": dates, "raw_ic": signal * 2, "raw_rank_ic": signal * 1.8, "rnic": signal, "rank_rnic": signal * .9}).to_parquet(cell / "test/rnic_daily.parquet", index=False)
                returns = .0001 * (m + 1) + .01 * np.sin(np.arange(48) + seed * 2)
                turnover = np.full(48, .1)
                turnover[-1] += 1.0
                pd.DataFrame({"date": dates, "gross_return": returns, "turnover": turnover,
                              "net_return_0bps": returns, "net_return_10bps": returns - turnover * .001,
                              "liquidation_turnover": [0.] * 47 + [1.], "missing_held_returns": 0,
                              "missing_held_return_weight": 0., "infeasible": False}).to_parquet(cell / "test/dollar_neutral_daily.parquet", index=False)
                pd.DataFrame({"factor_id": ["return-factor"], "expression": ["$return"], "metric": ["pearson_rnic"], "mean": [.01],
                              "p_value": [.1], "q_value": [.1], "final_pool_id": [f"pool-{year}-{method}-{seed}"]}).to_parquet(cell / "test/factor_significance.parquet", index=False)
                pd.DataFrame({"date": dates, "factor_id": "return-factor", "pearson_rnic": signal}).to_parquet(cell / "test/factor_rnic_daily.parquet", index=False)
    return config, raw, root


def test_five_window_report_and_exact_figure_source(rolling_report_data):
    config, raw, root = rolling_report_data
    result = build_report("recent", config)
    assert result["status"] == "complete"
    assert result["completed_cells"] == 60
    assert len(result["figures"]) == 10
    report = Path(result["report"])
    assert report.exists()
    assert "No p-values or t-statistics are averaged" in report.read_text()
    assert "annual-reset" in report.read_text()
    summary = pd.read_csv(result["tables"]["ic_summary"])
    assert len(summary) == 6 * 4 * 4
    assert set(summary.metric) == {"raw_ic", "raw_rank_ic", "rnic", "rank_rnic"}
    pairs = pd.read_csv(result["tables"]["paired_comparisons"])
    assert len(pairs) == 6 * 3 * 2
    assert {"seed_mean", "seed_sd"} <= set(summary)
    assert pairs[pairs.metric.eq("rnic") & pairs.second.eq("random")]["mean"].to_numpy() == pytest.approx(.009)
    factors = pd.read_parquet(report.parent / "factor_statistics.parquet")
    assert len(factors) == 60
    assert set(factors.test_year) == set(range(2021, 2026))
    assert factors.factor_daily_path.map(lambda p: Path(p).is_file()).all()
    for path in result["figures"]:
        assert Path(path).stat().st_size > 1000
    source = pd.read_csv(report.parent / "figures/dollar_neutral_2021_r1_oof_source.csv")
    original = pd.read_parquet(root / "test_2021/random/r1_oof/seed_0/test/dollar_neutral_daily.parquet")
    curve = source[(source.method == "random") & (source.seed == 0) & (source.cost_bps == 10)]
    np.testing.assert_allclose(curve.cumulative_return, np.cumprod(1 + original.net_return_10bps) - 1, atol=1e-14)
    svg = (report.parent / "figures/dollar_neutral_2021_r1_oof.svg").read_text()
    assert "Seed 0" in svg and "Seed 2" in svg and "GRPO" in svg


def test_seed_missingness_and_portfolio_estimand(rolling_report_data):
    _, raw, root = rolling_report_data
    cells = _load_cells(root, raw, None)
    cells[(2021, "random", "r1_oof", 0)]["ic"].loc[0, "rnic"] = np.nan
    seed_ic, method_ic, pairs, portfolio = _statistics(cells, {"bootstrap_samples": 12})
    row = method_ic[(method_ic.period == "2021") & (method_ic.method == "random") & (method_ic.metric == "rnic")].iloc[0]
    assert row["n"] == 26
    pair = pairs[(pairs.period == "2021") & (pairs.second == "random") & (pairs.metric == "rnic")].iloc[0]
    assert pair["n"] == 26
    summary = _portfolio_summary(portfolio)
    actual = summary[(summary.period == "overall") & (summary.method == "random") & (summary.cost_bps == 10)].iloc[0]
    individual = []
    arrays = []
    for seed in range(3):
        values = np.concatenate([cells[(y, "random", "r1_oof", seed)]["returns"].net_return_10bps for y in range(2021, 2026)])
        arrays.append(values)
        individual.append(return_metrics(values)["total_return"])
    assert actual.total_return_mean == pytest.approx(np.mean(individual))
    assert actual.total_return_seed_sd == pytest.approx(np.std(individual, ddof=1))
    pseudo_ensemble = return_metrics(np.mean(arrays, axis=0))["total_return"]
    assert abs(actual.total_return_mean - pseudo_ensemble) > 1e-4
    assert actual.liquidation_turnover_sum_mean == 5


@pytest.mark.parametrize("missing", ["test/metrics.json", "test/rnic_daily.parquet", "test/factor_significance.parquet"])
def test_missing_expected_artifact_refuses_report(rolling_report_data, missing):
    config, _, root = rolling_report_data
    (root / "test_2025/grpo_llm/r1_oof/seed_2" / missing).unlink()
    with pytest.raises(RuntimeError, match="missing_cells_or_files"):
        build_report("recent", config)
    assert not (root / "report/report.md").exists()
    assert json.loads((root / "report_status.json").read_text())["status"] == "incomplete"


def test_mismatched_seed_date_axis_is_not_inner_joined_away(rolling_report_data):
    _, raw, root = rolling_report_data
    path = root / "test_2021/random/r1_oof/seed_0/test/rnic_daily.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="axes differ"):
        _load_cells(root, raw, None)


def test_changed_frozen_window_identity_refuses_report(rolling_report_data, monkeypatch):
    config, _, root = rolling_report_data
    def reject(*args):
        raise RuntimeError("test opening refused: incompatible state")
    monkeypatch.setattr("rlalpha.evaluation.finalize._assert_experiment_frozen", reject)
    with pytest.raises(RuntimeError, match="incompatible state"):
        build_report("recent", config)
    assert not (root / "report/report.md").exists()


def test_method_subset_cannot_claim_complete_report(rolling_report_data):
    _, raw, root = rolling_report_data
    with pytest.raises(RuntimeError, match="method subset is incomplete"):
        _load_cells(root, raw, ["random"])


def test_empty_factor_statistics_cannot_claim_complete(rolling_report_data):
    config, _, root = rolling_report_data
    path = root / "test_2021/random/r1_oof/seed_0/test/factor_significance.parquet"
    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="factor statistics missing"):
        build_report("recent", config)
    assert json.loads((root / "report_status.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("artifact", ["final_pool.json", "test/metrics.json"])
def test_changed_finalized_content_refuses_report(rolling_report_data, artifact):
    config, _, root = rolling_report_data
    path = root / "test_2021/random/r1_oof/seed_0" / artifact
    content = json.loads(path.read_text())
    if artifact == "final_pool.json":
        content["expressions"] = ["$close"]
    else:
        content["test_ic_valid_days"] += 1
    _json(path, content)
    with pytest.raises(RuntimeError, match="finalized (inputs|metrics) changed"):
        build_report("recent", config)
    assert json.loads((root / "report_status.json").read_text())["status"] == "incomplete"
    assert not (root / "report/report.md").exists()
