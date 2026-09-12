"""Synthetic five-window smoke: real search, fit, backtest and report handoffs.

This deliberately uses a tiny deterministic proposal budget and synthetic
panels; it makes no claim about GPU training or real-market performance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rlalpha.config import load_paths
from rlalpha.data.store import SplitPanel
from rlalpha.dsl.parser import parse_expression
from rlalpha.evaluation.finalize import finalize_experiment
from rlalpha.matrix.runner import _expected_cell_identity
from rlalpha.reporting.build import build_report
from rlalpha.rolling import prepare_rolling_windows
from rlalpha.search.models import Candidate
from rlalpha.search.random_search import RandomSearcher
from rlalpha.search.run import run_search
from rlalpha.utils.experiment_log import update_progress
from rlalpha.utils.io import write_json, write_yaml


def test_five_window_search_to_finalization_to_report(tmp_path, monkeypatch):
    import rlalpha.evaluation.finalize as finalizer
    import rlalpha.search.run as search

    code_root = Path(__file__).parents[2]
    processed = tmp_path / "processed"
    dates = pd.bdate_range("2017-01-01", "2025-12-31")
    rng = np.random.default_rng(182)
    shape = (len(dates), 120)
    features = {f"${name}": rng.normal(size=shape) for name in ("open", "high", "low", "close", "volume", "return")}
    exposure = rng.normal(size=shape)
    label = .03 * features["$return"] + .02 * exposure + rng.normal(0, .04, shape)
    returns = rng.normal(0, .01, shape)
    risk = np.stack([np.ones(shape), exposure], axis=-1)
    reads = []

    class SyntheticStore:
        def __init__(self, root):
            assert Path(root) == processed

        def load_interval(self, name, start, end, history=252):
            reads.append((name, str(start), str(end)))
            selected = np.flatnonzero((dates >= start) & (dates <= end))
            first, stop = max(0, selected[0] - history), selected[-1] + 1
            source = slice(first, stop)
            target = slice(selected[0] - first, stop - first)
            local_label = label[source].copy()
            local_label[:target.start] = np.nan
            local_label[-21:] = np.nan
            mask = np.ones((stop - first, shape[1]), bool)
            return SplitPanel(name, dates[source], np.arange(shape[1]), {key: value[source] for key, value in features.items()},
                returns[source], local_label, mask, mask, risk[source], ("intercept", "style"), target, "synthetic-panel-v1")

    class OneFactorSearcher(RandomSearcher):
        def propose(self, context, n):
            return [Candidate(parse_expression("$return"), "random") for _ in range(n)]

    monkeypatch.setattr(search, "PanelStore", SyntheticStore)
    monkeypatch.setattr(finalizer, "PanelStore", SyntheticStore)
    monkeypatch.setattr(search, "searcher_for", lambda *a, **k: OneFactorSearcher(0))
    monkeypatch.setattr(search, "discover_data_files", lambda _: {})
    monkeypatch.setattr(search, "git_info", lambda _: {"synthetic": True})
    monkeypatch.setattr("rlalpha.manifest.git_info", lambda _: {"synthetic": True})
    monkeypatch.setattr("rlalpha.matrix.runner._repository_identity", lambda _: {"synthetic": True})
    monkeypatch.setattr(search, "_record_gpu_environment", lambda _: None)
    monkeypatch.setattr(search, "_score_validation", lambda *a, **k: (_ for _ in ()).throw(AssertionError("calibration feedback forbidden")))
    monkeypatch.setattr("rlalpha.evaluation.portfolio.project_fully_neutral", lambda *a, **k: (_ for _ in ()).throw(AssertionError("QP forbidden")))
    for filename in ("index.json", "build_manifest.yaml", "risk_build_manifest.yaml"):
        write_json(processed / "panel" / filename, {"fixture": "synthetic-five-window"})

    config = tmp_path / "experiment.yaml"
    write_yaml(config, {"paths": {"code_root": str(code_root), "processed_root": str(processed), "runs_root": str(tmp_path / "runs")},
        "rolling": {"test_years": list(range(2021, 2026))},
        "experiment": {"methods": ["random"], "rewards": ["r1_oof"], "seeds": [0], "search_steps": 1,
                       "proposal_group_size": 8, "pool_capacity": 20},
        "evaluation": {"bootstrap_samples": 24}})
    paths = load_paths(config)
    windows = prepare_rolling_windows(config, "synthetic")
    annual_wealth = []
    for year, child, child_id in windows:
        result = run_search(child, "random", "r1_oof", 0, 1, child_id)
        run_dir = Path(result["run_dir"])
        assert result["completed_steps"] == 1 and result["raw_proposals"] == 8
        assert reads[-1][0] == "train"
        # This is the matrix's acceptance commit; use its real frozen identity.
        identity = _expected_cell_identity(child.resolve(), paths, "random", "r1_oof", 0, 1)
        update_progress(run_dir / "progress.json", status="complete", cell_identity=identity, search_steps=1)
        evaluated = finalize_experiment(child_id, child)
        assert evaluated["random/r1_oof/seed_0"]["status"] == "complete"
        assert [row[0] for row in reads[-3:]] == ["train", "validation", "test"]
        metrics = evaluated["random/r1_oof/seed_0"]["metrics"]
        assert metrics["fit_period"] == "calibration"
        assert list(metrics["portfolios"]) == ["dollar_neutral"]
        assert not list((run_dir / "test").glob("fully_neutral*"))
        ic = pd.read_parquet(run_dir / "test/rnic_daily.parquet")
        pnl = pd.read_parquet(run_dir / "test/dollar_neutral_daily.parquet")
        assert ic.date.equals(pnl.date)
        assert ic.iloc[-21:][["raw_ic", "raw_rank_ic", "rnic", "rank_rnic"]].isna().all().all()
        assert (pd.DatetimeIndex(ic.date).year == year).all()
        annual_wealth.append(np.prod(1 + pnl.net_return_10bps.to_numpy()))

    report = build_report("synthetic", config)
    assert report["status"] == "complete" and report["completed_cells"] == 5
    assert len(report["figures"]) == 10
    summary = pd.read_csv(report["tables"]["ic_summary"])
    assert set(summary.metric) == {"raw_ic", "raw_rank_ic", "rnic", "rank_rnic"}
    assert len(summary) == 6 * 4
    overall = pd.read_csv(report["tables"]["portfolio_by_seed"])
    row = overall[(overall.period == "overall") & (overall.cost_bps == 10)].iloc[0]
    np.testing.assert_allclose(row.total_return + 1, np.prod(annual_wealth))
    for year in range(2021, 2026):
        source = pd.read_csv(Path(report["report"]).parent / f"figures/dollar_neutral_{year}_r1_oof_source.csv")
        pnl = pd.read_parquet(paths.runs_root / f"synthetic/test_{year}/random/r1_oof/seed_0/test/dollar_neutral_daily.parquet")
        np.testing.assert_allclose(source[source.cost_bps == 10].cumulative_return, np.cumprod(1 + pnl.net_return_10bps) - 1)
