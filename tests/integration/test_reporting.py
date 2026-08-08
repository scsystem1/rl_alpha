from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rlalpha.reporting.build import build_report


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _portfolio_metrics(sharpe: float) -> dict[str, object]:
    values = {"annual_return": 0.1, "sharpe": sharpe, "max_drawdown": -0.1, "average_turnover": 0.2, "average_gross": 1.0, "average_net": 0.0, "infeasible_days": 0, "missing_held_returns": 0}
    return {"0bps": values, "10bps": values, "max_realized_risk_exposure": 1e-9, "missing_held_exposure_days": 0}


def test_cross_method_report_generation_with_paired_outputs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    experiment = runs / "report_smoke"
    dates = pd.date_range("2022-01-03", periods=60, freq="B")
    for offset, method in enumerate(("random", "gp")):
        cell = experiment / method / "r0" / "seed_0"
        metrics = {"pool_size": 2, "average_pair_correlation": 0.2, "rnic": {"mean": 0.01 + offset * 0.01, "hac_t": 2.0, "bootstrap_95_ci": [0.0, 0.03]}, "portfolios": {"dollar_neutral": _portfolio_metrics(0.5 + offset), "fully_neutral": _portfolio_metrics(0.4 + offset)}}
        _json(cell / "test/metrics.json", metrics)
        _json(cell / "train_metrics.json", {"raw_proposals": 20, "valid_unique_evaluations": 10, "tokens": 0, "gpu_seconds": 0, "wall_seconds": 5})
        _json(cell / "final_pool.json", {"train": {"objective": 0.02}, "validation": {"objective": 0.01}})
        _json(cell / "checkpoint.json", {"pool_history": [{"admitted": True}]})
        pd.DataFrame({"valid": [True] * 10}).to_parquet(cell / "candidates.parquet", index=False)
        rnic = np.linspace(-0.02, 0.04, len(dates)) + offset * 0.01
        pd.DataFrame({"date": dates, "rnic": rnic}).to_parquet(cell / "test/rnic_daily.parquet", index=False)
        pd.DataFrame({"date": dates, "net_return_10bps": rnic / 100}).to_parquet(cell / "test/fully_neutral_daily.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"code_root": str(tmp_path), "raw_data_root": str(tmp_path), "processed_root": str(tmp_path), "cache_root": str(tmp_path), "runs_root": str(runs), "model_search_root": str(tmp_path), "alphagen_root": str(tmp_path), "quantevolver_root": str(tmp_path)}}), encoding="utf-8")
    result = build_report("report_smoke", config)
    assert result["completed_cells"] == 2
    assert result["paired_comparisons"] == 1
    assert (experiment / "search_efficiency.parquet").exists()
    assert (experiment / "paired_comparisons.csv").exists()
    assert "Paired comparisons" in (experiment / "report.md").read_text(encoding="utf-8")
