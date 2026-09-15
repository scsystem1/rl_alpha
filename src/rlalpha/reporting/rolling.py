"""Complete annual-window reports without treating seeds as market samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config import load_paths, load_yaml, resolve_data_evaluation
from ..evaluation.portfolio import return_metrics
from ..evaluation.statistics import bootstrap_date_indices, series_summary
from ..utils.io import atomic_write_text, write_json
from ..utils.hashing import stable_hash


IC_COLUMNS = ("raw_ic", "raw_rank_ic", "rnic", "rank_rnic")
COSTS = (0, 10)
METHOD_NAMES = {
    "random": "Random", "gp": "GP", "base_llm": "Base LLM", "grpo_llm": "GRPO",
    "quantevolver": "QuantEvolver", "alphasage": "AlphaSAGE",
}
METHOD_COLORS = {
    "random": "#666666", "gp": "#28659A", "base_llm": "#926400", "grpo_llm": "#B83240",
    "quantevolver": "#6B4C9A", "alphasage": "#23856D",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(root: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(root / f"{name}.csv", index=False)
    frame.to_parquet(root / f"{name}.parquet", index=False)


def _daily(path: Path, year: int, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not {"date", *columns} <= set(frame):
        raise RuntimeError(f"incomplete daily columns: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    dates = pd.DatetimeIndex(frame["date"])
    if not len(dates) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or not (dates.year == year).all():
        raise RuntimeError(f"invalid annual trading dates: {path}")
    return frame


def _load_cells(root: Path, raw: dict[str, Any], methods: list[str] | None) -> dict[tuple[int, str, str, int], dict[str, Any]]:
    experiment = raw["experiment"]
    combinations = experiment.get("cells") or [(m, r) for m in experiment["methods"] for r in experiment["rewards"]]
    selected = set(methods or [m for m, _ in combinations])
    if selected - {m for m, _ in combinations}:
        raise ValueError("report methods are not configured for this experiment")
    if selected != {m for m, _ in combinations}:
        raise RuntimeError("formal rolling report requires all configured methods; a method subset is incomplete")
    expected = [(int(y), m, r, int(s)) for y in raw["rolling"]["test_years"]
                for m, r in combinations if m in selected for s in experiment["seeds"]]
    required = ("train_metrics.json", "final_pool.json", "manifest.yaml", "progress.json", "test/finalization.json",
                "test/metrics.json", "test/rnic_daily.parquet", "test/dollar_neutral_daily.parquet",
                "test/factor_significance.parquet", "test/factor_rnic_daily.parquet")
    missing = []
    for y, m, r, s in expected:
        cell = root / f"test_{y}" / m / r / f"seed_{s}"
        missing.extend(str(cell / filename) for filename in required if not (cell / filename).is_file())
    if missing:
        raise RuntimeError(f"formal rolling report refused: missing_cells_or_files={missing}")
    scope = "_".join(sorted(set(methods or ())))
    for year in raw["rolling"]["test_years"]:
        marker = root / f"test_{year}" / (f"test_finalization_{scope}.json" if scope else "test_finalization.json")
        if not marker.exists() or _read(marker).get("status") != "complete":
            raise RuntimeError(f"formal rolling report refused: window finalization incomplete: {year}")
    result, axes, comparability = {}, {}, set()
    steps = int(experiment.get("search_steps", 100))
    for key in expected:
        year, method, reward, seed = key
        cell = root / f"test_{year}" / method / reward / f"seed_{seed}"
        state, train = _read(cell / "progress.json"), _read(cell / "train_metrics.json")
        if (state.get("status") != "complete" or int(state.get("search_steps", -1)) != steps
                or int(train.get("completed_steps", -1)) != steps
                or _read(cell / "test/finalization.json").get("status") != "complete"):
            raise RuntimeError(f"formal rolling report refused: incomplete or wrong-budget cell: {key}")
        manifest = yaml.safe_load((cell / "manifest.yaml").read_text(encoding="utf-8")) or {}
        panel_identity = tuple((Path(v["path"]).name, v["sha256"]) for v in manifest.get("panel_artifacts", []))
        if not panel_identity or not manifest.get("evaluator_version"):
            raise RuntimeError(f"formal rolling report refused: missing panel identity: {key}")
        comparability.add((panel_identity, manifest["evaluator_version"]))
        ic = _daily(cell / "test/rnic_daily.parquet", year, IC_COLUMNS)
        returns = _daily(cell / "test/dollar_neutral_daily.parquet", year, tuple(f"net_return_{c}bps" for c in COSTS))
        dates = pd.DatetimeIndex(ic["date"])
        if not dates.equals(pd.DatetimeIndex(returns["date"])) or (year in axes and not dates.equals(axes[year])):
            raise RuntimeError(f"formal rolling report refused: trading-date axes differ: {key}")
        axes[year] = dates
        metrics = _read(cell / "test/metrics.json")
        marker = _read(cell / "test/finalization.json")
        if stable_hash(metrics) != marker.get("metrics_hash"):
            raise RuntimeError(f"formal rolling report refused: finalized metrics changed: {key}")
        factor = pd.read_parquet(cell / "test/factor_significance.parquet")
        if factor.empty and metrics.get("pool_size", 0):
            raise RuntimeError(f"factor statistics missing for nonempty pool: {cell}")
        result[key] = {"path": cell, "ic": ic, "returns": returns, "train": train, "factor": factor,
                       "metrics": metrics, "pool": _read(cell / "final_pool.json")}
    if len(comparability) != 1:
        raise RuntimeError("formal rolling report refused: panel/evaluator identities differ")
    return result


def _summary_row(values: np.ndarray, dates: pd.DatetimeIndex, indices: np.ndarray, options: dict[str, Any]) -> dict[str, Any]:
    record = series_summary(values, hac_lag=int(options.get("hac_lag", 20)),
                            bootstrap_block=int(options.get("bootstrap_block_length", 20)),
                            bootstrap_samples=len(indices), seed=int(options.get("bootstrap_seed", 0)),
                            dates=dates, bootstrap_indices=indices)
    low, high = record.pop("bootstrap_95_ci")
    return {**record, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "calendar_days": len(dates), "valid_day_rate": record["n"] / len(dates)}


def _statistics(cells: dict, options: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted({key[0] for key in cells})
    groups = sorted({key[1:3] for key in cells})
    seeds = sorted({key[3] for key in cells})
    seed_ic, method_ic, paired, portfolio = [], [], [], []
    for period in [*years, "overall"]:
        active = years if period == "overall" else [period]
        dates = pd.DatetimeIndex(np.concatenate([cells[(y, *groups[0], seeds[0])]["ic"]["date"].to_numpy() for y in active]))
        indices = bootstrap_date_indices(dates, block=int(options.get("bootstrap_block_length", 20)),
                                         samples=int(options.get("bootstrap_samples", 2000)), seed=int(options.get("bootstrap_seed", 0)))
        matrices = {}
        for method, reward in groups:
            for metric in IC_COLUMNS:
                values = np.column_stack([np.concatenate([cells[(y, method, reward, s)]["ic"][metric].to_numpy(float) for y in active]) for s in seeds])
                matrices[(method, reward, metric)] = values
                # np.mean intentionally propagates a missing value in ANY declared seed.
                mean = values.mean(axis=1)
                metadata = {"period": str(period), "method": method, "reward": reward, "metric": metric}
                per_seed_means = np.asarray([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in values.T])
                method_ic.append({**metadata, "seeds": len(seeds), "seed_mean": float(per_seed_means.mean()),
                                  "seed_sd": float(per_seed_means.std(ddof=1)) if len(seeds) > 1 else float("nan"),
                                  **_summary_row(mean, dates, indices, options)})
                for column, seed in enumerate(seeds):
                    seed_ic.append({**metadata, "seed": seed, **_summary_row(values[:, column], dates, indices, options)})
            for seed in seeds:
                returns = pd.concat([cells[(y, method, reward, seed)]["returns"] for y in active], ignore_index=True)
                for cost in COSTS:
                    record = return_metrics(returns[f"net_return_{cost}bps"].to_numpy(float))
                    record.update({"period": str(period), "method": method, "reward": reward, "seed": seed, "cost_bps": cost})
                    for name in ("turnover", "liquidation_turnover", "missing_held_returns", "missing_held_return_weight", "infeasible", "gross_weight", "net_weight"):
                        if name in returns:
                            values = returns[name].to_numpy(float)
                            record[f"{name}_sum"] = float(values.sum())
                            record[f"{name}_mean"] = float(values.mean())
                    if "missing_held_returns" in returns:
                        record["missing_held_return_days"] = int((returns.missing_held_returns > 0).sum())
                    if {"gross_weight", "missing_held_return_weight"} <= set(returns):
                        gross = float(returns.gross_weight.sum())
                        missing = returns.missing_held_return_weight.to_numpy(float)
                        record["held_return_weight_coverage"] = float(np.clip(1 - missing.sum() / gross, 0, 1)) if gross > 0 else 1.0
                        record["maximum_missing_held_return_weight"] = float(missing.max())
                    portfolio.append(record)
        for reward in sorted({r for _, r in groups}):
            if ("grpo_llm", reward) not in groups:
                continue
            for baseline in ("random", "gp", "base_llm"):
                if (baseline, reward) not in groups:
                    continue
                for metric in ("rnic", "rank_rnic"):
                    delta = matrices[("grpo_llm", reward, metric)] - matrices[(baseline, reward, metric)]
                    paired.append({"period": str(period), "reward": reward, "first": "grpo_llm", "second": baseline,
                                   "metric": metric, "seeds": len(seeds), **_summary_row(delta.mean(axis=1), dates, indices, options)})
    return tuple(pd.DataFrame(rows) for rows in (seed_ic, method_ic, paired, portfolio))


def _portfolio_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("total_return", "cagr", "annualized_mean_return", "annual_volatility", "sharpe", "max_drawdown", "turnover_mean", "liquidation_turnover_sum",
               "missing_held_return_days", "held_return_weight_coverage", "maximum_missing_held_return_weight")
    for key, group in frame.groupby(["period", "method", "reward", "cost_bps"], sort=False):
        record = dict(zip(("period", "method", "reward", "cost_bps"), key))
        record["seeds"] = len(group)
        record["invalid_seed_paths"] = int(group["invalid_return_path"].sum())
        for metric in metrics:
            if metric in group:
                values = group[metric].to_numpy(float)
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def _annual_figures(cells: dict, root: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure_root = root / "figures"
    figure_root.mkdir(exist_ok=True)
    outputs = []
    seeds = sorted({key[3] for key in cells})
    groups = sorted({key[1:3] for key in cells})
    style = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
             "figure.facecolor": "#fffff8", "axes.facecolor": "#fffff8", "axes.spines.top": False,
             "axes.spines.right": False, "axes.grid": False, "axes.edgecolor": "#B8B8B0",
             "text.color": "#222222", "axes.labelcolor": "#555555", "xtick.color": "#555555", "ytick.color": "#555555"}
    for year in sorted({key[0] for key in cells}):
        for reward in sorted({r for _, r in groups}):
            source, curves = [], {}
            for method, candidate_reward in groups:
                if candidate_reward != reward:
                    continue
                for seed in seeds:
                    frame = cells[(year, method, reward, seed)]["returns"]
                    for cost in COSTS:
                        returns = frame[f"net_return_{cost}bps"].to_numpy(float)
                        curve = np.cumprod(1.0 + returns) - 1.0
                        curves[(method, seed, cost)] = (pd.DatetimeIndex(frame["date"]), curve)
                        source.append(pd.DataFrame({"date": frame["date"], "test_year": year, "method": method, "reward": reward,
                                                    "seed": seed, "cost_bps": cost, "daily_return": returns, "cumulative_return": curve}))
            stem = f"dollar_neutral_{year}_{reward}"
            pd.concat(source, ignore_index=True).to_csv(figure_root / f"{stem}_source.csv", index=False)
            finite = np.concatenate([v[1][np.isfinite(v[1])] for v in curves.values()])
            low, high = min(0.0, float(finite.min())) if len(finite) else 0.0, max(0.0, float(finite.max())) if len(finite) else 0.0
            span = max(high - low, .02)
            with plt.rc_context(style):
                fig, axes = plt.subplots(len(seeds), len(COSTS), figsize=(15, 3.15 * len(seeds)), sharey=True, squeeze=False)
                for row, seed in enumerate(seeds):
                    for col, cost in enumerate(COSTS):
                        ax = axes[row, col]
                        ax.axhline(0.0, color="#AAAAA0", linewidth=.6)
                        endpoints = []
                        for method, candidate_reward in groups:
                            if candidate_reward != reward:
                                continue
                            dates, curve = curves[(method, seed, cost)]
                            color = METHOD_COLORS.get(method, "#666666")
                            ax.plot(dates, curve, color=color, linewidth=1.25)
                            if np.isfinite(curve[-1]):
                                endpoints.append((float(curve[-1]), method, color, dates[-1]))
                            else:
                                ax.text(.02, .95 - .06 * len(endpoints), f"{METHOD_NAMES.get(method, method)}: invalid path", transform=ax.transAxes, color=color, va="top")
                        # Spread endpoint labels, with short leaders preserving their true endpoints.
                        label_positions = []
                        for value, method, color, date in sorted(endpoints):
                            label_y = max(value, label_positions[-1] + .055 * span) if label_positions else value
                            label_positions.append(label_y)
                        if label_positions and label_positions[-1] > high + .04 * span:
                            label_positions = [v - (label_positions[-1] - high - .04 * span) for v in label_positions]
                        for (value, method, color, date), label_y in zip(sorted(endpoints), label_positions):
                            ax.annotate(METHOD_NAMES.get(method, method), xy=(date, value), xytext=(date + pd.Timedelta(days=9), label_y),
                                        color=color, fontsize=9, va="center", arrowprops={"arrowstyle": "-", "color": color, "lw": .5})
                        ax.set_title(f"Seed {seed} · {cost} bps one-way cost", loc="left", fontsize=11)
                        ax.set_ylim(low - .07 * span, high + .1 * span)
                        dates = next(v[0] for k, v in curves.items() if k[1:] == (seed, cost))
                        ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=53))
                        ax.spines["bottom"].set_bounds(mdates.date2num(dates[0]), mdates.date2num(dates[-1]))
                        ax.spines["left"].set_bounds(low, high if high > low else low + .02)
                        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                        if col == 0:
                            ax.set_ylabel("Cumulative net return")
                fig.suptitle(f"{year}: dollar-neutral paths across independent search seeds", x=.065, ha="left", fontsize=16)
                fig.text(.065, .94, f"{reward} · annual cash reset · year-end liquidation included · identical y-scale · exact daily compounding", fontsize=10, color="#555555")
                fig.subplots_adjust(left=.065, right=.96, top=.88, bottom=.055, hspace=.35, wspace=.12)
                for extension in ("png", "svg"):
                    destination = figure_root / f"{stem}.{extension}"
                    fig.savefig(destination, dpi=150, facecolor=style["figure.facecolor"])
                    outputs.append(str(destination))
                plt.close(fig)
    return outputs


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    # Avoid a report-only dependency on the optional tabulate package.
    columns = [c for c in columns if c in frame]
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(cell(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None))])


def build_rolling_report(experiment_id: str, config: str | Path, methods: list[str] | None = None) -> dict[str, Any]:
    from ..evaluation.finalize import _assert_experiment_frozen, cell_input_hash, cell_finalization_scope_hash
    from ..rolling import prepare_rolling_windows

    raw, paths = load_yaml(config), load_paths(config)
    root = paths.runs_root / experiment_id
    scope = "_".join(sorted(set(methods or ())))
    destination = root / (f"report_{scope}" if scope else "report")
    try:
        window_configs = {}
        for year, child_config, child_id in prepare_rolling_windows(config, experiment_id):
            child = load_yaml(child_config)
            _assert_experiment_frozen(child_config, load_paths(child_config), paths.runs_root / child_id, child["experiment"], methods)
            window_configs[year] = child
        cells = _load_cells(root, raw, methods)
        for (year, method, reward, seed), cell in cells.items():
            child = window_configs[year]
            data, evaluation = resolve_data_evaluation(child, paths.code_root)
            expected = cell_input_hash(cell["path"], paths.processed_root, evaluation, data, child.get("protocol"),
                                       cell_finalization_scope_hash(f"{method}/{reward}/seed_{seed}"))
            marker = _read(cell["path"] / "test/finalization.json")
            if marker.get("input_hash") != expected or cell["metrics"].get("input_hash") != expected:
                raise RuntimeError(f"formal rolling report refused: finalized inputs changed: {(year, method, reward, seed)}")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / (f"report_status_{scope}.json" if scope else "report_status.json"), {"status": "incomplete", "error": str(exc)})
        raise
    destination.mkdir(parents=True, exist_ok=True)
    _, options = resolve_data_evaluation(raw, paths.code_root)
    seed_ic, method_ic, paired, portfolio = _statistics(cells, options)
    portfolio_summary = _portfolio_summary(portfolio)
    search, factor = [], []
    for (year, method, reward, seed), cell in cells.items():
        identity = {"test_year": year, "method": method, "reward": reward, "seed": seed}
        train, metrics, pool = cell["train"], cell["metrics"], cell["pool"]
        search.append({**identity, "completed_steps": train.get("completed_steps"), "raw_proposals": train.get("raw_proposals"),
                       "valid_unique_evaluations": train.get("valid_unique_evaluations"), "tokens": train.get("tokens"),
                       "gpu_hours": float(train.get("gpu_seconds", 0.0)) / 3600, "wall_hours": float(train.get("wall_seconds", 0.0)) / 3600,
                       "pool_size": metrics.get("pool_size"), "train_objective": pool.get("train", {}).get("objective"),
                       "fit_valid_days": metrics.get("fit_valid_days"), "test_ic_valid_days": metrics.get("test_ic_valid_days"),
                       "test_trade_valid_days": metrics.get("test_trade_valid_days"), "test_ic_observations": metrics.get("test_ic_observations"),
                       "factor_daily_path": str(cell["path"] / "test/factor_rnic_daily.parquet"), "cell_path": str(cell["path"])})
        stats = cell["factor"]
        stats = stats.assign(**identity, factor_daily_path=str(cell["path"] / "test/factor_rnic_daily.parquet"))
        factor.append(stats)
    tables = {"ic_by_seed": seed_ic, "ic_summary": method_ic, "paired_comparisons": paired,
              "portfolio_by_seed": portfolio, "portfolio_summary": portfolio_summary,
              "search_efficiency": pd.DataFrame(search), "factor_statistics": pd.concat(factor, ignore_index=True)}
    for name, frame in tables.items():
        _table(destination, name, frame)
    figures = _annual_figures(cells, destination)
    years = sorted({key[0] for key in cells})
    ridge_fit_period = str(options.get("ridge_fit_period", "calibration"))
    window_description = (
        "Search and ridge fitting use the same two complete calendar training years; "
        "there is no distinct validation/calibration period, and the next calendar year is test."
        if ridge_fit_period == "train"
        else "Search windows use two years, calibration uses the following half-year, and test covers the next calendar year. The terminal pool is frozen before calibration; only ridge weights use calibration labels."
    )
    text = [f"# {experiment_id}: recent-alpha rolling evaluation",
            f"Complete: {len(cells)} cells; test years {', '.join(map(str, years))}. {window_description}",
            "## Estimands and uncertainty",
            "IC is the equal-trading-day mean of daily cross-sectional correlations. Raw IC/raw rank IC are diagnostics; RNIC/rank RNIC use the declared risk-model residuals. Overall estimates concatenate annual daily observations, preserving missing label dates. Method-level IC first averages all declared seeds on the same date; any missing seed makes that date missing. `seed_mean` and `seed_sd` separately summarize the per-seed IC means on each seed's available dates. Seeds are repeated searches, not independent market histories. Inference is conditional on these fixed search seeds and observed historical years.",
            f"RNIC and rank RNIC report a two-sided HAC test of zero mean (lag {options.get('hac_lag', 20)}) and a 95% moving-block bootstrap interval (block {options.get('bootstrap_block_length', 20)}, {options.get('bootstrap_samples', 2000)} draws). All comparisons reuse the same sampled date indices within each period; blocks stay within calendar years. GRPO comparisons subtract the same baseline seed on the same date before averaging seeds. No p-values or t-statistics are averaged. Pairwise p-values are unadjusted planned comparisons; per-factor q-values retain their per-cell testing families.",
            "Each portfolio starts from cash at the beginning of its test year and liquidates at the last close with transaction costs. Overall returns concatenate those annual-reset daily paths separately for each seed. Total return compounds daily returns, CAGR uses 252 trading days per year, Sharpe uses zero risk-free rate and sample volatility, and drawdown includes initial wealth 1. Across-seed tables report mean and sample SD of these already-computed statistics; they do not construct a daily averaged seed portfolio. An invalid seed path makes its mean/SD unavailable. Calibration fit diagnostics are not out-of-sample results.",
            "## Overall IC",
            _markdown(method_ic[method_ic.period.eq("overall")], ["method", "reward", "metric", "mean", "seed_mean", "seed_sd", "n", "hac_se", "hac_t", "p_value", "bootstrap_ci_low", "bootstrap_ci_high"]),
            "## Overall dollar-neutral portfolios",
            _markdown(portfolio_summary[portfolio_summary.period.eq("overall")], ["method", "reward", "cost_bps", "total_return_mean", "total_return_seed_sd", "cagr_mean", "cagr_seed_sd", "sharpe_mean", "sharpe_seed_sd", "max_drawdown_mean", "max_drawdown_seed_sd"]),
            "## GRPO paired comparisons",
            _markdown(paired[paired.period.eq("overall") & paired.metric.isin(("rnic", "rank_rnic"))] if len(paired) else paired,
                      ["first", "second", "reward", "metric", "mean", "n", "hac_t", "p_value", "bootstrap_ci_low", "bootstrap_ci_high"]),
            "## Trading diagnostics",
            _markdown(portfolio_summary[portfolio_summary.period.eq("overall") & portfolio_summary.cost_bps.eq(10)],
                      ["method", "reward", "turnover_mean_mean", "liquidation_turnover_sum_mean", "missing_held_return_days_mean", "held_return_weight_coverage_mean", "maximum_missing_held_return_weight_mean"]),
            "## Search cost and coverage",
            _markdown(tables["search_efficiency"], ["test_year", "method", "seed", "completed_steps", "raw_proposals", "valid_unique_evaluations", "gpu_hours", "wall_hours", "pool_size", "fit_valid_days", "test_ic_valid_days"])]
    for year in years:
        text.extend([f"## {year}", _markdown(method_ic[method_ic.period.eq(str(year))], ["method", "reward", "metric", "mean", "seed_mean", "seed_sd", "n", "hac_t", "p_value", "bootstrap_ci_low", "bootstrap_ci_high"]),
                     _markdown(portfolio_summary[portfolio_summary.period.eq(str(year))], ["method", "reward", "cost_bps", "total_return_mean", "total_return_seed_sd", "cagr_mean", "cagr_seed_sd", "sharpe_mean", "sharpe_seed_sd", "max_drawdown_mean", "max_drawdown_seed_sd"])])
        for path in figures:
            if f"_{year}_" in path and path.endswith(".png"):
                relative = Path(path).relative_to(destination)
                text.append(f"![{year} dollar-neutral daily cumulative return, each seed and cost separately]({relative.as_posix()})")
    text.extend(["## Individual factor results",
                 f"The [factor statistics table](factor_statistics.csv) contains {len(tables['factor_statistics'])} factor × window × seed × metric rows, including formula identities, final weights, RNIC/rank RNIC tests and per-cell FDR q-values. Each row links its daily source and final-pool lineage; formulas found in different windows are not pooled into a fictitious long-lived factor.",
                 "## Reusable tables and diagnostics", "\n".join(f"- [{name} CSV]({name}.csv) · [Parquet]({name}.parquet)" for name in tables),
                 "Search efficiency includes raw/unique proposal counts, completed steps, GPU and wall time, pool size and data coverage. Portfolio seed tables include turnover, year-end liquidation turnover and missing held-return diagnostics. Factor tables preserve window, formula, seed and final-pool lineage; their daily source paths are linked rather than pooling different formulas across years. Every plotted path has an adjacent `*_source.csv`; figures are available as PNG and SVG.",
                 "These are retrospective results on previously inspected market history. Short-window performance and differences from old pools do not by themselves establish causal alpha decay. Dollar neutrality does not imply zero exposure to the full risk model."])
    report_path = destination / "report.md"
    atomic_write_text(report_path, "\n\n".join(text) + "\n")
    result = {"status": "complete", "experiment_id": experiment_id, "completed_cells": len(cells), "test_years": years,
              "report": str(report_path), "figures": figures, "tables": {name: str(destination / f"{name}.csv") for name in tables}}
    write_json(root / (f"report_status_{scope}.json" if scope else "report_status.json"), result)
    return result
