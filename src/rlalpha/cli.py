from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_paths
from .doctor import run_doctor

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
risk_app = typer.Typer(no_args_is_help=True)
factor_app = typer.Typer(no_args_is_help=True)
search_app = typer.Typer(no_args_is_help=True)
matrix_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(risk_app, name="risk")
app.add_typer(factor_app, name="factor")
app.add_typer(search_app, name="search")
app.add_typer(matrix_app, name="matrix")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(report_app, name="report")


@app.command()
def doctor(config: Optional[Path] = typer.Option(None, exists=True)) -> None:
    """Read-only environment, data, model and repository diagnostics."""
    typer.echo(json.dumps(run_doctor(load_paths(config)), indent=2, default=str))


@data_app.command("validate")
def data_validate(config: Path = typer.Option(..., exists=True)) -> None:
    from .data.validate import validate_raw_bundle

    paths = load_paths(config)
    report = validate_raw_bundle(paths.raw_data_root)
    typer.echo(json.dumps(report, indent=2, default=str))
    if not report["ok"]:
        raise typer.Exit(2)


@data_app.command("build")
def data_build(config: Path = typer.Option(..., exists=True)) -> None:
    from .data.panel import build_panel

    paths = load_paths(config)
    report = build_panel(paths.raw_data_root, paths.processed_root)
    typer.echo(json.dumps(report, indent=2, default=str))


@data_app.command("audit")
def data_audit(config: Path = typer.Option(..., exists=True), output: Path = typer.Option(Path("artifacts/data_audit"))) -> None:
    from .data.audit import run_data_audit

    paths = load_paths(config)
    report = run_data_audit(paths.raw_data_root, output)
    typer.echo(json.dumps(report, indent=2, default=str))
    if not report["ok"]:
        raise typer.Exit(2)


@risk_app.command("build")
def risk_build(config: Path = typer.Option(..., exists=True)) -> None:
    from .risk.builder import build_risk_panel

    paths = load_paths(config)
    report = build_risk_panel(paths.raw_data_root, paths.processed_root)
    typer.echo(json.dumps(report, indent=2, default=str))


@factor_app.command("eval")
def factor_eval(expr: str = typer.Option(...), split: str = typer.Option("train")) -> None:
    from .data.store import PanelStore
    from .dsl.parser import parse_expression

    node = parse_expression(expr)
    paths = load_paths(None)
    panel = PanelStore(paths.processed_root).load_split(split)
    signal = panel.evaluate(node)
    typer.echo(json.dumps({"split": split, "canonical": node.canonical(), "hash": node.expr_hash, "lookback": node.lookback, "shape": list(signal.shape), "finite": int(__import__("numpy").isfinite(signal).sum())}))


@search_app.command("run")
def search_run(method: str = typer.Option(...), reward: str = typer.Option(...), seed: int = typer.Option(0), budget: int = typer.Option(5000), experiment_id: str = typer.Option("preliminary_screen"), config: Path = typer.Option(Path("configs/experiment/preliminary_screen.yaml"), exists=True), resume: bool = typer.Option(True)) -> None:
    from .search.run import run_search

    typer.echo(json.dumps(run_search(config, method, reward, seed, budget, experiment_id, resume), indent=2, default=str))


@matrix_app.command("run")
def matrix_run(config: Path = typer.Option(..., exists=True), experiment_id: str = typer.Option("preliminary_screen"), resume: bool = typer.Option(True)) -> None:
    from .matrix.runner import run_matrix

    typer.echo(json.dumps(run_matrix(config, experiment_id, resume), indent=2, default=str))


@evaluate_app.command("run")
def evaluate_run(experiment_id: str = typer.Option(...), config: Path = typer.Option(Path("configs/experiment/preliminary_screen.yaml"), exists=True)) -> None:
    from .evaluation.finalize import finalize_experiment

    typer.echo(json.dumps(finalize_experiment(experiment_id, config), indent=2, default=str))


@report_app.command("build")
def report_build(experiment_id: str = typer.Option(...), config: Path = typer.Option(Path("configs/experiment/preliminary_screen.yaml"), exists=True)) -> None:
    from .reporting.build import build_report

    typer.echo(json.dumps(build_report(experiment_id, config), indent=2, default=str))


if __name__ == "__main__":
    app()
