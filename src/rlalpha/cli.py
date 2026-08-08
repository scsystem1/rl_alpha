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
app.add_typer(data_app, name="data")
app.add_typer(risk_app, name="risk")
app.add_typer(factor_app, name="factor")


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


@risk_app.command("build")
def risk_build(config: Path = typer.Option(..., exists=True)) -> None:
    from .risk.builder import build_risk_panel

    paths = load_paths(config)
    report = build_risk_panel(paths.raw_data_root, paths.processed_root)
    typer.echo(json.dumps(report, indent=2, default=str))


@factor_app.command("eval")
def factor_eval(expr: str = typer.Option(...), split: str = typer.Option("train")) -> None:
    from .dsl.parser import parse_expression

    node = parse_expression(expr)
    typer.echo(json.dumps({"split": split, "canonical": node.canonical(), "hash": node.expr_hash, "lookback": node.lookback}))


if __name__ == "__main__":
    app()
