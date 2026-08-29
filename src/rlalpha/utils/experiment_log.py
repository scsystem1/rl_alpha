from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import json

from .io import atomic_write_text, write_json


def append_event(path: str | Path, event: str, **fields: Any) -> None:
    """Append one compact, human-readable experiment event.

    Each run has a single writer.  Durable structured state is written through
    atomic replacement; this journal intentionally needs no sibling lock file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {event}{(' ' + details) if details else ''}\n")
        handle.flush()


def update_progress(path: str | Path, **fields: Any) -> dict[str, Any]:
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(fields)
    current["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    write_json(path, current)
    return current


def write_result_summary(
    path: str | Path,
    *,
    experiment_id: str,
    method: str,
    reward: str,
    seed: int,
    search_steps: int,
    ledger: dict[str, Any],
    pool_version: int,
    train_objective: float | None,
    validation_objective: float | None,
    expressions: Iterable[str],
    evaluation: dict[str, Any] | None = None,
) -> None:
    expressions = list(expressions)
    lines = [
        f"# {experiment_id}: {method}/{reward}/seed_{seed}",
        "",
        "- Status: complete",
        f"- Search steps: {ledger.get('completed_steps', 0)}/{search_steps}",
        f"- Valid unique factors: {ledger.get('valid_unique_evaluations', 0)}",
        f"- Raw proposals: {ledger.get('raw_proposals', 0)}",
        f"- Invalid: {ledger.get('invalid', 0)}",
        f"- Duplicates: {ledger.get('duplicates', 0)}",
        f"- Final pool version: {pool_version}",
        f"- Final pool size: {len(expressions)}",
        f"- Train objective: {train_objective}",
        f"- Validation objective: {validation_objective}",
        "",
        "## Final factors",
        "",
        *(f"{index}. `{expression}`" for index, expression in enumerate(expressions, 1)),
        "",
    ]
    if evaluation is not None:
        primary = evaluation.get("primary_pearson_rnic", evaluation.get("rnic", {}))
        rank = evaluation.get("rank_rnic", {})
        fully = evaluation.get("portfolios", {}).get("fully_neutral", {}).get("10bps", {})
        lines.extend(
            [
                "## Final evaluation",
                "",
                f"- Primary Pearson RNIC mean: {primary.get('mean')}",
                f"- Primary Pearson RNIC HAC t: {primary.get('hac_t')}",
                f"- Rank RNIC mean: {rank.get('mean')}",
                f"- Fully-neutral 10bps annual return: {fully.get('annual_return')}",
                f"- Fully-neutral 10bps Sharpe: {fully.get('sharpe')}",
                f"- Fully-neutral 10bps max drawdown: {fully.get('max_drawdown')}",
                "",
            ]
        )
    atomic_write_text(path, "\n".join(lines))
