from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import CONTRACTS
from .discovery import discover_data_files
from .splits import SPLITS
from .validate import validate_raw_bundle
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import atomic_write_text, write_json


PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "daily": ("PERMNO", "DlyCalDt"),
    "membership": ("PERMNO", "MbrStartDt", "MbrEndDt"),
    "market": ("DlyCalDt",),
    "ccm": ("gvkey", "lpermno", "linkdt", "linktype", "linkprim"),
    "fundamentals": ("gvkey", "datadate"),
    "delistings": ("PERMNO", "DelDlyDt"),
}

DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily": ("DlyCalDt",),
    "membership": ("MbrStartDt", "MbrEndDt"),
    "market": ("DlyCalDt",),
    "ccm": ("linkdt", "linkenddt"),
    "fundamentals": ("datadate",),
    "delistings": ("DelDlyDt",),
}


def _missingness(path: Path, dataset: str) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    columns = sorted(CONTRACTS[dataset].required)
    counts = {column: {"null": 0, "nonfinite": 0} for column in columns}
    total = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=131_072):
        total += batch.num_rows
        for column in columns:
            values = batch.column(batch.schema.get_field_index(column))
            counts[column]["null"] += int(values.null_count)
            if pa.types.is_floating(values.type):
                finite = pc.is_finite(values)
                counts[column]["nonfinite"] += int(pc.sum(pc.invert(pc.fill_null(finite, True))).as_py() or 0)
    rows = []
    for column in columns:
        null = counts[column]["null"]
        nonfinite = counts[column]["nonfinite"]
        rows.append({
            "dataset": dataset,
            "field": column,
            "row_count": total,
            "null_count": null,
            "null_rate": null / max(1, total),
            "nonfinite_nonnull_count": nonfinite,
            "primary_key_field": column in PRIMARY_KEYS[dataset],
        })
    keys = pd.read_parquet(path, columns=list(PRIMARY_KEYS[dataset]))
    rows.append({
        "dataset": dataset,
        "field": "__primary_key__",
        "row_count": total,
        "null_count": int(keys.isna().any(axis=1).sum()),
        "null_rate": float(keys.isna().any(axis=1).mean()),
        "nonfinite_nonnull_count": 0,
        "primary_key_field": True,
        "duplicate_key_count": int(keys.duplicated().sum()),
        "primary_key": list(PRIMARY_KEYS[dataset]),
    })
    return rows


def _date_coverage(path: Path, dataset: str) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path, columns=list(DATE_COLUMNS[dataset]))
    rows: list[dict[str, Any]] = []
    for column in DATE_COLUMNS[dataset]:
        values = pd.to_datetime(frame[column], errors="coerce")
        base = {
            "dataset": dataset,
            "date_field": column,
            "scope": "all",
            "minimum": values.min(),
            "maximum": values.max(),
            "valid_rows": int(values.notna().sum()),
            "unique_dates": int(values.nunique()),
        }
        rows.append(base)
        if dataset in {"daily", "market"} and column == "DlyCalDt":
            for name, split in SPLITS.items():
                selected = values.between(split.start, split.end)
                rows.append({**base, "scope": name, "minimum": values[selected].min(), "maximum": values[selected].max(), "valid_rows": int(selected.sum()), "unique_dates": int(values[selected].nunique())})
    return rows


def run_data_audit(raw_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    raw_root, output_root = Path(raw_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files = discover_data_files(raw_root, strict=True)
    inventory = {name: file_fingerprint(path) for name, path in sorted(files.items())}
    inventory_payload = {"schema_version": 1, "raw_root": str(raw_root.resolve()), "files": inventory}
    inventory_payload["inventory_hash"] = stable_hash(inventory_payload)
    write_json(output_root / "raw_file_inventory.json", inventory_payload)

    schema_report: dict[str, Any] = {"schema_version": 1, "datasets": {}}
    missingness_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for name, path in sorted(files.items()):
        parquet = pq.ParquetFile(path)
        fields = {field.name: str(field.type) for field in parquet.schema_arrow}
        required = sorted(CONTRACTS[name].required)
        schema_report["datasets"][name] = {
            "path": str(path.resolve()),
            "rows": parquet.metadata.num_rows,
            "primary_key": list(PRIMARY_KEYS[name]),
            "required_fields": required,
            "missing_required_fields": sorted(set(required) - set(fields)),
            "fields": fields,
        }
        missingness_rows.extend(_missingness(path, name))
        coverage_rows.extend(_date_coverage(path, name))
    schema_report["schema_hash"] = stable_hash(schema_report["datasets"])
    write_json(output_root / "schema_report.json", schema_report)
    pd.DataFrame(missingness_rows).to_parquet(output_root / "key_and_missingness_report.parquet", index=False)
    pd.DataFrame(coverage_rows).to_parquet(output_root / "date_coverage_report.parquet", index=False)

    gates = validate_raw_bundle(raw_root)
    report_lines = [
        "# Data audit report",
        "",
        f"Raw root: `{raw_root.resolve()}`",
        f"Inventory hash: `{inventory_payload['inventory_hash']}`",
        f"Schema hash: `{schema_report['schema_hash']}`",
        f"Hard-gate result: **{'PASS' if gates['ok'] else 'FAIL'}**",
        "",
        "## Tables",
        "",
        "| dataset | rows | date coverage | duplicate primary keys |",
        "|---|---:|---|---:|",
    ]
    missingness = pd.DataFrame(missingness_rows)
    coverage = pd.DataFrame(coverage_rows)
    for name in sorted(files):
        schema = schema_report["datasets"][name]
        dates = coverage[(coverage["dataset"] == name) & (coverage["scope"] == "all")]
        rendered_dates = "; ".join(f"{row.date_field}: {row.minimum} to {row.maximum}" for row in dates.itertuples())
        key_row = missingness[(missingness["dataset"] == name) & (missingness["field"] == "__primary_key__")].iloc[0]
        report_lines.append(f"| {name} | {schema['rows']} | {rendered_dates} | {int(key_row.get('duplicate_key_count', 0))} |")
    report_lines.extend([
        "",
        "## Hard gates",
        "",
        f"Failures: `{gates['failures']}`",
        f"Daily range: `{gates['date_range']}`; membership checkpoints: `{gates['membership_counts']}`.",
        f"CRSP capitalization identity median: `{gates['cap_identity_ratio_median']}`.",
        "",
        "Missing values in these reports retain source-null/non-finite distinctions. Adjustment failures, unresolved delisting returns, computation failures, and filtering decisions are recorded later in the panel build manifest rather than rewritten as zeros.",
        "",
        "## Trust boundary",
        "",
        "This audit validates the raw bundle. A formal experiment additionally requires a version-3 panel build, Balanced-22 diagnostics, point-in-time lineage sampling, and matching data/config/code/evaluator fingerprints.",
    ])
    atomic_write_text(output_root.parent.parent / "docs/data_audit_report.md", "\n".join(report_lines) + "\n")
    return {"ok": gates["ok"], "failures": gates["failures"], "output_root": str(output_root), "inventory_hash": inventory_payload["inventory_hash"], "schema_hash": schema_report["schema_hash"]}
