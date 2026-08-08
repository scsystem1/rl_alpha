from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .contracts import CONTRACTS


class DataDiscoveryError(RuntimeError):
    pass


def discover_data_files(root: str | Path, strict: bool = True) -> dict[str, Path]:
    root = Path(root)
    matches: dict[str, list[Path]] = {name: [] for name in CONTRACTS}
    for path in sorted(root.glob("*.parquet")):
        try:
            columns = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
        for name, contract in CONTRACTS.items():
            if contract.required <= columns:
                matches[name].append(path)
    ambiguous = {name: values for name, values in matches.items() if len(values) > 1}
    missing = [name for name, values in matches.items() if not values]
    if strict and (ambiguous or missing):
        raise DataDiscoveryError(f"schema discovery failed; missing={missing}, ambiguous={ambiguous}")
    return {name: values[0] for name, values in matches.items() if len(values) == 1}

