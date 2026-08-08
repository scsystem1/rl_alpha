from __future__ import annotations


def assert_neutral(diagnostics: list[dict[str, object]] | dict[str, object], tolerance: float = 1e-8) -> None:
    records = diagnostics if isinstance(diagnostics, list) else [diagnostics]
    failures = [record for record in records if float(record["max_residual_exposure"]) >= tolerance]
    if failures:
        raise ValueError(f"neutrality tolerance {tolerance} exceeded: {failures[:3]}")
