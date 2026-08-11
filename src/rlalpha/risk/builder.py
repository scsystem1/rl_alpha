from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..data.discovery import discover_data_files
from ..data.fundamentals import compute_accounting_exposures, filter_standard_fundamentals, select_ccm_links
from ..utils.hashing import file_fingerprint, stable_hash
from ..utils.io import write_json, write_yaml
from .exposures import STYLE_NAMES, _rolling_market_regression, preprocess_exposures

RISK_ARTIFACT_VERSION = 3


def _dense(frame: pd.DataFrame, column: str, dates: pd.DatetimeIndex, permnos: np.ndarray) -> np.ndarray:
    result = np.full((len(dates), len(permnos)), np.nan, dtype=np.float64)
    date_codes = pd.Categorical(pd.to_datetime(frame["DlyCalDt"]), categories=dates).codes
    permno_codes = pd.Categorical(pd.to_numeric(frame["PERMNO"]).astype(int), categories=permnos).codes
    result[date_codes, permno_codes] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    return result


def _rolling_market_styles(returns: np.ndarray, close: np.ndarray, volume: np.ndarray, cap: np.ndarray, market: np.ndarray) -> dict[str, np.ndarray]:
    days, assets = returns.shape
    output = {name: np.full((days, assets), np.nan, dtype=np.float64) for name in STYLE_NAMES[:6]}
    output["size"] = np.log(np.where(cap > 0, cap, np.nan))
    market_series = pd.Series(market)
    for asset in range(assets):
        stock = pd.Series(returns[:, asset])
        beta_series, residual_volatility = _rolling_market_regression(stock, market_series)
        beta = beta_series.to_numpy(float)
        output["beta_252"][:, asset] = beta
        output["resid_vol_252"][:, asset] = residual_volatility.to_numpy(float)
        log_return = np.log1p(stock.where(stock > -1))
        output["reversal_1m"][:, asset] = np.expm1(log_return.rolling(21, min_periods=21).sum()).to_numpy(float)
        output["momentum_12_1"][:, asset] = np.expm1(log_return.shift(21).rolling(232, min_periods=232).sum()).to_numpy(float)
        illiquidity = stock.abs() / (pd.Series(np.abs(close[:, asset] * volume[:, asset])) + 1e-12)
        output["amihud_20"][:, asset] = np.log(illiquidity.rolling(20, min_periods=15).mean()).to_numpy(float)
    return output


def _accounting_arrays(records: pd.DataFrame, dates: pd.DatetimeIndex, permnos: np.ndarray, cap: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = ["book_equity", "operating_profitability", "investment", "leverage"]
    arrays = {name: np.full((len(dates), len(permnos)), np.nan, dtype=np.float64) for name in names}
    comp_sic = np.full((len(dates), len(permnos)), np.nan, dtype=np.float64)
    permno_to_index = {int(value): index for index, value in enumerate(permnos)}
    for permno, group in records.groupby("PERMNO", sort=False):
        if not np.isfinite(permno) or int(permno) not in permno_to_index:
            continue
        asset = permno_to_index[int(permno)]
        group = group.sort_values("available_date").drop_duplicates("available_date", keep="last")
        available = pd.DatetimeIndex(group["available_date"])
        locations = np.searchsorted(available.to_numpy(), dates.to_numpy(), side="right") - 1
        valid = locations >= 0
        clipped = np.maximum(locations, 0)
        expiry = available[clipped] + pd.DateOffset(months=18)
        valid &= dates <= expiry
        link_start = pd.DatetimeIndex(pd.to_datetime(group["linkdt"], errors="coerce"))[clipped]
        link_end = pd.DatetimeIndex(pd.to_datetime(group["linkenddt"], errors="coerce"))[clipped]
        valid &= (dates >= link_start) & (link_end.isna() | (dates <= link_end))
        for name in names:
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(float)
            arrays[name][valid, asset] = values[clipped[valid]]
        sic_values = pd.to_numeric(group["sich"], errors="coerce").to_numpy(float)
        comp_sic[valid, asset] = sic_values[clipped[valid]]
    arrays["book_to_market"] = arrays.pop("book_equity") * 1000.0 / np.where(cap > 0, cap, np.nan)
    return arrays, comp_sic


def build_risk_panel(raw_root: str | Path, processed_root: str | Path) -> dict[str, Any]:
    files = discover_data_files(raw_root)
    panel_root = Path(processed_root) / "panel"
    import yaml

    with (panel_root / "build_manifest.yaml").open(encoding="utf-8") as handle:
        panel_manifest = yaml.safe_load(handle) or {}
    if int(panel_manifest.get("artifact_version", -1)) < 3:
        raise RuntimeError("risk build requires a version-3 membership-aware panel")
    index = pd.read_json(panel_root / "index.json", typ="series")
    dates = pd.DatetimeIndex(index["dates"])
    permnos = np.asarray(index["permnos"], dtype=np.int64)
    relevant_files = {name: file_fingerprint(files[name]) for name in ("market", "ccm", "fundamentals")}
    source_fingerprint = stable_hash({name: {"size": item["size"], "sha256": item["sha256"]} for name, item in relevant_files.items()})
    code_fingerprint = stable_hash([{key: value for key, value in file_fingerprint(path).items() if key in {"size", "sha256"}} for path in (Path(__file__), Path(__file__).with_name("exposures.py"), Path(__file__).with_name("ff12.py"), Path(__file__).parents[1] / "data/fundamentals.py")])
    build_identity = stable_hash({"artifact_version": RISK_ARTIFACT_VERSION, "source_fingerprint": source_fingerprint, "panel_build_fingerprint": panel_manifest["build_fingerprint"], "code_fingerprint": code_fingerprint, "definition": "Balanced-22 same-window-market-regression-v3"})
    manifest_path = panel_root / "risk_build_manifest.yaml"
    risk_path = panel_root / "risk_exposures.zarr"
    if manifest_path.exists() and risk_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            prior = yaml.safe_load(handle) or {}
        if prior.get("build_identity") == build_identity:
            return {**prior, "reused": True}
    daily_columns = ["PERMNO", "DlyCalDt", "DlyRet", "adj_close", "adj_volume", "DlyCap", "SICCD"]
    # This adjusted layer contains the audited/final total return, including
    # the explicit delisting rule used by labels and portfolio evaluation.
    daily = pd.read_parquet(panel_root / "daily", columns=daily_columns)
    returns, close = _dense(daily, "DlyRet", dates, permnos), _dense(daily, "adj_close", dates, permnos)
    volume, cap, crsp_sic = _dense(daily, "adj_volume", dates, permnos), _dense(daily, "DlyCap", dates, permnos), _dense(daily, "SICCD", dates, permnos)
    market_frame = pd.read_parquet(files["market"], columns=["DlyCalDt", "vwretd"]).set_index("DlyCalDt").reindex(dates)
    styles = _rolling_market_styles(returns, close, volume, cap, market_frame["vwretd"].to_numpy(float))
    fundamentals = filter_standard_fundamentals(pd.read_parquet(files["fundamentals"]))
    fundamentals, accounting_audit = compute_accounting_exposures(fundamentals)
    linked = select_ccm_links(pd.read_parquet(files["ccm"]), fundamentals)
    accounting, comp_sic = _accounting_arrays(linked, dates, permnos, cap)
    styles.update(accounting)
    sic = np.where(np.isfinite(crsp_sic), crsp_sic, comp_sic)
    membership = zarr.open_group(str(panel_root / "membership.zarr"), mode="r")["membership"][:]
    exposures = np.full((len(dates), len(permnos), 22), np.nan, dtype=np.float32)
    daily_diagnostics = []
    for day, date in enumerate(dates):
        active = membership[day]
        if active.sum() == 0:
            continue
        style_frame = pd.DataFrame({name: styles[name][day, active] for name in STYLE_NAMES})
        result = preprocess_exposures(style_frame, pd.Series(sic[day, active]))
        exposures[day, active] = result.matrix.astype(np.float32)
        disabled = [name for name, record in result.diagnostics["styles"].items() if record["disabled"]]
        daily_diagnostics.append({"date": str(date.date()), "n_members": int(active.sum()), "rank": result.diagnostics["rank"], "condition_number": result.diagnostics["condition_number"], "disabled_styles": disabled})
    formal_days = [record for record in daily_diagnostics if pd.Timestamp(record["date"]) >= pd.Timestamp("2010-01-01")]
    rank_failures = [record for record in formal_days if int(record["rank"]) < 22]
    condition_failures = [record for record in formal_days if not np.isfinite(record["condition_number"]) or float(record["condition_number"]) > 1e8]
    if rank_failures or condition_failures:
        raise RuntimeError(f"Balanced-22 hard gate failed: rank_days={len(rank_failures)}, condition_days={len(condition_failures)}")
    complete_exposure = np.isfinite(exposures).all(axis=2)
    member_complete_rate = float(complete_exposure[membership].mean()) if membership.any() else float("nan")
    formal_mask = np.asarray(dates >= pd.Timestamp("2010-01-01"))[:, None] & membership
    formal_member_complete_rate = float(complete_exposure[formal_mask].mean()) if formal_mask.any() else float("nan")
    if not np.isfinite(formal_member_complete_rate) or formal_member_complete_rate < 0.999:
        raise RuntimeError(f"Balanced-22 coverage hard gate failed: formal_member_complete_rate={formal_member_complete_rate}")
    temporary_root = Path(tempfile.mkdtemp(prefix=".risk-build-", dir=panel_root))
    temporary_risk = temporary_root / "risk_exposures.zarr"
    group = zarr.open_group(str(temporary_risk), mode="w")
    group.create_array("exposures", data=exposures, chunks=(21, min(256, len(permnos)), 22), overwrite=True)
    group.attrs["columns"] = list(preprocess_exposures(pd.DataFrame({name: [0.0] * 40 for name in STYLE_NAMES}), pd.Series([9999] * 40)).columns)
    write_json(temporary_root / "risk_diagnostics.json", {"days": daily_diagnostics, "accounting": accounting_audit})
    lineage_columns = [column for column in ("PERMNO", "gvkey", "datadate", "available_date", "linkdt", "linkenddt", "linktype", "linkprim", "book_equity", "operating_profitability", "investment", "leverage") if column in linked]
    linked[lineage_columns].sort_values(["PERMNO", "available_date", "gvkey"], kind="stable").to_parquet(temporary_root / "fundamental_lineage.parquet", index=False)
    backup = None
    if risk_path.exists():
        prior_hash = str((prior if "prior" in locals() else {}).get("build_fingerprint", "unknown"))[:12]
        backup = panel_root / f"risk_exposures.legacy_unverified.{prior_hash}.zarr"
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite legacy risk backup: {backup}")
        os.replace(risk_path, backup)
    try:
        os.replace(temporary_risk, risk_path)
    except Exception:
        if backup is not None and backup.exists() and not risk_path.exists():
            os.replace(backup, risk_path)
        raise
    os.replace(temporary_root / "risk_diagnostics.json", panel_root / "risk_diagnostics.json")
    os.replace(temporary_root / "fundamental_lineage.parquet", panel_root / "fundamental_lineage.parquet")
    temporary_root.rmdir()
    lineage_fingerprint = file_fingerprint(panel_root / "fundamental_lineage.parquet")
    manifest = {"artifact_version": RISK_ARTIFACT_VERSION, "build_identity": build_identity, "source_fingerprint": source_fingerprint, "source_files": relevant_files, "panel_build_fingerprint": panel_manifest["build_fingerprint"], "code_fingerprint": code_fingerprint, "definition": "Balanced-22 same-window-market-regression-v3", "shape": list(exposures.shape), "finite_rate": float(np.isfinite(exposures).mean()), "member_complete_rate": member_complete_rate, "formal_member_complete_rate": formal_member_complete_rate, "fundamental_lineage": lineage_fingerprint, "accounting_audit": accounting_audit, "hard_gates": {"rank_failures": 0, "condition_failures": 0, "condition_threshold": 1e8, "minimum_formal_member_complete_rate": 0.999}}
    manifest["build_fingerprint"] = stable_hash(manifest)
    write_yaml(manifest_path, manifest)
    return manifest
