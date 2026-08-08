from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..data.discovery import discover_data_files
from ..data.fundamentals import compute_accounting_exposures, filter_standard_fundamentals, select_ccm_links
from ..utils.hashing import files_fingerprint, stable_hash
from ..utils.io import write_json, write_yaml
from .exposures import STYLE_NAMES, preprocess_exposures

RISK_ARTIFACT_VERSION = 1


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
        covariance = stock.rolling(252, min_periods=126).cov(market_series)
        variance = market_series.where(stock.notna()).rolling(252, min_periods=126).var()
        beta = (covariance / variance.where(variance.abs() > 1e-12)).to_numpy(float)
        output["beta_252"][:, asset] = beta
        residual = stock.to_numpy(float) - beta * market
        output["resid_vol_252"][:, asset] = pd.Series(residual).rolling(252, min_periods=126).std().to_numpy(float)
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
    index = pd.read_json(panel_root / "index.json", typ="series")
    dates = pd.DatetimeIndex(index["dates"])
    permnos = np.asarray(index["permnos"], dtype=np.int64)
    source_fingerprint = files_fingerprint(files.values())
    manifest_path = panel_root / "risk_build_manifest.yaml"
    risk_path = panel_root / "risk_exposures.zarr"
    if manifest_path.exists() and risk_path.exists():
        import yaml

        with manifest_path.open(encoding="utf-8") as handle:
            prior = yaml.safe_load(handle) or {}
        if prior.get("source_fingerprint") == source_fingerprint and prior.get("artifact_version") == RISK_ARTIFACT_VERSION:
            return {**prior, "reused": True}
    daily_columns = ["PERMNO", "DlyCalDt", "DlyRet", "DlyClose", "DlyVol", "DlyCap", "SICCD"]
    daily = pd.read_parquet(files["daily"], columns=daily_columns)
    returns, close = _dense(daily, "DlyRet", dates, permnos), _dense(daily, "DlyClose", dates, permnos)
    volume, cap, crsp_sic = _dense(daily, "DlyVol", dates, permnos), _dense(daily, "DlyCap", dates, permnos), _dense(daily, "SICCD", dates, permnos)
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
        daily_diagnostics.append({"date": str(date.date()), "n_members": int(active.sum()), "condition_number": result.diagnostics["condition_number"], "disabled_styles": disabled})
    group = zarr.open_group(str(risk_path), mode="w")
    group.create_array("exposures", data=exposures, chunks=(21, min(256, len(permnos)), 22), overwrite=True)
    group.attrs["columns"] = list(preprocess_exposures(pd.DataFrame({name: [0.0] * 40 for name in STYLE_NAMES}), pd.Series([9999] * 40)).columns)
    write_json(panel_root / "risk_diagnostics.json", {"days": daily_diagnostics, "accounting": accounting_audit})
    manifest = {"artifact_version": RISK_ARTIFACT_VERSION, "source_fingerprint": source_fingerprint, "shape": list(exposures.shape), "finite_rate": float(np.isfinite(exposures).mean()), "accounting_audit": accounting_audit}
    manifest["build_fingerprint"] = stable_hash(manifest)
    write_yaml(manifest_path, manifest)
    return manifest
