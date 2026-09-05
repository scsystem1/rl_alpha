from __future__ import annotations

from typing import Any

from ..prompts import DSL_GRAMMAR
from ...dsl.operators import CONSTANTS, FEATURES, OPERATORS, WINDOWS
from ...utils.hashing import stable_hash


PROMPT_VERSION = "quantevolver_seeded_dico_rlalpha_v1"
SYSTEM_PROMPT = (
    "You are an expert quantitative factor researcher. Mutate one seed into one "
    "daily cross-sectional alpha factor. Return exactly one <expr>...</expr> block "
    "and no explanation."
)


SEED_LIBRARY: tuple[dict[str, str], ...] = (
    {
        "id": "return_sharpe_60",
        "family": "momentum",
        "expression": "Div(Mean($return,60),Std($return,60))",
    },
    {
        "id": "price_zscore_reversal_120",
        "family": "mean_reversion",
        "expression": "Mul(-1,Div(Sub($close,Mean($close,120)),Std($close,120)))",
    },
    {
        "id": "volume_return_corr_60",
        "family": "price_volume",
        "expression": "Corr($return,Log($volume),60)",
    },
    {
        "id": "short_long_momentum",
        "family": "multi_horizon",
        "expression": "Sub(Mean($return,20),Mean($return,120))",
    },
    {
        "id": "range_volatility_20",
        "family": "volatility",
        "expression": "Div(Mean(Sub($high,$low),20),Mean($close,20))",
    },
    {
        "id": "volume_surprise_20_120",
        "family": "liquidity",
        "expression": "Div(Mean($volume,20),Mean($volume,120))",
    },
    {
        "id": "cross_sectional_reversal",
        "family": "cross_sectional",
        "expression": "Mul(-1,CSRank(Mean($return,20)))",
    },
    {
        "id": "volatility_adjusted_momentum",
        "family": "risk_adjusted",
        "expression": "Div(Mean($return,20),Std($return,120))",
    },
)


REGIME_WINDOWS: tuple[dict[str, str], ...] = (
    {"name": "train_early", "start": "2010-01-01", "end": "2012-12-31"},
    {"name": "train_middle", "start": "2013-01-01", "end": "2015-12-31"},
    {"name": "train_late", "start": "2016-01-01", "end": "2018-12-31"},
    {"name": "train_full", "start": "2010-01-01", "end": "2018-12-31"},
)


def task_for_round(round_index: int, seed: int) -> dict[str, Any]:
    seed_record = SEED_LIBRARY[(int(round_index) + int(seed)) % len(SEED_LIBRARY)]
    window = REGIME_WINDOWS[(int(round_index) // len(SEED_LIBRARY) + int(seed)) % len(REGIME_WINDOWS)]
    return {
        "round": int(round_index),
        "task_id": f"{seed_record['id']}__{window['name']}__{round_index:04d}",
        "seed_id": seed_record["id"],
        "seed_expr": seed_record["expression"],
        "family": seed_record["family"],
        "time_split": window["name"],
        "start_date": window["start"],
        "end_date": window["end"],
    }


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""Seed factor: <expr>{task['seed_expr']}</expr>
Seed family: {task['family']}
Training regime: {task['time_split']} ({task['start_date']} to {task['end_date']})

Mutate the seed structurally to improve mean daily cross-sectional RankIC on the specified training regime. Change timescale or operator, add price-volume or volatility normalization, invert a contrarian signal, or combine short and long horizons. Produce a non-trivial, numerically stable, non-duplicate factor using only past/current information.

Allowed features: {', '.join(sorted(FEATURES))}
Allowed operators: {', '.join(sorted(OPERATORS))}
Allowed windows: {', '.join(map(str, WINDOWS))}
Allowed constants: {', '.join(map(str, CONSTANTS))}
Constraints: nodes<=21, depth<=6, cumulative lookback<=252.

Output only <expr>FORMULA</expr>."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_contract() -> dict[str, Any]:
    payload = {
        "version": PROMPT_VERSION,
        "system": SYSTEM_PROMPT,
        "grammar": DSL_GRAMMAR,
        "seed_library": list(SEED_LIBRARY),
        "regime_windows": list(REGIME_WINDOWS),
        "sampling_protocol": "one seeded task per optimizer update; eight completions",
    }
    return {**payload, "hash": stable_hash(payload)}

