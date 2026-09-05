from __future__ import annotations

from .r0 import R0Objective
from .r1 import R1Objective
from .r2_lcb import R2LCBObjective
from .walk_forward import DEFAULT_TIME_FOLDS, WalkForwardObjective


OOF_REWARDS = ("r1_oof", "r2_paired_oof")
CHECKPOINT_SCHEMA_VERSION = 8
REWARD_POOL_SEMANTICS = "fixed-universe-rolling-paired-oof-v8"


def objective_contract(objective):
    """Settings that must not change when resuming a frozen pool."""
    from ..utils.hashing import stable_hash

    keys = ("ridge", "hac_lag", "critical_value", "min_pool_valid_days",
            "min_pool_valid_day_rate", "min_pool_observation_rate", "time_folds", "horizon_trading_days")
    contract = {key: getattr(objective, key) for key in keys if hasattr(objective, key)}
    contract["type"] = type(objective).__name__
    if hasattr(objective, "dates"):
        contract["trading_dates_hash"] = stable_hash(objective.dates.astype(str).tolist())
    return stable_hash(contract)


def objective_for(reward, panel, reward_config=None, *, evaluation=False):
    """Single factory for the main process and frozen GRPO reward workers."""
    config = reward_config or {}
    support = {key: config.get(key, default) for key, default in (
        ("min_pool_valid_day_rate", .80), ("min_pool_observation_rate", .80),
        ("min_pool_valid_days", 252), ("ridge", .001))}
    label, mask = panel.target(panel.label), panel.target(panel.common_mask)
    if reward == "r0":
        return R0Objective(label, mask, **support)
    exposures = panel.target(panel.exposures)
    if reward == "r1" or (reward in OOF_REWARDS and evaluation):
        return R1Objective(label, mask, exposures, **support)
    lag = config.get("hac_lag") if config.get("hac_lag") is not None else 20
    critical = config.get("critical_value") if config.get("critical_value") is not None else 1.645
    if reward == "r2_lcb":
        return R2LCBObjective(label, mask, exposures, hac_lag=lag, critical_value=critical, **support)
    if reward in OOF_REWARDS:
        if not config.get("time_folds"):
            raise ValueError("OOF rewards require explicit time_folds in the reward configuration")
        return WalkForwardObjective(label, mask, exposures, dates=panel.target_dates,
            time_folds=config["time_folds"], hac_lag=lag,
            critical_value=critical if reward == "r2_paired_oof" else 0.0, **support)
    raise ValueError(f"unknown reward {reward!r}")


def prompt_objective_for(panel, reward_config=None):
    config = {**(reward_config or {}), "time_folds": DEFAULT_TIME_FOLDS}
    # Prompt evidence is canonical across reward variants, including R0.
    return objective_for("r1_oof", panel, config)
