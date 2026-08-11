from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dollar_neutral_target(score: np.ndarray, eligible: np.ndarray, quantile: float = 0.2) -> np.ndarray:
    score = np.asarray(score, float)
    eligible = np.asarray(eligible, bool) & np.isfinite(score)
    indices = np.flatnonzero(eligible)
    target = np.zeros_like(score)
    if len(indices) < 10:
        return target
    count = max(1, int(np.floor(quantile * len(indices))))
    ordered = indices[np.argsort(score[indices], kind="stable")]
    short, long = ordered[:count], ordered[-count:]
    target[long] = 0.5 / len(long)
    target[short] = -0.5 / len(short)
    return target


def project_fully_neutral(
    target: np.ndarray,
    score: np.ndarray,
    exposures: np.ndarray,
    eligible: np.ndarray,
    max_weight: float = 0.02,
    *,
    net_tolerance: float = 1e-8,
    exposure_tolerance: float = 1e-6,
    gross_tolerance: float = 1e-6,
    weight_tolerance: float = 1e-6,
    previous_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
) -> tuple[np.ndarray | None, dict[str, object]]:
    import cvxpy as cp

    target, score, exposures = np.asarray(target, float), np.asarray(score, float), np.asarray(exposures, float)
    valid = np.asarray(eligible, bool) & np.isfinite(score) & np.isfinite(exposures).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < int(np.ceil(1 / max_weight)):
        return None, {"status": "insufficient_eligible"}
    local_score, local_target = score[indices], target[indices]
    x = exposures[indices, 1:] if exposures.shape[1] == 22 else exposures[indices]
    median = np.median(local_score)
    long_ok, short_ok = local_score >= median, local_score < median
    plus, minus = cp.Variable(len(indices), nonneg=True), cp.Variable(len(indices), nonneg=True)
    weight = plus - minus
    constraints = [cp.sum(plus) == 0.5, cp.sum(minus) == 0.5, x.T @ weight == 0, plus <= max_weight, minus <= max_weight]
    constraints += [plus[~long_ok] == 0, minus[~short_ok] == 0]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weight - local_target) + 1e-4 * cp.sum_squares(weight)), constraints)
    solver_used = None
    for solver in ("OSQP", "CLARABEL"):
        try:
            options = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 100000} if solver == "OSQP" else {}
            problem.solve(solver=solver, warm_start=True, verbose=False, **options)
        except cp.error.SolverError:
            continue
        solver_used = solver
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weight.value is None:
        return None, {"status": str(problem.status), "solver": solver_used}
    result = np.zeros_like(target)
    result[indices] = np.asarray(weight.value).ravel()
    residual = x.T @ result[indices]
    net = float(result.sum())
    gross = float(np.abs(result).sum())
    largest_weight = float(np.abs(result).max(initial=0))
    largest_exposure = float(np.abs(residual).max(initial=0))
    nontradable_weight = float(np.abs(result[~valid]).max(initial=0))
    turnover = None if previous_weights is None else float(np.abs(result - np.asarray(previous_weights, dtype=float)).sum())
    violations = []
    if abs(net) > net_tolerance:
        violations.append("net")
    if largest_exposure > exposure_tolerance:
        violations.append("risk_exposure")
    if abs(gross - 1.0) > gross_tolerance:
        violations.append("gross")
    if largest_weight > max_weight + weight_tolerance:
        violations.append("max_weight")
    if nontradable_weight > weight_tolerance:
        violations.append("tradability")
    if turnover_limit is not None and (turnover is None or not np.isfinite(turnover) or turnover > turnover_limit + weight_tolerance):
        violations.append("turnover")
    solver_stats = problem.solver_stats
    audit = {
        "status": str(problem.status),
        "solver": solver_used,
        "solver_iterations": getattr(solver_stats, "num_iters", None),
        "solve_time": getattr(solver_stats, "solve_time", None),
        "net": net,
        "gross": gross,
        "max_weight": largest_weight,
        "max_risk_exposure": largest_exposure,
        "max_nontradable_weight": nontradable_weight,
        "turnover": turnover,
        "tolerances": {
            "net": net_tolerance,
            "exposure": exposure_tolerance,
            "gross": gross_tolerance,
            "weight": weight_tolerance,
            "turnover_limit": turnover_limit,
        },
        "constraint_violations": violations,
        "accepted": not violations,
    }
    if violations:
        return None, audit
    return result, audit


@dataclass(frozen=True)
class PortfolioResult:
    weights: np.ndarray
    gross_returns: np.ndarray
    turnover: np.ndarray
    missing_held_returns: np.ndarray
    infeasible: np.ndarray
    audits: list[dict[str, object]]


class PortfolioBacktester:
    def __init__(self, rebalance_days: int = 5, holding_days: int = 20, execution_delay: int = 1, pnl_delay: int = 1):
        if holding_days % rebalance_days:
            raise ValueError("holding_days must be divisible by rebalance_days")
        self.rebalance_days = rebalance_days
        self.sleeves = holding_days // rebalance_days
        self.activation_delay = execution_delay + pnl_delay

    def run(self, scores: np.ndarray, returns: np.ndarray, eligible: np.ndarray, exposures: np.ndarray | None = None, fully_neutral: bool = False, max_weight: float = 0.02, neutral_tolerances: dict[str, float] | None = None) -> PortfolioResult:
        scores, returns, eligible = np.asarray(scores, float), np.asarray(returns, float), np.asarray(eligible, bool)
        if scores.shape != returns.shape or scores.shape != eligible.shape:
            raise ValueError("portfolio panel shapes differ")
        days, assets = scores.shape
        sleeve_weights = np.zeros((self.sleeves, assets))
        pending: dict[int, tuple[int, np.ndarray, dict[str, object]]] = {}
        weights = np.zeros((days, assets))
        gross_returns, turnover = np.zeros(days), np.zeros(days)
        missing = np.zeros(days, dtype=int)
        infeasible = np.zeros(days, dtype=bool)
        audits: list[dict[str, object]] = []
        for day in range(days):
            if day in pending:
                sleeve, target, audit = pending.pop(day)
                sleeve_weights[sleeve] = target
                audits.append({"day": day, **audit})
            current = sleeve_weights.mean(axis=0)
            weights[day] = current
            finite_return = np.isfinite(returns[day])
            held_missing = (np.abs(current) > 0) & ~finite_return
            missing[day] = int(held_missing.sum())
            gross_returns[day] = np.nan if held_missing.any() else float(np.dot(current[finite_return], returns[day, finite_return]))
            if day % self.rebalance_days == 0 and day + self.activation_delay < days:
                sleeve = (day // self.rebalance_days) % self.sleeves
                base = dollar_neutral_target(scores[day], eligible[day])
                audit: dict[str, object] = {"status": "dollar_neutral", "signal_day": day}
                target = base
                if fully_neutral:
                    if exposures is None:
                        raise ValueError("exposures required")
                    projected, audit = project_fully_neutral(base, scores[day], exposures[day], eligible[day], max_weight, **(neutral_tolerances or {}))
                    audit["signal_day"] = day
                    if projected is None:
                        infeasible[day] = True
                        target = sleeve_weights[sleeve].copy()
                    else:
                        target = projected
                execution_day = day + self.activation_delay - 1
                turnover[execution_day] += float(np.abs(target - sleeve_weights[sleeve]).sum()) / self.sleeves
                pending[day + self.activation_delay] = (sleeve, target, audit)
        return PortfolioResult(weights, gross_returns, turnover, missing, infeasible, audits)


def portfolio_metrics(result: PortfolioResult, cost_bps: float) -> dict[str, object]:
    net_returns = result.gross_returns - cost_bps / 10000.0 * result.turnover
    invalid_return_path = bool(result.missing_held_returns.sum() > 0 or (~np.isfinite(net_returns)).any())
    finite = net_returns[np.isfinite(net_returns)]
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + finite)]) if len(finite) else np.array([1.0])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1 if len(wealth) else np.array([])
    mean, std = (float(finite.mean()), float(finite.std(ddof=1))) if len(finite) > 1 else (0.0, 0.0)
    annual_return = mean * 252
    annual_volatility = std * np.sqrt(252)
    sharpe = mean / std * np.sqrt(252) if std > 0 else float("nan")
    maximum_drawdown = float(drawdown.min(initial=0))
    if invalid_return_path:
        annual_return = annual_volatility = sharpe = maximum_drawdown = float("nan")
    return {
        "cost_bps": cost_bps,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown,
        "average_turnover": float(result.turnover.mean()),
        "average_gross": float(np.abs(result.weights).sum(axis=1).mean()),
        "average_net": float(result.weights.sum(axis=1).mean()),
        "infeasible_days": int(result.infeasible.sum()),
        "missing_held_returns": int(result.missing_held_returns.sum()),
        "invalid_return_path": invalid_return_path,
    }
