import numpy as np
import pandas as pd
from scipy.optimize import minimize


def make_linear_constraint(tickers: list[str], constraint_tickers: list[str], limit: float, direction: str) -> dict:
    idx = np.array([tickers.index(t) for t in constraint_tickers if t in tickers], dtype=int)
    if direction not in ("<=", ">="):
        raise ValueError("direction must be '<=' or '>='")
    if direction == "<=":
        return {"type": "ineq", "fun": lambda w, ix=idx, lim=limit: lim - float(np.sum(w[ix]))}
    return {"type": "ineq", "fun": lambda w, ix=idx, lim=limit: float(np.sum(w[ix])) - lim}


def build_portfolio_constraints(tickers: list[str], config: dict) -> tuple[list[tuple[float, float]], list[dict]]:
    p = config["portfolio"]
    bounds = [(p["gld_bounds"][0], p["gld_bounds"][1]) if t == "GLD" else (0.0, p["max_single_position"]) for t in tickers]
    equities = [t for t in ("SPY", "EFA", "EEM", "XLK", "XLE") if t in tickers]
    defensive = [t for t in ("AGG", "GLD") if t in tickers]
    eq_ix = np.array([tickers.index(t) for t in equities], dtype=int)
    def_ix = np.array([tickers.index(t) for t in defensive], dtype=int)
    constraints = [
        {"type": "ineq", "fun": lambda w, ix=eq_ix, lim=p["max_equity_exposure"]: lim - float(w[ix].sum())},
        {"type": "ineq", "fun": lambda w, ix=def_ix, lim=p["min_defensive_exposure"]: float(w[ix].sum()) - lim},
    ]
    for ticker, key in (("EEM", "max_eem"), ("XLK", "max_xlk"), ("XLE", "max_xle")):
        if ticker in tickers:
            i, lim = tickers.index(ticker), p[key]
            constraints.append({"type": "ineq", "fun": lambda w, idx=i, limit=lim: float(limit - w[idx])})
    return bounds, constraints


def optimize_target_return_min_variance(mu: pd.Series, sigma: pd.DataFrame, target_return: float, tickers: list[str], bounds: list[tuple[float, float]], extra_constraints: list[dict] | None = None) -> pd.Series:
    muv = mu.reindex(tickers).to_numpy(dtype=float)
    cov = sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    if not np.isfinite(muv).all() or not np.isfinite(cov).all():
        raise ValueError("Expected returns and covariance must be finite")
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1)}, {"type": "ineq", "fun": lambda w: float(muv @ w - target_return)}] + (extra_constraints or [])
    initial = np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)
    initial /= initial.sum()
    result = minimize(lambda w: float(w @ cov @ w), initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1500, "ftol": 1e-11})
    if not result.success or abs(result.x.sum() - 1) > 1e-6 or muv @ result.x + 1e-7 < target_return:
        raise ValueError(f"Optimizer failed for target {target_return}: {result.message}")
    return pd.Series(result.x, index=tickers, name="weight")


def optimize_configured(mu: pd.Series, sigma: pd.DataFrame, target_return: float, tickers: list[str], config: dict) -> pd.Series:
    bounds, extra = build_portfolio_constraints(tickers, config)
    return optimize_target_return_min_variance(mu, sigma, target_return, tickers, bounds, extra)


def efficient_frontier(mu: pd.Series, sigma: pd.DataFrame, targets: list[float], tickers: list[str], config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, weight_rows = [], []
    for target in targets:
        try:
            weights = optimize_configured(mu, sigma, float(target), tickers, config)
            cov = sigma.reindex(index=tickers, columns=tickers).to_numpy()
            rows.append({"target_return": target, "expected_return": float(mu.reindex(tickers) @ weights), "volatility": float(np.sqrt(weights.to_numpy() @ cov @ weights.to_numpy())), "status": "feasible"})
            weight_rows.append({"target_return": target, **weights.to_dict()})
        except ValueError as exc:
            rows.append({"target_return": target, "expected_return": np.nan, "volatility": np.nan, "status": f"infeasible: {exc}"})
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)
