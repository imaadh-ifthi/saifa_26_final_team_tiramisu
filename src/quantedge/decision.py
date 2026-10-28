"""Turning the risk numbers into portfolio decisions.

Three jobs:
1. Risk budgeting. Who in the portfolio is actually taking the risk? We split the
   portfolio's VaR and ES into one piece per asset (Euler allocation, Tasche 1999): the
   pieces add up exactly to the total. Marginal ES tells you how much the portfolio ES moves
   if you add $1 to an asset, so it's the number you use for position sizing.
2. Scenarios. Tomorrow's possible returns come from filtered historical simulation: every
   past day's asset shocks (2007 onwards, GJR-filtered, de-correlated with DCC) are re-played
   with TODAY's volatilities and TODAY's correlations. That's the same engine as the
   asset_dcc risk model, so the risk split reacts to a broken stock-bond hedge straight away.
   Daniel's earlier split used the raw last 252 days, which jumps around a lot (for example,
   GLD at 85% of 90% VaR on one day).
3. Small trades. For every "sell 5% of A, buy 5% of B" that keeps the mandate limits, we
   recompute ES, the worst stress loss and the expected return. That gives the committee a
   short menu: what each trade buys in safety and what it costs in return.
"""
import itertools
import numpy as np
import pandas as pd
from .dcc import fit_dcc, correlation_path
from .garch import fit_gjr_garch, filter_volatility

EQUITY = ("SPY", "EFA", "EEM", "XLE", "XLK")
DEFENSIVE = ("AGG", "GLD")


def filtered_scenarios(asset_log: pd.DataFrame, use_dcc: bool = True) -> tuple[np.ndarray, dict]:
    """Next-day scenarios of asset SIMPLE returns (one row per past day), built from data up to the last row."""
    a = asset_log.to_numpy(dtype=float)
    n, k = a.shape
    params = [fit_gjr_garch(a[:, j]) for j in range(k)]
    sig = np.column_stack([filter_volatility(a[:, j], params[j]) for j in range(k)])   # n+1 rows
    mus = np.array([p["mu"] for p in params])
    z = (a - mus) / sig[:-1]
    info = {"sigma_today": dict(zip(asset_log.columns, sig[-1])),
            "sigma_long_run": dict(zip(asset_log.columns, a.std(axis=0)))}
    if use_dcc:
        dcc = fit_dcc(z)
        corr = correlation_path(z, dcc["a"], dcc["b"], dcc["qbar"])
        chol = np.linalg.cholesky(corr)
        eta = np.linalg.solve(chol[:-1], z[:, :, None])[:, :, 0]
        shocks = eta @ chol[-1].T
        info.update({"dcc_a": dcc["a"], "dcc_b": dcc["b"],
                     "corr_today": pd.DataFrame(corr[-1], index=asset_log.columns, columns=asset_log.columns),
                     "corr_average": pd.DataFrame(dcc["qbar"], index=asset_log.columns, columns=asset_log.columns)})
    else:
        shocks = z
    return np.expm1(mus + shocks * sig[-1]), info


def tail_risk(port_loss: np.ndarray, confidence: float) -> tuple[float, float, np.ndarray]:
    """Empirical VaR, ES and the boolean tail mask (losses at or beyond VaR)."""
    var = float(np.quantile(port_loss, confidence))
    tail = port_loss >= var
    return var, float(port_loss[tail].mean()), tail


def euler_contributions(scen: np.ndarray, weights: pd.Series, confidence: float, notional: float,
                        bandwidth: float | None = None) -> pd.DataFrame:
    """Split VaR and ES into asset pieces that add up to the totals.

    ES piece of asset i  = w_i * average loss of asset i on the tail scenarios.
    VaR piece of asset i = w_i * average loss of asset i on scenarios NEAR the VaR line
                           (Gaussian kernel weights; a single scenario would be far too noisy).
    """
    w = weights.to_numpy(dtype=float)
    asset_loss = -scen
    port_loss = asset_loss @ w
    var, es, tail = tail_risk(port_loss, confidence)
    h = bandwidth or 0.5 * port_loss.std() * len(port_loss) ** (-0.2)
    kern = np.exp(-0.5 * ((port_loss - var) / h) ** 2)
    m_var = (kern @ asset_loss) / kern.sum()
    m_es = asset_loss[tail].mean(axis=0)
    c_var, c_es = w * m_var, w * m_es
    standalone = np.array([tail_risk(asset_loss[:, j], confidence)[1] for j in range(len(w))])
    out = pd.DataFrame({"asset": weights.index, "weight": w, "position_usd": w * notional,
                        "marginal_var": m_var, "marginal_es": m_es,
                        "component_var_usd": c_var / c_var.sum() * var * notional,
                        "component_es_usd": c_es * notional,
                        "var_share": c_var / c_var.sum(), "es_share": c_es / c_es.sum(),
                        "standalone_es_usd": w * standalone * notional})
    out["es_share_per_weight"] = np.where(w > 1e-9, out.es_share / np.maximum(w, 1e-12), np.nan)
    out.attrs.update({"confidence": confidence, "portfolio_var_usd": var * notional,
                      "portfolio_es_usd": es * notional,
                      "diversification_benefit_usd": float(out.standalone_es_usd.sum() - es * notional)})
    return out


def mandate_breaches(weights: pd.Series, limits: dict) -> list[str]:
    """Return a list of broken mandate limits (empty list = allowed)."""
    w = weights.reindex(list(EQUITY) + list(DEFENSIVE)).fillna(0.0)
    out = []
    if w[list(EQUITY)].sum() > limits["max_equity_exposure"] + 1e-9:
        out.append("equity above limit")
    if w[list(DEFENSIVE)].sum() < limits["min_defensive_exposure"] - 1e-9:
        out.append("defensive below limit")
    for t, key in (("EEM", "max_eem"), ("XLK", "max_xlk"), ("XLE", "max_xle")):
        if w[t] > limits[key] + 1e-9:
            out.append(f"{t} above limit")
    lo, hi = limits["gld_bounds"]
    if not lo - 1e-9 <= w["GLD"] <= hi + 1e-9:
        out.append("GLD outside bounds")
    if w.max() > limits["max_single_position"] + 1e-9:
        out.append("single position above limit")
    if (w < -1e-9).any():
        out.append("short position")
    return out


def trade_menu(scen: np.ndarray, weights: pd.Series, limits: dict, expected_returns: pd.Series,
               stress_fn, confidence: float, notional: float, step: float = 0.05) -> pd.DataFrame:
    """Every 'sell step of A, buy step of B' that keeps the mandate, with its effect on risk and return.

    stress_fn(weights) must return the worst stress-scenario loss (a positive fraction).
    """
    asset_loss = -scen
    base_var, base_es, _ = tail_risk(asset_loss @ weights.to_numpy(), confidence)
    base_stress = stress_fn(weights)
    base_ret = float(weights @ expected_returns.reindex(weights.index))
    rows = []
    for sell, buy in itertools.permutations(weights.index, 2):
        if weights[sell] < step - 1e-9:
            continue
        w = weights.copy()
        w[sell] -= step
        w[buy] += step
        if mandate_breaches(w, limits):
            continue
        var, es, _ = tail_risk(asset_loss @ w.to_numpy(), confidence)
        stress = stress_fn(w)
        ret = float(w @ expected_returns.reindex(w.index))
        rows.append({"sell": sell, "buy": buy, "size": step,
                     "es_usd": es * notional, "es_change_usd": (es - base_es) * notional,
                     "var_change_usd": (var - base_var) * notional,
                     "worst_stress_loss_pct": stress, "worst_stress_change_pct": stress - base_stress,
                     "expected_return": ret, "expected_return_change": ret - base_ret})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("es_change_usd").reset_index(drop=True)
    out.attrs.update({"base_es_usd": base_es * notional, "base_var_usd": base_var * notional,
                      "base_worst_stress": base_stress, "base_expected_return": base_ret})
    return out


def reverse_stress(weights: pd.Series, betas: pd.Series, loss_limit: float) -> float:
    """How big an S&P 500 fall (with every asset moving beta x the S&P move) takes the portfolio to loss_limit.

    betas come from crisis-day regressions, so they already carry the 'hedges fail' pattern.
    """
    sens = float((weights * betas.reindex(weights.index).fillna(0.0)).sum())
    return float(loss_limit / sens) if sens > 0 else float("inf")
