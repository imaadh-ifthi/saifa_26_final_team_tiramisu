"""New risk models for the Layer 3-5 upgrade, plus PIT values for the existing baselines.

New models (all one-day-ahead, using only data before the forecast day):
  gjr_t        GJR-GARCH volatility x Student-t shocks
  gjr_fhs      GJR-GARCH volatility x past standardised shocks (filtered historical simulation,
               Barone-Adesi, Giannopoulos and Vosper 1999)
  gjr_evt      GJR-GARCH volatility x GPD tail on shocks above their 90% quantile
               (conditional EVT, McNeil and Frey 2000)
  gjr_evt_eqd  same, but the threshold is chosen each month by the EQD rule
  combo        average of the gjr_t, gjr_fhs and gjr_evt quantile functions (Taylor 2020)

Each model gives a quantile function q(p) on a fixed probability grid; VaR, ES and the PIT
all come from that (see src/quantedge/tail_grid.py).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from src.quantedge.tail_grid import (probability_grid, risk_from_quantiles, pit_from_quantiles,
                                     empirical_quantiles, gpd_tail_quantiles)
from src.quantedge.evt_extra import fit_gpd, eqd_threshold


def load_portfolio(root: Path, base_config: dict):
    """Frozen 2007-2017 allocation log returns (same series as the baseline OOS run)."""
    prices = pd.read_csv(root / "outputs/tables/prices.csv", index_col=0, parse_dates=True,
                         float_precision="round_trip").sort_index()
    simple = prices.pct_change(fill_method=None).iloc[1:]
    run = json.loads((root / "outputs/models/analytical_run.json").read_text(encoding="utf-8"))
    weights = pd.Series(run["weights"], dtype=float)
    tickers = base_config["data"]["tickers"]
    complete = simple[tickers].dropna()
    history = complete.loc[pd.Timestamp(base_config["validation"]["train_start"]):pd.Timestamp(run["test_end"])]
    log_returns = np.log1p(history[weights.index] @ weights)
    current = pd.Series(run["current_weights"], dtype=float)
    return log_returns, run, complete, current


def _shock_quantiles(model: str, z: np.ndarray, nu: float, grid: np.ndarray, evt_q: float | None):
    """Quantile function of the standardised shock z for one model (unit-free)."""
    if model == "gjr_t":
        return student_t.ppf(grid, nu) * np.sqrt((nu - 2) / nu)
    if model == "gjr_fhs":
        return empirical_quantiles(z, grid)
    losses = -z
    u = float(np.quantile(losses, evt_q))
    excess = losses[losses > u] - u
    shape, scale = fit_gpd(excess)
    q = gpd_tail_quantiles(z, grid, u, shape, scale, len(excess) / len(z))
    return q, {"threshold_quantile": evt_q, "shape": shape, "scale": scale}


def build_garch_family(r: pd.Series, test_start: pd.Timestamp, cfg: dict, confidences: list[float],
                       grid: np.ndarray | None = None):
    """Run the five new models over the evaluation period. Returns (forecasts, fit log)."""
    grid = probability_grid() if grid is None else grid
    values = r.to_numpy()
    window, step = int(cfg["garch"]["window"]), int(cfg["garch"]["refit_every"])
    first = int(np.searchsorted(r.index, test_start))
    if first < window:
        raise ValueError("not enough history before the test start for the GARCH window")
    ecfg = cfg["evt"]
    members = cfg["combination"]["members"]
    rows, fits = [], []
    previous = None
    for start in range(first, len(values), step):
        sample = values[start - window:start]
        params = fit_gjr_garch(sample, previous)
        previous = params
        z = params["residuals"]
        eqd = eqd_threshold(-z, ecfg["eqd_candidates"], ecfg["eqd_bootstrap"], ecfg["eqd_min_excesses"],
                            seed=start)
        shock_q = {"gjr_t": _shock_quantiles("gjr_t", z, params["nu"], grid, None),
                   "gjr_fhs": _shock_quantiles("gjr_fhs", z, params["nu"], grid, None)}
        shock_q["gjr_evt"], evt_info = _shock_quantiles("gjr_evt", z, params["nu"], grid,
                                                        ecfg["fixed_threshold_quantile"])
        shock_q["gjr_evt_eqd"], eqd_info = _shock_quantiles("gjr_evt", z, params["nu"], grid,
                                                            eqd["best"]["threshold_quantile"])
        fits.append({"fit_date": r.index[start], "estimation_end": r.index[start - 1],
                     **{k: params[k] for k in ("mu", "omega", "alpha", "gamma", "beta", "nu",
                                               "persistence", "loglik", "converged")},
                     "evt_shape": evt_info["shape"], "evt_scale": evt_info["scale"],
                     "eqd_threshold_quantile": eqd_info["threshold_quantile"], "eqd_shape": eqd_info["shape"]})
        stop = min(start + step, len(values))
        # sigma for each day in this block uses returns before that day only
        sigmas = filter_volatility(values[start - window:stop - 1], params)[window:]
        for k, i in enumerate(range(start, stop)):
            sigma = float(sigmas[k])
            quantiles = {m: params["mu"] + sigma * q for m, q in shock_q.items()}
            quantiles["combo"] = np.mean([quantiles[m] for m in members], axis=0)
            for method, q in quantiles.items():
                pit = pit_from_quantiles(values[i], q, grid)
                for c in confidences:
                    rows.append({"date": r.index[i], "method": method, "confidence": c,
                                 "estimation_end": r.index[i - 1], "fit_end": r.index[start - 1],
                                 **risk_from_quantiles(q, grid, c), "pit": pit, "sigma": sigma,
                                 "actual_log_return": values[i], "evaluation_index": i})
    return pd.DataFrame(rows), pd.DataFrame(fits)


def baseline_pits(r: pd.Series, baseline: pd.DataFrame, base_config: dict, grid: np.ndarray | None = None):
    """PIT values for Daniel's four main-pipeline models, rebuilt with his exact settings.

    Also returns the reconstructed VaR so we can check the rebuild matches his saved numbers.
    """
    grid = probability_grid() if grid is None else grid
    values = r.to_numpy()
    vcfg = base_config["var"]
    hw, mw, sims = int(vcfg["historical_window"]), int(vcfg["monte_carlo_window"]), int(vcfg["monte_carlo_simulations"])
    ewma = r.ewm(alpha=1 - vcfg["ewma_lambda"], min_periods=vcfg.get("ewma_min_periods", 60), adjust=False).std().shift(1)
    evt_window = int(base_config["evt"]["window"])
    evt_rows = baseline[(baseline.method == "evt_gpd")].drop_duplicates("date").set_index("date")
    out = []
    cache = {}
    for date, i in baseline.drop_duplicates("date").set_index("date").evaluation_index.items():
        i = int(i)
        y = values[i]
        sigma = float(ewma.iloc[i])
        hist_q = empirical_quantiles(values[i - hw:i], grid)
        rng = np.random.default_rng(base_config["project"]["seed"] + i)
        boot_q = empirical_quantiles(rng.choice(values[i - mw:i], size=sims, replace=True), grid)
        fit = evt_rows.loc[date]
        key = fit.estimation_end
        if key not in cache:
            end = int(r.index.get_loc(pd.Timestamp(key)))
            window = values[end + 1 - evt_window:end + 1]
            losses = -window
            u = float(np.quantile(losses, fit.threshold_quantile))
            p_u = float((losses > u).mean())
            cache[key] = gpd_tail_quantiles(window, grid, u, fit["shape"], fit.scale, p_u)
        evt_q = cache[key]
        for method, pit, q in (("historical", pit_from_quantiles(y, hist_q, grid), hist_q),
                               ("monte_carlo_bootstrap", pit_from_quantiles(y, boot_q, grid), boot_q),
                               ("parametric_ewma", float(norm.cdf(y / sigma)), sigma * norm.ppf(grid)),
                               ("evt_gpd", pit_from_quantiles(y, evt_q, grid), evt_q)):
            out.append({"date": date, "method": method, "pit": pit, "sigma": sigma,
                        "check_var99": -float(np.interp(0.01, grid, q))})
    return pd.DataFrame(out)
