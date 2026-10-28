"""The crisis-response candidate models, all run the same way.

Every candidate gives, for each forecast day t, a quantile function q(p) of tomorrow's
portfolio log return, built only from data up to t-1. VaR, ES and the PIT all come from
q(p) through src/quantedge/tail_grid.py, so every model is scored with the same arithmetic.

Candidate families (the `filter` field in config.yaml):
  gjr         GJR-GARCH on the portfolio, then GPD on the worst 10% of shocks (this is gjr_evt)
  gjrx_vix    same, plus yesterday's VIX in the variance equation (GARCH-X)
  gjrx_range  same, plus yesterday's high-low range variance of the portfolio
  har_range   HAR model on the range variance gives sigma, then the same GPD tail
  asset_fhs   GJR filter on each asset, rebuild 1,000 portfolio scenarios with today's asset
              volatilities, GPD on the scenario tail (keeps the old co-movement of shocks)
  asset_dcc   same as asset_fhs, but the shocks are also re-correlated with today's DCC
              correlation, so a broken stock-bond hedge shows up straight away
  ms_gjr      two-regime Markov-switching GJR with Student-t shocks (no EVT step)

Timing rule, checked in the tests: the variance for day t uses returns up to t-1, and the
VIX or range input for day t is the value from day t-1 (we shift every series by one day).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.layers345_upgrade.models import load_portfolio
from src.quantedge.dcc import fit_dcc, correlation_path
from src.quantedge.evt_extra import fit_gpd
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from src.quantedge.garch_x import fit_gjr_garch_x, filter_volatility_x
from src.quantedge.ms_garch import fit_ms_gjr, predictive, mixture_quantiles
from src.quantedge.range_vol import asset_range_variances, portfolio_range_variance, fit_har, har_variance_path
from src.quantedge.tail_grid import (probability_grid, risk_from_quantiles, pit_from_quantiles,
                                     gpd_tail_quantiles)

MIN_WEIGHT = 1e-6  # assets below this weight (EFA, EEM, XLE at about 1e-16) are left out


def load_inputs(root: Path, base_cfg: dict) -> dict:
    """Portfolio returns, asset returns and the lagged VIX / range inputs on one date index."""
    r, run, complete, current = load_portfolio(root, base_cfg)
    weights = pd.Series(run["weights"], dtype=float)
    held = weights[weights > MIN_WEIGHT]
    held = held / held.sum()
    simple = complete.reindex(r.index)

    vix = pd.read_csv(root / "data/raw/vixcls_fred.csv", parse_dates=["date"], index_col="date").VIXCLS
    yahoo = pd.read_csv(root / "data/raw/vix_yahoo.csv", parse_dates=["date"], index_col="date").vix_close
    vix = vix.combine_first(yahoo).reindex(r.index)
    if vix.isna().any():
        raise ValueError("VIX is missing on some portfolio dates")
    ohlc = pd.read_csv(root / "data/raw/ohlc.csv", parse_dates=["date"])
    asset_var = asset_range_variances(ohlc, list(weights.index)).reindex(r.index)
    if asset_var[held.index].isna().any().any():
        raise ValueError("high/low data is missing on some portfolio dates")
    range_var = portfolio_range_variance(asset_var, simple, held)

    # shift(1): the input for day t is what we knew at the close of day t-1.
    # The very first day has no previous value; it only sits at the start of the first window.
    implied_var = ((vix / 100) ** 2 / 252)
    return {"r": r, "values": r.to_numpy(), "dates": r.index, "weights": held,
            "asset_log": np.log1p(simple[held.index]).to_numpy(),
            "x_vix": implied_var.shift(1).bfill().to_numpy(),
            "x_range": range_var.shift(1).bfill().to_numpy(),
            "range_var": range_var.to_numpy(), "vix": vix, "run": run, "complete": complete,
            "current": current}


def _evt_shock_quantiles(z, grid, threshold_q):
    losses = -np.asarray(z, dtype=float)
    u = float(np.quantile(losses, threshold_q))
    excess = losses[losses > u] - u
    shape, scale = fit_gpd(excess)
    q = gpd_tail_quantiles(z, grid, u, shape, scale, len(excess) / len(z))
    return q, shape, scale


def _rows_for_day(i, q, values, dates, grid, confidences, method, sigma, fit_end):
    pit = pit_from_quantiles(values[i], q, grid)
    return [{"date": dates[i], "method": method, "confidence": c, "estimation_end": dates[i - 1],
             "fit_end": fit_end, **risk_from_quantiles(q, grid, c), "pit": pit, "sigma": sigma,
             "actual_log_return": values[i], "evaluation_index": i} for c in confidences]


def _univariate(spec, data, first, last, grid, confidences):
    values, dates = data["values"], data["dates"]
    window, step, kind = int(spec["window"]), int(spec["refit_every"]), spec["filter"]
    x_all = {"gjr": None, "gjrx_vix": data["x_vix"], "gjrx_range": data["x_range"]}.get(kind)
    rows, fits, previous = [], [], None
    for start in range(first, last + 1, step):
        stop = min(start + step, last + 1)
        lo = start - window
        if kind == "har_range":
            rv = data["range_var"]
            fit = fit_har(rv[lo:start])
            path = har_variance_path(rv[lo:start], fit)          # forecasts for days lo+22 .. start
            z = values[lo + 22:start] / np.sqrt(path[:-1])
            block_var = har_variance_path(rv[lo:stop - 1], fit)[-(stop - start):]
            sigmas, mu = np.sqrt(block_var), 0.0
            info = {"har_b0": fit["beta"][0], "har_day": fit["beta"][1], "har_week": fit["beta"][2],
                    "har_month": fit["beta"][3]}
        else:
            x = None if x_all is None else x_all[lo:start]
            params = fit_gjr_garch_x(values[lo:start], x, previous)
            previous = params
            z, mu = params["residuals"], params["mu"]
            xb = None if x_all is None else x_all[lo:stop - 1]
            x_next = 0.0 if x_all is None else float(x_all[stop - 1])
            sigmas = filter_volatility_x(values[lo:stop - 1], xb, params, x_next)[window:]
            info = {k: params[k] for k in ("mu", "omega", "alpha", "gamma", "beta", "delta", "nu",
                                           "persistence", "loglik", "converged")}
        qz, shape, scale = _evt_shock_quantiles(z, grid, spec.get("threshold_quantile", 0.90))
        fits.append({"method": spec["name"], "fit_date": dates[start], "estimation_end": dates[start - 1],
                     **info, "evt_shape": shape, "evt_scale": scale})
        for k, i in enumerate(range(start, stop)):
            q = mu + float(sigmas[k]) * qz
            rows += _rows_for_day(i, q, values, dates, grid, confidences, spec["name"], float(sigmas[k]),
                                  dates[start - 1])
    return rows, fits


def _asset_level(spec, data, first, last, grid, confidences):
    """Asset-level filtered historical simulation with a GPD tail, optionally with DCC."""
    values, dates, a = data["values"], data["dates"], data["asset_log"]
    w = data["weights"].to_numpy()
    window, step, use_dcc = int(spec["window"]), int(spec["refit_every"]), spec["filter"] == "asset_dcc"
    k_assets = a.shape[1]
    rows, fits, previous = [], [], [None] * k_assets
    for start in range(first, last + 1, step):
        stop = min(start + step, last + 1)
        lo = start - window
        params = [fit_gjr_garch(a[lo:start, j], previous[j]) for j in range(k_assets)]
        previous = params
        # sigma and shocks for the window plus the block (block shocks use frozen parameters)
        sig = np.column_stack([filter_volatility(a[lo:stop - 1, j], params[j]) for j in range(k_assets)])
        mus = np.array([p["mu"] for p in params])
        z_all = (a[lo:stop - 1] - mus) / sig[:-1]
        z_win = z_all[:window]
        info = {"method": spec["name"], "fit_date": dates[start], "estimation_end": dates[start - 1],
                **{f"persistence_{j}": p["persistence"] for j, p in enumerate(params)}}
        if use_dcc:
            dcc = fit_dcc(z_win)
            corr = correlation_path(z_all, dcc["a"], dcc["b"], dcc["qbar"])   # R for days lo .. stop-1
            chol = np.linalg.cholesky(corr)
            # strip each past day's own correlation: eta_s = L_s^-1 z_s are uncorrelated shocks
            eta = np.linalg.solve(chol[:window], z_win[:, :, None])[:, :, 0]
            info.update({"dcc_a": dcc["a"], "dcc_b": dcc["b"]})
        for k, i in enumerate(range(start, stop)):
            s_today = sig[window + k]
            shocks = eta @ chol[window + k].T if use_dcc else z_win
            scen = np.log1p(np.expm1(mus + shocks * s_today) @ w)
            losses = -scen
            u = float(np.quantile(losses, spec.get("threshold_quantile", 0.90)))
            excess = losses[losses > u] - u
            shape, scale = fit_gpd(excess)
            q = gpd_tail_quantiles(scen, grid, u, shape, scale, len(excess) / len(scen))
            rows += _rows_for_day(i, q, values, dates, grid, confidences, spec["name"],
                                  float(np.std(scen)), dates[start - 1])
        info.update({"evt_shape_last": shape})
        fits.append(info)
    return rows, fits


def _markov_switching(spec, data, first, last, grid, confidences):
    values, dates = data["values"], data["dates"]
    window, step = int(spec["window"]), int(spec["refit_every"])
    rows, fits, previous = [], [], None
    for start in range(first, last + 1, step):
        stop = min(start + step, last + 1)
        lo = start - window
        params = fit_ms_gjr(values[lo:start], previous)
        previous = params
        prob, h = predictive(values[lo:stop - 1], params)
        fits.append({"method": spec["name"], "fit_date": dates[start], "estimation_end": dates[start - 1],
                     **{k: v for k, v in params.items() if k != "start_var"}})
        for k, i in enumerate(range(start, stop)):
            p, v = prob[window + k], h[window + k]
            q = mixture_quantiles(p, v, params["mu"], params["nu"], grid)
            rows += _rows_for_day(i, q, values, dates, grid, confidences, spec["name"],
                                  float(np.sqrt(p @ v)), dates[start - 1])
    return rows, fits


def run_candidate(spec: dict, data: dict, start_date, end_date, confidences, grid=None):
    """Forecast every day from start_date to end_date (inclusive). Returns (forecasts, fit log)."""
    grid = probability_grid() if grid is None else grid
    dates = data["dates"]
    first = int(np.searchsorted(dates, pd.Timestamp(start_date)))
    last = int(np.searchsorted(dates, pd.Timestamp(end_date), side="right")) - 1
    if first < int(spec["window"]):
        raise ValueError(f"{spec['name']}: not enough history before {start_date}")
    kind = spec["filter"]
    if kind in ("gjr", "gjrx_vix", "gjrx_range", "har_range"):
        rows, fits = _univariate(spec, data, first, last, grid, confidences)
    elif kind in ("asset_fhs", "asset_dcc"):
        rows, fits = _asset_level(spec, data, first, last, grid, confidences)
    elif kind == "ms_gjr":
        rows, fits = _markov_switching(spec, data, first, last, grid, confidences)
    else:
        raise ValueError(f"unknown filter {kind}")
    return pd.DataFrame(rows), pd.DataFrame(fits)
