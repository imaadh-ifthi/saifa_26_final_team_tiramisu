"""Causal EWMA-filtered and GJR-GARCH-filtered historical, bootstrap, Student-t and POT/GPD models."""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t
from src.quantedge.evt import fit_thresholds, select_threshold, gpd_risk
from src.quantedge.tail import empirical_risk
from .candidate_models.gjr_filtered_historical import GJRGARCHFilteredHistorical
from .candidate_models.gjr_filtered_bootstrap import GJRGARCHFilteredBootstrap


def causal_volatility(returns: pd.Series, decay: float = .94, min_periods: int = 60,
                       floor: float = 1e-8) -> pd.Series:
    """Same bias-corrected EWMA std as the baseline, shifted before standardizing."""
    if (not 0 < decay < 1 or min_periods < 2 or not np.isfinite(floor) or floor <= 0
            or returns.index.has_duplicates or not returns.index.is_monotonic_increasing
            or not np.isfinite(returns.to_numpy()).all()):
        raise ValueError("Invalid returns or volatility filter settings")
    return returns.ewm(alpha=1-decay, min_periods=min_periods, adjust=False).std().shift(1).clip(lower=floor)


def fit_student_df(residuals: np.ndarray, bounds=(2.1, 100.0)) -> tuple[float, str]:
    """MLE of df with location=0 and variance=1; scale is not standard deviation."""
    residuals = np.asarray(residuals, dtype=float)
    lo, hi = bounds
    if (len(residuals) < 20 or not np.isfinite(residuals).all()
            or np.std(residuals) <= 0 or not np.isfinite([lo, hi]).all() or not 2 < lo < hi):
        raise ValueError("Student-t needs varying finite residuals and df bounds above 2")

    def nll(df):
        scale = np.sqrt((df-2)/df)
        return -float((t.logpdf(residuals/scale, df)-np.log(scale)).sum())

    fit = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError("Student-t degrees-of-freedom fit failed")
    status = "warning_df_at_bound" if min(fit.x-lo, hi-fit.x) < .01 else "ok"
    return float(fit.x), status


@lru_cache(maxsize=4096)
def _student_tail(df: float, confidence: float):
    # Conditional quantile quadrature evaluates the bounded simple-loss expectation.
    nodes, weights = _quadrature()
    probabilities = (nodes+1)/2
    scale = np.sqrt((df-2)/df)
    return scale*t.isf((1-confidence)*probabilities, df), weights/2


@lru_cache(maxsize=1)
def _quadrature():
    return np.polynomial.legendre.leggauss(512)


def student_risk(volatility: float, df: float, confidence: float) -> dict:
    if not np.isfinite([volatility, df, confidence]).all() or volatility < 0 or df <= 2 or not .5 < confidence < 1:
        raise ValueError("Invalid Student-t volatility, df or confidence")
    scale = np.sqrt((df-2)/df)
    quantile = float(t.ppf(confidence, df))
    var_log = volatility*scale*quantile
    es_log = volatility*scale*(df+quantile**2)/(df-1)*t.pdf(quantile, df)/(1-confidence)
    tail_quantiles, weights = _student_tail(float(df), float(confidence))
    es_simple = float(weights @ -np.expm1(-volatility*tail_quantiles))
    return {"var_log": var_log, "es_log": float(es_log),
            "var_simple_pct": float(-np.expm1(-var_log)), "es_simple_pct": es_simple}


def filtered_gpd_risk(fit: pd.Series, volatility: float, confidence: float) -> dict:
    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("Filtered GPD requires positive finite forecast volatility")
    return gpd_risk(fit.threshold_log*volatility, fit["shape"], fit.scale*volatility,
                    fit.exceedance_probability, confidence)


def empirical_variant(returns: np.ndarray, confidence: float, quantile_method: str = "linear") -> dict:
    risk = empirical_risk(returns, confidence)
    # Define VaR on losses directly. For discontinuous empirical CDFs, negating
    # a lower return quantile is not always the upper loss quantile at a boundary.
    var_log = max(0., float(np.quantile(-np.asarray(returns), confidence, method=quantile_method)))
    risk.update(var_log=var_log, var_simple_pct=float(-np.expm1(-var_log)))
    return risk


def build_improved_forecasts(returns: pd.Series, base_config: dict, experiment: dict,
                             start_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = experiment["filter"]
    sigma = causal_volatility(returns, f["ewma_lambda"], f["min_periods"], f["volatility_floor"])
    standardized = returns/sigma
    hwindow = int(experiment["historical_window"])
    tcfg, ecfg = experiment["student_t"], experiment["evt"]
    twindow, ewindow = int(tcfg["window"]), int(ecfg["window"])
    n_sims = int(experiment["monte_carlo_simulations"])
    raw_windows = experiment.get("sensitivity", {}).get("raw_historical_windows", [])
    quantile_methods = experiment.get("sensitivity", {}).get("empirical_quantile_methods", [])
    base_window = int(base_config["var"]["historical_window"])
    confidences = base_config["var"]["confidence_levels"]

    declared_candidates = set(experiment.get("candidates", []))
    supported_candidates = {"gjr_filtered_historical", "gjr_filtered_bootstrap"}
    unknown_candidates = declared_candidates - supported_candidates
    if unknown_candidates:
        raise ValueError(f"Unknown experiment candidates: {sorted(unknown_candidates)}")

    gjr_cfg = experiment.get("gjr_garch")
    include_gjr_hist = "gjr_filtered_historical" in declared_candidates
    include_gjr_boot = "gjr_filtered_bootstrap" in declared_candidates
    include_gjr = include_gjr_hist or include_gjr_boot
    if include_gjr and gjr_cfg is None:
        raise ValueError("Declare gjr_garch settings when a GJR-GARCH candidate is enabled")

    gjr_hist_model = None
    gjr_boot_model = None
    gjr_window = 0
    if include_gjr:
        gjr_window = int(gjr_cfg.get("window", 1000))
        gjr_refit = int(gjr_cfg.get("refit_every", 21))
        if gjr_refit < 1 or gjr_window < 250:
            raise ValueError("Invalid GJR-GARCH window or refit interval")
        if include_gjr_hist:
            gjr_hist_model = GJRGARCHFilteredHistorical(window=gjr_window, refit_every=gjr_refit)
        if include_gjr_boot:
            gjr_boot_model = GJRGARCHFilteredBootstrap(
                window=gjr_window,
                refit_every=gjr_refit,
                n_simulations=n_sims,
                seed=int(base_config["project"]["seed"]),
            )

    if (min(hwindow, twindow, ewindow) < 20 or n_sims < 100
            or int(tcfg["refit_every"]) < 1 or int(ecfg["refit_every"]) < 1
            or any(w < 20 for w in raw_windows) or not confidences
            or any(not .5 < c < 1 for c in confidences)):
        raise ValueError("Invalid experiment windows, simulations, refit intervals or confidences")

    required_warmup = [f["min_periods"]+max(hwindow, twindow, ewindow), base_window, *raw_windows]
    if include_gjr:
        required_warmup.append(gjr_window)
    first = max(required_warmup)

    rows, diagnostics = [], []
    next_t, next_evt = 0, 0
    df, chosen = None, None
    t_end, evt_end = None, None
    t_status, evt_status = "not_fitted", "not_fitted"
    for i in range(first, len(returns)):
        date, vol = returns.index[i], float(sigma.iloc[i])
        if start_date is not None and date < pd.Timestamp(start_date):
            continue
        if i >= next_t:
            sample = standardized.iloc[i-twindow:i].to_numpy()
            df, t_status = fit_student_df(sample, tuple(tcfg["df_bounds"]))
            t_end = returns.index[i-1]
            diagnostics.append({"model": "ewma_student_t", "fit_date": date,
                                "estimation_end": t_end, "df": df, "status": t_status,
                                "observations": twindow, "units": "unit_variance_residual"})
            next_t = i+int(tcfg["refit_every"])
        if i >= next_evt:
            sample = standardized.iloc[i-ewindow:i]
            report = fit_thresholds(sample, ecfg["threshold_quantiles"], ecfg["min_exceedances"],
                                    ecfg["shape_tolerance"], ecfg["scale_tolerance"])
            chosen, evt_status = select_threshold(report, min(confidences))
            if chosen is None:
                raise ValueError(f"No eligible filtered EVT fit on {date}; no fallback to another model")
            evt_end = returns.index[i-1]
            for index, row in report.iterrows():
                entry = row.to_dict()
                entry["threshold_standardized"] = entry.pop("threshold_log")
                diagnostics.append({**entry, "model": "filtered_evt", "fit_date": date,
                                    "estimation_end": evt_end, "selection_status": evt_status,
                                    "selected": index == chosen.name, "units": "unit_variance_residual"})
            next_evt = i+int(ecfg["refit_every"])
        trailing = standardized.iloc[i-hwindow:i].to_numpy()*vol
        rng = np.random.default_rng(base_config["project"]["seed"]+i)
        simulated = rng.choice(trailing, size=n_sims, replace=True)

        # GJR-GARCH candidates if included
        gjr_hist_estimates = None
        gjr_boot_estimates = None
        if gjr_hist_model is not None:
            gjr_hist_estimates, gjr_diag = gjr_hist_model.fit_and_forecast(returns, i, confidences)
            if gjr_diag is not None:
                diagnostics.append({**gjr_diag, "candidate": "gjr_filtered_historical"})
        if gjr_boot_model is not None:
            gjr_boot_estimates, gjr_diag = gjr_boot_model.fit_and_forecast(
                returns, i, confidences, seed_offset=i)
            if gjr_diag is not None:
                diagnostics.append({**gjr_diag, "candidate": "gjr_filtered_bootstrap"})

        for c in confidences:
            estimates = {
                "filtered_historical": (empirical_risk(trailing, c), "ok", returns.index[i-1], vol),
                "filtered_bootstrap": (empirical_risk(simulated, c), "ok", returns.index[i-1], vol),
                "ewma_student_t": (student_risk(vol, df, c), t_status, t_end, vol),
                "filtered_evt": (filtered_gpd_risk(chosen, vol, c), evt_status, evt_end, vol),
            }
            if gjr_hist_estimates is not None:
                estimates["gjr_filtered_historical"] = (
                    {k: gjr_hist_estimates[c][k] for k in ("var_log", "es_log", "var_simple_pct", "es_simple_pct")},
                    gjr_hist_estimates[c]["status"],
                    gjr_hist_estimates[c]["tail_fit_end"],
                    gjr_hist_estimates[c]["forecast_volatility"],
                )
            if gjr_boot_estimates is not None:
                estimates["gjr_filtered_bootstrap"] = (
                    {k: gjr_boot_estimates[c][k] for k in ("var_log", "es_log", "var_simple_pct", "es_simple_pct")},
                    gjr_boot_estimates[c]["status"],
                    gjr_boot_estimates[c]["tail_fit_end"],
                    gjr_boot_estimates[c]["forecast_volatility"],
                )
            for window in raw_windows:
                estimates[f"historical_window_{window}"] = (
                    empirical_risk(returns.iloc[i-window:i].to_numpy(), c), "sensitivity", returns.index[i-1], vol)
            for method in quantile_methods:
                estimates[f"historical_quantile_{method}"] = (
                    empirical_variant(returns.iloc[i-base_window:i].to_numpy(), c, method),
                    "sensitivity", returns.index[i-1], vol)
            for method, (risk, status, shape_end, m_vol) in estimates.items():
                rows.append({"date": date, "method": method, "confidence": c, **risk,
                             "status": status, "forecast_volatility": m_vol,
                             "estimation_end": returns.index[i-1], "tail_fit_end": shape_end,
                             "student_df": df if method == "ewma_student_t" else (
                                 (gjr_hist_model.current_params["nu"] if gjr_hist_model else
                                  gjr_boot_model.current_params["nu"] if gjr_boot_model else np.nan)
                                 if method in ("gjr_filtered_historical", "gjr_filtered_bootstrap") else np.nan
                             ),
                             "threshold_quantile": chosen.threshold_quantile if method == "filtered_evt" else np.nan,
                             "evaluation_index": i, "actual_log_return": returns.iloc[i]})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Insufficient history for experimental model warmups and evaluation")
    value = float(base_config["project"]["portfolio_value_usd"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Portfolio notional must be positive and finite")
    result["actual_simple_return"] = np.expm1(result.actual_log_return)
    result["actual_loss_usd"] = -result.actual_simple_return*value
    result["var_usd"] = result.var_simple_pct*value
    result["es_usd"] = result.es_simple_pct*value
    result["log_es_finite"] = np.isfinite(result.es_log)
    result["exception"] = result.actual_log_return < -result.var_log
    return result.sort_values(["date", "method", "confidence"]).reset_index(drop=True), pd.DataFrame(diagnostics)
