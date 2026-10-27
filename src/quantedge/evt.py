"""POT/GPD tail fits, threshold stability diagnostics and sensitivity estimates."""
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import genpareto, kstest


def gpd_risk(threshold: float, shape: float, scale: float,
             exceedance_probability: float, confidence: float) -> dict:
    if (not np.isfinite([threshold, shape, scale, exceedance_probability, confidence]).all()
            or scale <= 0 or not 0 < exceedance_probability < 1 or not 0 < confidence < 1):
        raise ValueError("invalid GPD parameters")
    tail_probability = 1-confidence
    if tail_probability > exceedance_probability + 1e-12:
        raise ValueError("Requested quantile lies below the fitted tail threshold")
    ratio = exceedance_probability/tail_probability
    extra = scale*np.log(ratio) if abs(shape) < 1e-8 else scale*np.expm1(shape*np.log(ratio))/shape
    raw_var = float(threshold+extra)
    conditional_scale = scale+shape*extra
    if not np.isfinite(raw_var) or conditional_scale <= 0:
        raise ValueError("Invalid GPD quantile or conditional scale")
    es_log = (raw_var+scale-shape*threshold)/(1-shape) if shape < 1 else np.inf
    # Conditional excess Y above VaR is GPD(xi, beta + xi*(VaR-u)).
    # E[1-exp(-(VaR+Y))] = 1-exp(-VaR) + exp(-VaR)*integral exp(-y) S_Y(y) dy.
    upper = min(-conditional_scale/shape, 40.0) if shape < 0 else 40.0
    integral = quad(lambda y: np.exp(-y)*genpareto.sf(y, shape, scale=conditional_scale),
                    0, upper, epsabs=1e-9)[0]
    return {"var_log": max(0.0, raw_var), "es_log": max(0.0, es_log),
            "var_simple_pct": max(0.0, float(-np.expm1(-raw_var))),
            "es_simple_pct": max(0.0, float(-np.expm1(-raw_var)+np.exp(-raw_var)*integral))}


def fit_thresholds(log_returns: pd.Series, quantiles: list[float], min_exceedances: int = 50,
                   shape_tolerance: float = .15, scale_tolerance: float = .25) -> pd.DataFrame:
    losses = -np.asarray(log_returns, dtype=float)
    if (losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all()
            or min_exceedances < 3 or shape_tolerance <= 0 or scale_tolerance <= 0
            or not quantiles or any(not 0 < q < 1 for q in quantiles)):
        raise ValueError("Invalid EVT sample or threshold settings")
    rows = []
    for q in sorted(set(quantiles)):
        threshold = float(np.quantile(losses, q))
        excess = losses[losses > threshold]-threshold
        row = {"threshold_quantile": q, "threshold_log": threshold,
               "observations": len(losses), "exceedances": len(excess),
               "exceedance_probability": len(excess)/len(losses),
               "mean_excess": float(excess.mean()) if len(excess) else np.nan,
               "shape": np.nan, "scale": np.nan, "modified_scale": np.nan,
               "ks_statistic": np.nan, "status": "insufficient_exceedances"}
        if len(excess) >= min_exceedances and np.ptp(excess) > 0:
            try:
                shape, _, scale = genpareto.fit(excess, floc=0)
                log_likelihood = genpareto.logpdf(excess, shape, scale=scale).sum()
                if not np.isfinite([shape, scale, log_likelihood]).all() or scale <= 0:
                    raise ValueError("Nonfinite likelihood or invalid scale")
                row.update(shape=float(shape), scale=float(scale),
                           modified_scale=float(scale-shape*threshold),
                           ks_statistic=float(kstest(excess, "genpareto", args=(shape, 0, scale)).statistic),
                           status="ok")
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                row["status"] = f"fit_failed: {exc}"
        rows.append(row)
    result = pd.DataFrame(rows)
    result["stable_with_next"] = False
    for i in range(len(result)-1):
        a, b = result.iloc[i], result.iloc[i+1]
        if a.status == b.status == "ok":
            stable = (abs(a["shape"]-b["shape"]) <= shape_tolerance and
                      abs(a.modified_scale-b.modified_scale)/max(a.scale, b.scale) <= scale_tolerance)
            result.loc[i, "stable_with_next"] = bool(stable)
    return result


def select_threshold(diagnostics: pd.DataFrame, minimum_confidence: float) -> tuple[pd.Series | None, str]:
    eligible = diagnostics[(diagnostics.status == "ok") &
                           (diagnostics.exceedance_probability >= 1-minimum_confidence-1e-12)]
    stable = eligible[eligible.stable_with_next]
    if not stable.empty:
        return stable.iloc[0], "stable_adjacent_thresholds"
    if not eligible.empty:
        # A visible warning accompanies this deterministic fallback; it is not evidence of stability.
        return eligible.sort_values(["ks_statistic", "threshold_quantile"]).iloc[0], "warning_no_stable_threshold"
    return None, "insufficient_tail_data"


def rolling_evt(log_returns: pd.Series, confidences: list[float], config: dict,
                start_date: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = int(config.get("window", 1000))
    refit_every = int(config.get("refit_every", 21))
    if window < 20 or refit_every < 1 or not confidences or any(not 0 < c < 1 for c in confidences):
        raise ValueError("Invalid EVT window, refit interval or confidence")
    quantiles = config.get("threshold_quantiles", [.85, .90, .925, .95])
    forecasts, reports = [], []
    cached = {}
    fit_end = None
    next_refit = 0
    for i in range(window, len(log_returns)):
        date = log_returns.index[i]
        if start_date is not None and date < pd.Timestamp(start_date):
            continue
        if i >= next_refit:
            sample = log_returns.iloc[i-window:i]
            diagnostics = fit_thresholds(sample, quantiles, config.get("min_exceedances", 50),
                                         config.get("shape_tolerance", .15), config.get("scale_tolerance", .25))
            chosen, selection_status = select_threshold(diagnostics, min(confidences))
            fit_end = sample.index[-1]
            cached = {}
            diagnostics["fit_date"] = date
            diagnostics["estimation_end"] = fit_end
            diagnostics["selection_status"] = selection_status
            diagnostics["selected"] = False
            if chosen is not None:
                diagnostics.loc[chosen.name, "selected"] = True
                for c in confidences:
                    cached[c] = {**gpd_risk(chosen.threshold_log, chosen["shape"], chosen.scale,
                                            chosen.exceedance_probability, c),
                                 "threshold_quantile": chosen.threshold_quantile,
                                 "shape": chosen["shape"], "scale": chosen.scale,
                                 "status": selection_status}
            # Diagnostics include VaR/ES at every candidate, not just the chosen threshold.
            for _, row in diagnostics.iterrows():
                for c in confidences:
                    report = {**row.to_dict(), "confidence": c}
                    if row.status == "ok" and 1-c <= row.exceedance_probability+1e-12:
                        report.update(gpd_risk(row.threshold_log, row["shape"], row.scale,
                                               row.exceedance_probability, c))
                    reports.append(report)
            next_refit = i+refit_every
        for c in confidences:
            if c in cached:
                forecasts.append({"date": date, "method": "evt_gpd", "confidence": c,
                                  "estimation_end": fit_end, **cached[c]})
    return pd.DataFrame(forecasts), pd.DataFrame(reports)
