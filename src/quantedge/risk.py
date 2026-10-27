"""Shared one-step-ahead VaR/ES forecasts in log, simple-return, and USD units."""
import numpy as np
import pandas as pd
from .tail import empirical_risk, normal_risk


def build_risk_forecasts(log_returns: pd.Series, config: dict,
                         start_date=None, include_evt: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = log_returns.to_numpy(dtype=float)
    if (log_returns.index.has_duplicates or not log_returns.index.is_monotonic_increasing
            or not np.isfinite(values).all()):
        raise ValueError("Returns must have a sorted unique index and finite values")
    cfg = config["var"]
    confidences = cfg["confidence_levels"]
    historical_window = int(cfg["historical_window"])
    mc_window, n_sims = int(cfg["monte_carlo_window"]), int(cfg["monte_carlo_simulations"])
    decay = float(cfg["ewma_lambda"])
    warmup = int(cfg.get("ewma_min_periods", 60))
    value = float(config["project"]["portfolio_value_usd"])
    if (min(historical_window, mc_window, warmup) < 2 or n_sims < 100 or not 0 < decay < 1
            or not confidences or any(not .5 < c < 1 for c in confidences)
            or value <= 0 or not np.isfinite(value)):
        raise ValueError("Invalid risk forecast settings")
    vol = log_returns.ewm(alpha=1-decay, min_periods=warmup, adjust=False).std().shift(1)
    rows = []
    for i in range(max(historical_window, mc_window, warmup), len(values)):
        date = log_returns.index[i]
        if start_date is not None and date < pd.Timestamp(start_date):
            continue
        trailing = values[i-historical_window:i]
        rng = np.random.default_rng(config["project"]["seed"]+i)
        simulated = rng.choice(values[i-mc_window:i], size=n_sims, replace=True)
        for c in confidences:
            estimates = {"historical": empirical_risk(trailing, c),
                         "parametric_ewma": normal_risk(float(vol.iloc[i]), c),
                         "monte_carlo_bootstrap": empirical_risk(simulated, c)}
            for method, estimates_for_method in estimates.items():
                rows.append({"date": date, "method": method, "confidence": c,
                             "estimation_end": log_returns.index[i-1], "status": "ok",
                             **estimates_for_method})
    forecasts = pd.DataFrame(rows)
    diagnostics = pd.DataFrame()
    if include_evt:
        raise NotImplementedError("EVT integration follows in a separate commit")
        forecasts = pd.concat([forecasts, evt], ignore_index=True)
    if forecasts.empty:
        raise ValueError("Insufficient observations to produce risk forecasts")
    forecasts["actual_log_return"] = forecasts.date.map(log_returns)
    forecasts["actual_simple_return"] = np.expm1(forecasts.actual_log_return)
    forecasts["actual_loss_usd"] = -forecasts.actual_simple_return*value
    forecasts["var_usd"] = forecasts.var_simple_pct*value
    forecasts["es_usd"] = forecasts.es_simple_pct*value
    forecasts["log_es_finite"] = np.isfinite(forecasts.es_log)
    forecasts["exception"] = forecasts.actual_log_return < -forecasts.var_log
    forecasts["evaluation_index"] = forecasts.date.map(pd.Series(np.arange(len(values)), index=log_returns.index))
    return forecasts.sort_values(["date", "method", "confidence"]).reset_index(drop=True), diagnostics


def summarize_risk(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, confidence), frame in forecasts.groupby(["method", "confidence"], sort=True):
        latest = frame.sort_values("date").iloc[-1]
        rows.append({"method": method, "confidence": confidence, "latest_date": latest.date,
                     "latest_var_pct": latest.var_simple_pct, "latest_var_usd": latest.var_usd,
                     "latest_es_pct": latest.es_simple_pct, "latest_es_usd": latest.es_usd,
                     "average_var_pct": frame.var_simple_pct.mean(),
                     "number_of_exceptions": int(frame.exception.sum()), "observations": len(frame)})
    return pd.DataFrame(rows)


def common_evaluation_sample(forecasts: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Equal dates across methods/confidences; trim warmups, reject internal model gaps."""
    expected = {(m, c) for m in methods for c in forecasts.confidence.unique()}
    groups = {(m, c): set(f.date) for (m, c), f in forecasts.groupby(["method", "confidence"])}
    if not expected.issubset(groups):
        raise ValueError("A required risk model produced no forecasts")
    common = set.intersection(*(groups[key] for key in expected))
    if not common:
        raise ValueError("Risk models have no common evaluation dates")
    subset = forecasts[forecasts.method.isin(methods) & forecasts.date.isin(common)].copy()
    for _, frame in subset.groupby(["method", "confidence"]):
        if (frame.sort_values("date").evaluation_index.diff().dropna() != 1).any():
            raise ValueError("Risk model availability has internal gaps; inspect EVT diagnostics")
    return subset.reset_index(drop=True)
