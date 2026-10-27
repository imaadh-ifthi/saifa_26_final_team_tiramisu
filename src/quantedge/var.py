import numpy as np
import pandas as pd
from scipy.stats import norm


def rolling_historical_var(log_returns: pd.Series, confidence: float, window: int = 252) -> pd.Series:
    if not 0 < confidence < 1 or window < 2:
        raise ValueError("confidence must be in (0, 1) and window >= 2")
    return (-log_returns.rolling(window, min_periods=window).quantile(1-confidence).shift(1)).clip(lower=0).dropna().rename(f"historical_var_{int(confidence*100)}")


def rolling_parametric_var(log_returns: pd.Series, confidence: float, ewma_lambda: float = .94) -> pd.Series:
    if not 0 < confidence < 1 or not 0 < ewma_lambda < 1:
        raise ValueError("invalid confidence or EWMA lambda")
    vol = log_returns.ewm(alpha=1-ewma_lambda, min_periods=60, adjust=False).std().shift(1)
    return (-norm.ppf(1-confidence)*vol).dropna().rename(f"parametric_var_{int(confidence*100)}")


def rolling_bootstrap_monte_carlo_var(log_returns: pd.Series, confidence: float, window: int = 252, n_sims: int = 10000, seed: int = 42) -> pd.Series:
    if not 0 < confidence < 1 or window < 2 or n_sims < 100:
        raise ValueError("invalid confidence, window, or simulation count")
    values, idx = log_returns.to_numpy(dtype=float), log_returns.index
    forecasts, dates = [], []
    for i in range(window, len(values)):
        trailing = values[i-window:i]
        trailing = trailing[np.isfinite(trailing)]
        if len(trailing) < window:
            continue
        rng = np.random.default_rng(seed + i)
        forecasts.append(max(0.0, -float(np.quantile(rng.choice(trailing, size=n_sims, replace=True), 1-confidence))))
        dates.append(idx[i])
    return pd.Series(forecasts, index=pd.DatetimeIndex(dates), name=f"monte_carlo_var_{int(confidence*100)}")


def make_var_forecast_table(log_returns: pd.Series, var_series: pd.Series, method: str, confidence: float, portfolio_value_usd: float) -> pd.DataFrame:
    aligned = pd.concat([log_returns.rename("actual_log_return"), var_series.rename("var_log")], axis=1).dropna()
    simple = 1 - np.exp(-aligned["var_log"])
    return pd.DataFrame({"date": aligned.index, "method": method, "confidence": confidence, "var_log": aligned["var_log"].to_numpy(), "var_simple_pct": simple.to_numpy(), "var_usd": simple.to_numpy()*portfolio_value_usd, "actual_log_return": aligned["actual_log_return"].to_numpy(), "exception": (aligned["actual_log_return"] < -aligned["var_log"]).to_numpy(dtype=bool)})


def build_var_suite(log_returns: pd.Series, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current-allocation historical replay, now with ES and common model warmups.

    Formal validation uses analytics.run_analytical_layers and training-only weights.
    """
    from .risk import build_risk_forecasts, summarize_risk
    forecasts, _ = build_risk_forecasts(log_returns, config, include_evt=False)
    forecasts["evaluation_basis"] = "retrospective_current_allocation"
    return forecasts, summarize_risk(forecasts)
