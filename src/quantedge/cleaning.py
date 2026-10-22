import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive to compute log returns")
    simple = prices.pct_change(fill_method=None).iloc[1:]
    log = np.log(prices / prices.shift(1)).iloc[1:]
    return log, simple


def flag_missing(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.isna()


def flag_extreme_returns(log_returns: pd.DataFrame, mad_threshold: float = 8.0) -> pd.DataFrame:
    median = log_returns.median()
    mad = (log_returns - median).abs().median().replace(0, np.nan)
    modified_z = 0.6745 * (log_returns - median).divide(mad)
    return modified_z.abs().gt(mad_threshold).fillna(False)


def adf_report(log_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in log_returns:
        values = log_returns[col].dropna()
        if len(values) < 20 or values.nunique() < 2:
            rows.append({"asset": col, "adf_statistic": np.nan, "p_value": np.nan, "used_lag": np.nan, "n_obs": len(values), "stationary_at_1pct": False, "status": "insufficient or constant observations"})
            continue
        try:
            result = adfuller(values, autolag="AIC")
            rows.append({"asset": col, "adf_statistic": result[0], "p_value": result[1], "used_lag": result[2], "n_obs": result[3], "stationary_at_1pct": result[1] < .01, "status": "ok"})
        except Exception as exc:
            rows.append({"asset": col, "adf_statistic": np.nan, "p_value": np.nan, "used_lag": np.nan, "n_obs": len(values), "stationary_at_1pct": False, "status": str(exc)})
    return pd.DataFrame(rows)


def build_data_quality_report(prices: pd.DataFrame, log_returns: pd.DataFrame, missing_flags: pd.DataFrame, outlier_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in prices.columns:
        valid, returns = prices[col].dropna(), log_returns[col].dropna()
        rows.append({"asset": col, "first_price_date": valid.index.min() if not valid.empty else pd.NaT, "last_price_date": valid.index.max() if not valid.empty else pd.NaT, "missing_price_days": int(missing_flags[col].sum()), "extreme_return_flags": int(outlier_flags[col].sum()), "min_log_return": returns.min() if not returns.empty else np.nan, "max_log_return": returns.max() if not returns.empty else np.nan})
    return pd.DataFrame(rows)
