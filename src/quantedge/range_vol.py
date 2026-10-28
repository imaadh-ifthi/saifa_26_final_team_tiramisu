"""Daily volatility from open, high, low and close prices, plus a simple HAR model on it.

A single close-to-close return tells you very little about how wild a day was. A day can
swing 4% and still close flat. The high-low range looks at the whole day, so it is a much
sharper measure (Parkinson 1980, Garman and Klass 1980). We use Garman-Klass for the
trading hours and add the squared overnight gap, because ETFs often jump at the open:

    v_t = ln(O_t / C_{t-1})^2 + 0.5 * ln(H_t / L_t)^2 - (2 ln 2 - 1) * ln(C_t / O_t)^2

Open and close are first moved onto the dividend-adjusted scale (using that day's
adj_close / close), so a dividend payment doesn't look like an overnight drop.

A portfolio has no high or low price of its own. So we combine each asset's variance with
a 60-day correlation of daily returns:  v_p = w' D C D w,  where D = diag(sqrt(v_i)).

HAR (Corsi 2009): predict tomorrow's log variance from the average log variance of the
last day, the last week (5 days) and the last month (22 days). Simple, and hard to beat.
"""
import numpy as np
import pandas as pd

GK_CONST = 2 * np.log(2) - 1


def garman_klass_gap(ohlc: pd.DataFrame) -> pd.Series:
    """Daily variance for one asset. ohlc has open, high, low, close, adj_close and a date index."""
    f = ohlc.sort_index()
    factor = f.adj_close / f.close
    o, c = f.open * factor, f.adj_close
    gap = np.log(o / c.shift(1))
    hl = np.log(f.high / f.low)
    co = np.log(f.close / f.open)
    v = gap**2 + 0.5 * hl**2 - GK_CONST * co**2
    return v.rename("gk_variance")


def asset_range_variances(ohlc_long: pd.DataFrame, tickers) -> pd.DataFrame:
    """Wide table (date x ticker) of Garman-Klass-with-gap variances from the long ohlc.csv."""
    out = {}
    for t in tickers:
        f = ohlc_long[ohlc_long.ticker == t].set_index("date")
        out[t] = garman_klass_gap(f)
    return pd.DataFrame(out)


def portfolio_range_variance(asset_var: pd.DataFrame, asset_returns: pd.DataFrame, weights: pd.Series,
                             corr_window: int = 60, min_periods: int = 20) -> pd.Series:
    """w' D C D w each day. C uses returns up to and including that day (lag the result before use).

    The first few days of the data have no correlation yet. There we fall back to that day's
    squared portfolio return, which is a rough but honest one-day variance. It only touches
    the first min_periods days of 2007.
    """
    w = weights[weights.abs() > 1e-9]
    names = list(w.index)
    v = asset_var[names].to_numpy()
    r = asset_returns[names].reindex(asset_var.index)
    corr = r.rolling(corr_window, min_periods=min_periods).corr().to_numpy().reshape(len(r), len(names), len(names))
    d = np.sqrt(v) * w.to_numpy()
    out = np.einsum("ti,tij,tj->t", d, corr, d)
    fallback = (r.to_numpy() @ w.to_numpy()) ** 2
    out = np.where(np.isfinite(out), out, fallback)
    return pd.Series(out, index=asset_var.index, name="portfolio_range_variance")


def har_design(log_v: np.ndarray) -> np.ndarray:
    """Columns: 1, log v_{t-1}, mean log v over t-5..t-1, mean over t-22..t-1 (rows t = 22..n)."""
    n = len(log_v)
    c = np.concatenate([[0.0], np.cumsum(log_v)])
    rows = np.arange(22, n + 1)          # include n: the row for the next, unseen day
    day = log_v[rows - 1]
    week = (c[rows] - c[rows - 5]) / 5
    month = (c[rows] - c[rows - 22]) / 22
    return np.column_stack([np.ones(len(rows)), day, week, month])


def fit_har(range_var: np.ndarray) -> dict:
    """OLS of log v_t on its HAR lags. Returns coefficients and residual variance (log scale)."""
    v = np.maximum(np.asarray(range_var, dtype=float), 1e-12)
    log_v = np.log(v)
    x = har_design(log_v)[:-1]
    y = log_v[22:]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return {"beta": beta, "resid_var": float(resid.var(ddof=x.shape[1]))}


def har_variance_path(range_var: np.ndarray, fit: dict) -> np.ndarray:
    """One-step-ahead variance forecasts for rows 22..n (the last is the next, unseen day).

    The exp(0.5 s^2) factor turns a log forecast into a mean-variance forecast. The overall
    level does not matter for EVT on standardised returns, only the day-to-day shape.
    """
    v = np.maximum(np.asarray(range_var, dtype=float), 1e-12)
    pred = har_design(np.log(v)) @ fit["beta"]
    return np.exp(pred + 0.5 * fit["resid_var"])
