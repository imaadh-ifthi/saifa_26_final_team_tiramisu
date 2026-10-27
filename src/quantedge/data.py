from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests


def download_prices(tickers: list[str], start: str, end: str | None = None, cache_path: Path | None = None) -> pd.DataFrame:
    """Fetch adjusted closes from Yahoo Finance; optionally cache raw source data."""
    try:
        import yfinance as yf
        raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="column", threads=True)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed for {tickers}: {exc}") from exc
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Yahoo Finance response has no Adj Close field")
        prices = raw["Adj Close"].copy()
    elif "Adj Close" in raw.columns:
        prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        raise RuntimeError("Yahoo Finance response has no Adj Close field")
    prices = prices.reindex(columns=tickers).dropna(how="all")
    absent = [ticker for ticker in tickers if prices[ticker].notna().sum() == 0]
    if absent:
        raise RuntimeError(f"No adjusted close data returned for: {absent}")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path, index_label="date")
    return prices


def download_fred_series(series_id: str = "DGS3MO", cache_path: Path | None = None) -> pd.Series:
    """Fetch a public FRED CSV series and raise a clear error on source failure."""
    try:
        response = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=30)
        response.raise_for_status()
        first = response.text.splitlines()[0].strip().lower()
        date_col = "observation_date" if "observation_date" in first else "DATE"
        frame = pd.read_csv(io.StringIO(response.text), parse_dates=[date_col])
        values = pd.to_numeric(frame[series_id].replace(".", np.nan), errors="coerce")
        series = pd.Series(values.to_numpy(), index=pd.to_datetime(frame[date_col]), name=series_id).dropna().sort_index()
        if series.empty:
            raise ValueError("empty series")
    except Exception as exc:
        raise RuntimeError(f"FRED download failed for {series_id}: {exc}") from exc
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(cache_path, index_label="date")
    return series


def load_or_download_prices(tickers: list[str], start: str, end: str | None, cache: bool, root: Path) -> pd.DataFrame:
    cache_path = root / "data/raw/adjusted_close.csv"
    if cache and cache_path.exists():
        prices = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        if set(tickers).issubset(prices.columns) and prices.index.min() <= pd.Timestamp(start):
            return prices.reindex(columns=tickers)
    return download_prices(tickers, start, end, cache_path if cache else None)
