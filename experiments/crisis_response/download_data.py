"""Download the extra data the crisis-response models need and save it in data/raw/.

    python -m experiments.crisis_response.download_data

Files written (data/raw/ is git-ignored, so run this once on a new machine):
  vix_yahoo.csv      ^VIX daily close from Yahoo Finance
  vixcls_fred.csv    VIXCLS from FRED (used to fill any gaps in the Yahoo series)
  ohlc.csv           daily open, high, low, close and adjusted close for the seven ETFs

Only public, approved sources are used. Nothing here is used for a forecast on the same day:
the models lag every series by one day before using it.
"""
import warnings
from pathlib import Path
import pandas as pd
import yaml
from src.quantedge.data import download_fred_series

ROOT = Path.cwd()
RAW = ROOT / "data/raw"


def download_ohlc(tickers, start):
    import yfinance as yf
    frames = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError(f"Yahoo returned no OHLC data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Adj Close": "adj_close"}).rename(columns=str.lower)
        raw = raw[["open", "high", "low", "close", "adj_close", "volume"]]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        frames.append(raw.assign(ticker=ticker))
    out = pd.concat(frames).rename_axis("date").reset_index()
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def download_vix(start):
    import yfinance as yf
    raw = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo returned no ^VIX data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    vix = raw["Close"].rename("vix_close")
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix


def main():
    warnings.filterwarnings("ignore")  # yfinance prints many pandas deprecation notes
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    start = cfg["data"]["start_date"]
    RAW.mkdir(parents=True, exist_ok=True)
    ohlc = download_ohlc(cfg["data"]["tickers"], start)
    ohlc.to_csv(RAW / "ohlc.csv", index=False)
    download_vix(start).to_csv(RAW / "vix_yahoo.csv", index_label="date")
    download_fred_series("VIXCLS", RAW / "vixcls_fred.csv")
    print("saved:", ", ".join(p.name for p in sorted(RAW.glob("*.csv"))))
    print("OHLC rows:", len(ohlc), "from", ohlc.date.min().date(), "to", ohlc.date.max().date())


if __name__ == "__main__":
    main()
