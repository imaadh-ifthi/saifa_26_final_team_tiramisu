import numpy as np
import pandas as pd
from src.quantedge.cleaning import compute_returns


def test_returns_are_aligned_sorted_and_nonempty():
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    prices = pd.DataFrame({"SPY": np.arange(100, 108), "AGG": np.arange(90, 98)}, index=dates)
    log_returns, simple_returns = compute_returns(prices)
    assert not prices.empty
    assert list(log_returns.columns) == ["SPY", "AGG"]
    assert log_returns.shape == simple_returns.shape
    assert log_returns.index.is_monotonic_increasing
    assert not log_returns.isna().all().any()
