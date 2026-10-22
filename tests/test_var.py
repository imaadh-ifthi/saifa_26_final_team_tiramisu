import numpy as np
import pandas as pd
from src.quantedge.var import rolling_historical_var, make_var_forecast_table


def test_historical_var_is_one_step_ahead_and_exception_is_boolean():
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    returns = pd.Series(np.linspace(-.03, .02, 40), index=dates)
    var = rolling_historical_var(returns, .95, window=10)
    assert (var.index > dates[9]).all()
    table = make_var_forecast_table(returns, var, "historical", .95, 1_000_000)
    assert (table.var_log >= 0).all()
    assert table.exception.dtype == bool
    assert table.var_simple_pct.between(0, 1).all()
