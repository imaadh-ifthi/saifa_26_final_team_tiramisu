import numpy as np
import pandas as pd

from experiments.improved_risk.evt_response import detector_path
from experiments.improved_risk.filtered_response import build_filtered_response
from experiments.improved_risk.models import causal_volatility
from src.quantedge.tail import empirical_risk


def test_filtered_response_starts_after_trigger_and_preserves_empirical_tail_math():
    dates = pd.bdate_range("2010-01-01", periods=95)
    returns = pd.Series(np.sin(np.arange(95) * .7) * .004, index=dates)
    returns.iloc[80] = -.05
    returns = returns.iloc[:84]
    dates = returns.index
    slow = causal_volatility(returns, .94, 20)
    fast = causal_volatility(returns, .80, 20)
    alerts = detector_path(returns, slow, threshold=2., drift=1., active_days=3)
    assert alerts.loc[80, "trigger_after_close"]
    assert not alerts.loc[80, "alert_active_for_forecast"]
    assert alerts.loc[81, "alert_active_for_forecast"]
    value, seed, window, simulations = 10_000_000., 42, 40, 200
    residuals = returns / slow
    rows = []
    for i in range(78, 84):
        shock = residuals.iloc[i-window:i].to_numpy() * slow.iloc[i]
        bootstrap = np.random.default_rng(seed+i).choice(shock, simulations, replace=True)
        for name, sample in (("filtered_historical", shock),
                             ("filtered_bootstrap", bootstrap)):
            for confidence in (.9, .95, .99):
                risk = empirical_risk(sample, confidence)
                rows.append({"date": dates[i], "method": name,
                             "confidence": confidence, "estimation_end": dates[i-1],
                             "forecast_volatility": slow.iloc[i],
                             "var_log": risk["var_log"], "es_log": risk["es_log"],
                             "var_simple_pct": risk["var_simple_pct"],
                             "es_simple_pct": risk["es_simple_pct"],
                             "var_usd": risk["var_simple_pct"]*value,
                             "es_usd": risk["es_simple_pct"]*value,
                             "actual_log_return": returns.iloc[i],
                             "exception": returns.iloc[i] < -risk["var_log"]})
    result = build_filtered_response(pd.DataFrame(rows), returns, slow, fast,
                                     alerts, value, seed, window, simulations,
                                     ["filtered_historical", "filtered_bootstrap"],
                                     [.9, .95, .99])
    same_day = result[result.date.eq(dates[80])]
    assert same_day.response_ratio.eq(1).all()
    next_day = result[result.date.eq(dates[81])]
    alert = next_day[next_day.variant.eq("alert")]
    baseline = next_day[next_day.variant.eq("reference")]
    assert alert.response_ratio.gt(1).all()
    assert np.all(alert.var_usd.to_numpy() > baseline.var_usd.to_numpy())
    assert np.all(alert.es_usd.to_numpy() > baseline.es_usd.to_numpy())
    assert (result.estimation_end < result.date).all()
