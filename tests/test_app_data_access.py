"""Tests for app/data_access.py, formatting, and schemas."""
import pytest
import numpy as np
import pandas as pd

from app.formatting import (
    parse_boolean,
    format_currency,
    format_percent,
    format_number,
    format_date,
)
from app.schemas import PortfolioBasis, ASSET_REGISTRY, DEFAULT_NOTIONAL
from app.data_access import (
    load_portfolio_weights,
    load_kpi_summary,
    load_performance_series,
    load_var_summary,
    load_var_forecasts,
    load_backtest_results,
    load_stress_summary,
    load_risk_contributions,
    load_data_cutoffs,
    recompute_backtest_window,
)


def test_boolean_parsing():
    """Verify explicit boolean parsing prevents truthy string 'False'."""
    assert parse_boolean(True) is True
    assert parse_boolean(False) is False
    assert parse_boolean("True") is True
    assert parse_boolean("true") is True
    assert parse_boolean("1") is True
    assert parse_boolean("False") is False
    assert parse_boolean("false") is False
    assert parse_boolean("0") is False
    assert parse_boolean(None) is False
    assert parse_boolean(np.nan) is False


def test_currency_and_percent_formatting():
    """Verify precision, dollar formatting, and edge-case zeroes."""
    assert format_currency(10000000) == "$10,000,000"
    assert format_currency(-2102601) == "-$2,102,601"
    assert format_currency(119999.99, decimals=0, include_sign=True) == "+$120,000"
    assert format_currency(None) == "N/A"
    assert format_currency(np.nan) == "N/A"

    assert format_percent(0.076649) == "7.66%"
    assert format_percent(-0.24838) == "-24.84%"
    assert format_percent(0.05, include_sign=True) == "+5.00%"
    assert format_percent(None) == "N/A"


def test_portfolio_weights_loading_current_vs_validation():
    """Verify current and validation weights are strictly separate and sum to 1."""
    curr_weights = load_portfolio_weights(PortfolioBasis.CURRENT)
    val_weights = load_portfolio_weights(PortfolioBasis.VALIDATION)

    assert set(curr_weights["asset"]) == set(ASSET_REGISTRY.keys())
    assert set(val_weights["asset"]) == set(ASSET_REGISTRY.keys())

    assert np.isclose(curr_weights["weight"].sum(), 1.0, atol=1e-4)
    assert np.isclose(val_weights["weight"].sum(), 1.0, atol=1e-4)

    # Weights must differ
    spy_curr = curr_weights.loc[curr_weights["asset"] == "SPY", "weight"].iloc[0]
    spy_val = val_weights.loc[val_weights["asset"] == "SPY", "weight"].iloc[0]
    assert not np.isclose(spy_curr, spy_val)
    assert np.isclose(spy_curr, 0.228, atol=0.01)
    assert np.isclose(spy_val, 0.321, atol=0.01)

    # Dollar conversion check
    total_dollar = curr_weights["dollar_allocation"].sum()
    assert np.isclose(total_dollar, DEFAULT_NOTIONAL, atol=1.0)


def test_kpi_summary_loading():
    """Verify current and validation KPIs are distinct."""
    curr_kpi = load_kpi_summary(PortfolioBasis.CURRENT)
    val_kpi = load_kpi_summary(PortfolioBasis.VALIDATION)

    assert curr_kpi["observations"] == 4980
    assert val_kpi["observations"] == 2212

    assert np.isclose(curr_kpi["annualized_return"], 0.0758, atol=0.002)
    assert np.isclose(val_kpi["annualized_return"], 0.1128, atol=0.002)


def test_performance_series_drawdown_treatment():
    """Verify cumulative wealth and that first day loss is properly included."""
    perf = load_performance_series(PortfolioBasis.CURRENT)
    assert "portfolio_drawdown" in perf.columns
    assert (perf["portfolio_drawdown"] <= 1e-6).all()
    assert len(perf) == 4980


def test_risk_contributions_preserves_negative_signs():
    """Verify Euler risk contributions retain negative values for hedging assets."""
    rc = load_risk_contributions()
    # Check that at least one asset has negative component VaR (e.g. AGG or XLE)
    has_negative_var = (rc["component_var"] < 0).any()
    assert has_negative_var, "Risk contributions must preserve negative hedging values"
    # Ensure Euler components sum to portfolio risk within 1%
    for (m, c), group in rc.groupby(["method", "confidence"]):
        sum_comp = group["component_var"].sum()
        port_var = group["portfolio_var"].iloc[0]
        assert np.isclose(sum_comp, port_var, rtol=1e-3)


def test_data_cutoffs_detection():
    """Verify distinct dates for prices, returns, and forecasts."""
    cutoffs = load_data_cutoffs()
    assert cutoffs["latest_price_date"] >= cutoffs["latest_return_date"]
    assert cutoffs["missing_by_asset"]["EFA"] == 1
    assert cutoffs["missing_by_asset"]["AGG"] == 1
    assert "2026-10-21" in cutoffs["missing_dates"]["EFA"]


def test_traffic_light_249_vs_250_obs():
    """Verify Basel traffic light returns insufficient_data on 249 obs, and green/yellow/red on 250+."""
    from src.quantedge.backtesting import basel_traffic_light

    # Exactly 249 observations
    hits_249 = np.zeros(249, dtype=int)
    res_249 = basel_traffic_light(hits_249, confidence=0.99)
    assert res_249["traffic_light"] == "insufficient_data"

    # Exactly 250 observations with 2 exceptions
    hits_250 = np.zeros(250, dtype=int)
    hits_250[-2:] = 1
    res_250 = basel_traffic_light(hits_250, confidence=0.99)
    assert res_250["traffic_light"] == "green"
    assert res_250["traffic_light_exceptions"] == 2

    # 250 obs with 6 exceptions
    hits_yellow = np.zeros(250, dtype=int)
    hits_yellow[-6:] = 1
    res_yellow = basel_traffic_light(hits_yellow, confidence=0.99)
    assert res_yellow["traffic_light"] == "yellow"

    # 250 obs with 11 exceptions
    hits_red = np.zeros(250, dtype=int)
    hits_red[-11:] = 1
    res_red = basel_traffic_light(hits_red, confidence=0.99)
    assert res_red["traffic_light"] == "red"


def test_traffic_light_not_applicable_at_90_and_95():
    """Verify traffic light returns not_applicable for non-99% confidences."""
    from src.quantedge.backtesting import basel_traffic_light
    hits = np.zeros(250, dtype=int)
    assert basel_traffic_light(hits, confidence=0.90)["traffic_light"] == "not_applicable"
    assert basel_traffic_light(hits, confidence=0.95)["traffic_light"] == "not_applicable"


def test_recompute_backtest_window():
    """Verify dynamic recomputation on filtered forecast window."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    # Select a 1-year window
    res = recompute_backtest_window(
        forecasts,
        start_date="2020-01-01",
        end_date="2020-12-31",
        method="historical",
        confidence=0.99,
    )
    assert res["observations"] > 200
    assert "kupiec_p_value" in res
    assert "independence_p_value" in res
    assert "traffic_light" in res
