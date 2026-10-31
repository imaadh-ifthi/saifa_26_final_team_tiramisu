"""Integration tests for the Streamlit application layer and output consistency."""
import pytest
import numpy as np
import pandas as pd

from app.schemas import PortfolioBasis, ASSET_REGISTRY
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_performance_series,
    load_kpi_summary,
    load_var_summary,
    load_var_forecasts,
    load_backtest_results,
    load_stress_summary,
    load_stress_paths,
    load_stress_shock_contributions,
    load_risk_contributions,
    load_data_quality_report,
    load_adf_report,
    load_efficient_frontier,
    clear_app_cache,
)
from app.charts import (
    plot_allocation_donut,
    plot_cumulative_wealth,
    plot_drawdowns,
    plot_efficient_frontier,
    plot_var_es_time_series,
    plot_var_method_comparison_bar,
    plot_validation_breaches,
    plot_stress_paths,
    plot_hypothetical_shocks,
    plot_risk_contributions_diverging,
)


def test_all_output_tables_load_successfully():
    """Verify all 15 required analytical outputs load cleanly without errors."""
    portfolio_meta = load_selected_portfolio()
    assert portfolio_meta["portfolio_value_usd"] == 10000000

    curr_weights = load_portfolio_weights(PortfolioBasis.CURRENT)
    assert len(curr_weights) == 7

    val_weights = load_portfolio_weights(PortfolioBasis.VALIDATION)
    assert len(val_weights) == 7

    perf_curr = load_performance_series(PortfolioBasis.CURRENT)
    assert len(perf_curr) == 4980

    perf_val = load_performance_series(PortfolioBasis.VALIDATION)
    assert len(perf_val) == 2212

    kpi_curr = load_kpi_summary(PortfolioBasis.CURRENT)
    assert kpi_curr["observations"] == 4980

    kpi_val = load_kpi_summary(PortfolioBasis.VALIDATION)
    assert kpi_val["observations"] == 2212

    var_sum_curr = load_var_summary(PortfolioBasis.CURRENT)
    assert len(var_sum_curr) == 9  # 3 methods x 3 confidences

    var_sum_val = load_var_summary(PortfolioBasis.VALIDATION)
    assert len(var_sum_val) == 18  # 4 baseline + 2 experimental GJR methods x 3 confidences
    assert "gjr_filtered_historical" in set(var_sum_val["method"])

    forecasts_val = load_var_forecasts(PortfolioBasis.VALIDATION)
    gjr_forecasts = forecasts_val[forecasts_val["method"] == "gjr_filtered_historical"]
    assert len(gjr_forecasts) == 2212 * 3
    assert gjr_forecasts["exception"].dtype == bool
    assert "gjr_filtered_historical" not in set(
        load_var_forecasts(PortfolioBasis.CURRENT)["method"])

    backtests = load_backtest_results()
    assert len(backtests) == 18
    assert "gjr_filtered_historical" in set(backtests["method"])

    stress_sum = load_stress_summary()
    assert len(stress_sum) == 4

    stress_paths = load_stress_paths()
    assert len(stress_paths) > 500

    shock_contribs = load_stress_shock_contributions()
    assert len(shock_contribs) == 7

    risk_contribs = load_risk_contributions()
    assert len(risk_contribs) == 42  # 2 methods x 3 confidences x 7 assets

    quality = load_data_quality_report()
    assert len(quality) == 7

    adf = load_adf_report()
    assert len(adf) == 7

    frontier = load_efficient_frontier()
    assert len(frontier) == 6


def test_chart_generation_does_not_raise():
    """Verify Plotly figure generation functions execute without exceptions."""
    weights_df = load_portfolio_weights(PortfolioBasis.CURRENT)
    perf_df = load_performance_series(PortfolioBasis.CURRENT)
    frontier_df = load_efficient_frontier()
    var_sum = load_var_summary(PortfolioBasis.CURRENT)
    forecasts_df = load_var_forecasts(PortfolioBasis.VALIDATION)
    paths_df = load_stress_paths()
    shock_df = load_stress_shock_contributions()
    risk_df = load_risk_contributions()

    fig1 = plot_allocation_donut(weights_df)
    assert fig1 is not None

    fig2 = plot_cumulative_wealth(perf_df)
    assert fig2 is not None

    fig3 = plot_drawdowns(perf_df)
    assert fig3 is not None

    fig4 = plot_efficient_frontier(frontier_df)
    assert fig4 is not None

    fig5 = plot_var_es_time_series(forecasts_df, "historical", 0.95)
    assert fig5 is not None

    fig6 = plot_var_method_comparison_bar(var_sum, 0.95)
    assert fig6 is not None

    fig7 = plot_validation_breaches(forecasts_df, "parametric_ewma", 0.99)
    assert fig7 is not None

    fig8 = plot_stress_paths(paths_df, "GFC 2008")
    assert fig8 is not None

    fig9 = plot_hypothetical_shocks(shock_df)
    assert fig9 is not None

    fig10 = plot_risk_contributions_diverging(risk_df, "component_var_usd", 0.95, "historical_simple")
    assert fig10 is not None


def test_hypothetical_shocks_sum_to_total_pnl():
    """Verify position P&L contributions sum exactly to scenario net P&L."""
    stress_sum = load_stress_summary()
    shock_contribs = load_stress_shock_contributions()

    hypo_summary = stress_sum[stress_sum["type"] == "hypothetical_one_period"].iloc[0]
    expected_pnl = float(hypo_summary["pnl_usd"])
    sum_contribs = float(shock_contribs["pnl_contribution_usd"].sum())

    assert np.isclose(expected_pnl, sum_contribs, atol=1e-2)


def test_cache_clearing_works():
    """Verify cache clear utility runs safely."""
    clear_app_cache()
    # Reloading should succeed after cache clear
    df = load_portfolio_weights(PortfolioBasis.CURRENT)
    assert not df.empty
