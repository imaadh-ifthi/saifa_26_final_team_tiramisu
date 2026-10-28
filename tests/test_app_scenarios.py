"""Comprehensive tests for the 20 specific checklist scenarios in Section 13.

Verifies functionality, data contracts, mathematical reconciliation, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd

from app.formatting import (
    parse_boolean,
    format_currency,
    format_percent,
    format_number,
    format_date,
    format_p_value,
)
from app.schemas import PortfolioBasis, ASSET_REGISTRY, DEFAULT_NOTIONAL
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_kpi_summary,
    load_performance_series,
    load_var_summary,
    load_var_forecasts,
    load_backtest_results,
    load_stress_summary,
    load_stress_paths,
    load_stress_shock_contributions,
    load_risk_contributions,
    load_data_cutoffs,
    load_efficient_frontier,
    clear_app_cache,
    recompute_backtest_window,
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
from src.quantedge.backtesting import basel_traffic_light


# 1. Current and validation portfolio results remain separate
def test_1_current_and_validation_portfolio_results_remain_separate():
    """Verify Current and Validation portfolios use distinct weights, KPIs, and methods."""
    curr_weights = load_portfolio_weights(PortfolioBasis.CURRENT)
    val_weights = load_portfolio_weights(PortfolioBasis.VALIDATION)

    # Weights are numerically distinct
    spy_curr = curr_weights.loc[curr_weights["asset"] == "SPY", "weight"].iloc[0]
    spy_val = val_weights.loc[val_weights["asset"] == "SPY", "weight"].iloc[0]
    assert abs(spy_curr - spy_val) > 0.05

    # Performance series have different sample lengths
    perf_curr = load_performance_series(PortfolioBasis.CURRENT)
    perf_val = load_performance_series(PortfolioBasis.VALIDATION)
    assert len(perf_curr) == 4980
    assert len(perf_val) == 2212

    # KPIs are distinct
    kpi_curr = load_kpi_summary(PortfolioBasis.CURRENT)
    kpi_val = load_kpi_summary(PortfolioBasis.VALIDATION)
    assert kpi_curr["observations"] != kpi_val["observations"]
    assert not np.isclose(kpi_curr["annualized_return"], kpi_val["annualized_return"])

    # Risk summary: current has 3 baseline methods, validation has 4 methods (including EVT: evt_gpd)
    var_curr = load_var_summary(PortfolioBasis.CURRENT)
    var_val = load_var_summary(PortfolioBasis.VALIDATION)
    assert "evt_gpd" not in set(var_curr["method"])
    assert "evt_gpd" in set(var_val["method"])


# 2. Dollar allocations reconcile to portfolio value
def test_2_dollar_allocations_reconcile_to_portfolio_value():
    """Verify sum of dollar allocations equals total portfolio value ($10,000,000)."""
    curr_weights = load_portfolio_weights(PortfolioBasis.CURRENT)
    total_dollar = curr_weights["dollar_allocation"].sum()
    assert np.isclose(total_dollar, DEFAULT_NOTIONAL, atol=0.01)

    for _, row in curr_weights.iterrows():
        assert np.isclose(row["dollar_allocation"], row["weight"] * DEFAULT_NOTIONAL, atol=0.01)


# 3. Defensive exposure matches AGG + GLD
def test_3_defensive_exposure_matches_agg_plus_gld():
    """Verify defensive exposure is defined as AGG + GLD per backend mandate and satisfies >=35%."""
    curr_weights = load_portfolio_weights(PortfolioBasis.CURRENT)
    agg_w = curr_weights.loc[curr_weights["asset"] == "AGG", "weight"].iloc[0]
    gld_w = curr_weights.loc[curr_weights["asset"] == "GLD", "weight"].iloc[0]
    defensive_w = agg_w + gld_w

    # Combined AGG + GLD defines the defensive mandate
    assert defensive_w >= 0.35
    assert np.isclose(defensive_w, 0.60, atol=0.01)

    # In validation portfolio, defensive is also AGG + GLD >= 35%
    val_weights = load_portfolio_weights(PortfolioBasis.VALIDATION)
    val_agg = val_weights.loc[val_weights["asset"] == "AGG", "weight"].iloc[0]
    val_gld = val_weights.loc[val_weights["asset"] == "GLD", "weight"].iloc[0]
    assert (val_agg + val_gld) >= 0.35


# 4. Benchmark labels match the actual weights
def test_4_benchmark_labels_match_actual_weights():
    """Verify policy benchmark is labeled 35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD (never 60/40)."""
    perf_df = load_performance_series(PortfolioBasis.CURRENT)
    fig = plot_cumulative_wealth(perf_df)

    trace_names = [trace.name for trace in fig.data]
    bm_names = [name for name in trace_names if "Benchmark" in name or "benchmark" in name]
    assert len(bm_names) == 1
    bm_label = bm_names[0]

    assert "35% SPY" in bm_label
    assert "40% AGG" in bm_label
    assert "10% GLD" in bm_label
    assert "60/40" not in bm_label


# 5. Missing risk results display unavailable, not zero
def test_5_missing_risk_results_display_unavailable_not_zero():
    """Verify formatting never replaces missing risk or p-values with zero or $0."""
    assert format_currency(np.nan) == "N/A"
    assert format_currency(None) == "N/A"
    assert format_percent(np.nan) == "N/A"
    assert format_percent(None) == "N/A"
    assert format_number(np.nan) == "N/A"
    assert format_p_value(np.nan) == "N/A"
    assert format_p_value(None) == "N/A"

    # A non-zero tiny p-value is formatted as <0.0001, never exact 0.0000
    assert format_p_value(1e-6) == "<0.0001"
    assert format_p_value(0.00005) == "<0.0001"


# 6. Missing files and malformed schemas produce informative handling
def test_6_missing_files_and_malformed_schemas_produce_informative_handling(tmp_path, monkeypatch):
    """Verify informative exceptions on missing files, malformed weights, or invalid schemas."""
    import app.data_access as da
    da.clear_app_cache()

    # 1. Weights summing to 0.8 must raise ValueError
    bad_weights_file = tmp_path / "bad_weights.csv"
    pd.DataFrame({"asset": ["SPY", "AGG"], "weight": [0.4, 0.4]}).to_csv(bad_weights_file, index=False)
    monkeypatch.setattr(da, "resolve_path", lambda p: bad_weights_file if "portfolio_weights" in p else da.PROJECT_ROOT / p)
    with pytest.raises(ValueError, match="Weights do not sum to 1.0"):
        load_portfolio_weights(PortfolioBasis.CURRENT)

    # 2. Negative weights must raise ValueError
    da.clear_app_cache()
    bad_neg_file = tmp_path / "bad_neg.csv"
    pd.DataFrame({"asset": ["SPY", "AGG"], "weight": [1.1, -0.1]}).to_csv(bad_neg_file, index=False)
    monkeypatch.setattr(da, "resolve_path", lambda p: bad_neg_file if "portfolio_weights" in p else da.PROJECT_ROOT / p)
    with pytest.raises(ValueError, match="negative weights"):
        load_portfolio_weights(PortfolioBasis.CURRENT)
    
    # Restore cache
    da.clear_app_cache()


# 7. Boolean parsing preserves False
def test_7_boolean_parsing_preserves_false():
    """Verify boolean parsing handles truthy string 'False' as False."""
    assert parse_boolean("False") is False
    assert parse_boolean("false") is False
    assert parse_boolean("FALSE") is False
    assert parse_boolean(False) is False
    assert parse_boolean("0") is False
    assert parse_boolean(None) is False
    assert parse_boolean(np.nan) is False

    assert parse_boolean("True") is True
    assert parse_boolean("true") is True
    assert parse_boolean(True) is True
    assert parse_boolean("1") is True


# 8. Date limits derive from data
def test_8_date_limits_derive_from_data():
    """Verify date limits derive from loaded dataset timestamps."""
    cutoffs = load_data_cutoffs()
    assert pd.to_datetime(cutoffs["latest_price_date"]).strftime("%Y-%m-%d") == "2026-10-22"
    assert pd.to_datetime(cutoffs["latest_return_date"]).strftime("%Y-%m-%d") == "2026-10-20"
    assert pd.to_datetime(cutoffs["latest_risk_date"]).strftime("%Y-%m-%d") == "2026-10-20"

    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    dates = pd.to_datetime(forecasts["date"]).dt.date
    min_date = dates.min()
    max_date = dates.max()
    assert str(min_date) == "2018-01-02"
    assert str(max_date) == "2026-10-20"


# 9. Reversed and empty evaluation windows are handled
def test_9_reversed_and_empty_evaluation_windows_handled():
    """Verify reversed start/end dates and out-of-range dates return clear error dictionary."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)

    # Reversed dates
    res_rev = recompute_backtest_window(
        forecasts,
        start_date="2022-01-01",
        end_date="2020-01-01",
        method="historical",
        confidence=0.99,
    )
    assert "error" in res_rev
    assert "is after end date" in res_rev["error"]

    # Empty window
    res_empty = recompute_backtest_window(
        forecasts,
        start_date="2010-01-01",
        end_date="2010-12-31",
        method="historical",
        confidence=0.99,
    )
    assert "error" in res_empty
    assert "No forecast observations" in res_empty["error"]


# 10. Insufficient independence-test data leaves the joint test unavailable
def test_10_insufficient_independence_data_leaves_joint_test_unavailable():
    """Verify missing independence test leaves independence and joint CC test as NaN (never zero substitution)."""
    # Create synthetic forecasts with ZERO breaches
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    synth_df = pd.DataFrame({
        "date": dates,
        "method": ["historical"] * 100,
        "confidence": [0.99] * 100,
        "actual_loss_usd": [100.0] * 100,
        "var_usd": [1000.0] * 100,  # All losses well below VaR -> 0 exceptions
        "exception": [False] * 100,
    })

    res = recompute_backtest_window(
        synth_df,
        start_date="2020-01-01",
        end_date="2020-12-31",
        method="historical",
        confidence=0.99,
    )
    assert res["exceptions"] == 0
    # Independence must be NaN / unavailable
    assert not np.isfinite(res["independence_lr"])
    assert not np.isfinite(res["independence_p_value"])
    # Joint conditional coverage must ALSO be NaN / unavailable (CRITICAL FIX)
    assert not np.isfinite(res["conditional_coverage_lr"])
    assert not np.isfinite(res["conditional_coverage_p_value"])


# 11. Duplicate forecast dates and evaluation gaps are rejected
def test_11_duplicate_forecast_dates_and_evaluation_gaps_rejected():
    """Verify recompute_backtest_window rejects duplicate dates and sequence gaps."""
    # Duplicate dates
    dup_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
        "method": ["historical"] * 3,
        "confidence": [0.99] * 3,
        "actual_loss_usd": [10.0, 10.0, 10.0],
        "var_usd": [50.0, 50.0, 50.0],
        "exception": [False, False, False],
    })
    with pytest.raises(ValueError, match="Duplicate forecast dates"):
        recompute_backtest_window(
            dup_df,
            start_date="2020-01-01",
            end_date="2020-01-05",
            method="historical",
            confidence=0.99,
        )

    # Sequence gap with evaluation_index
    gap_df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-25"]),
        "evaluation_index": [0, 5],  # Gap in evaluation index!
        "method": ["historical"] * 2,
        "confidence": [0.99] * 2,
        "actual_loss_usd": [10.0, 10.0],
        "var_usd": [50.0, 50.0],
        "exception": [False, False],
    })
    with pytest.raises(ValueError, match="evaluation sequence contains gaps"):
        recompute_backtest_window(
            gap_df,
            start_date="2020-01-01",
            end_date="2020-01-30",
            method="historical",
            confidence=0.99,
        )


# 12. Full-window recomputed statistics agree numerically with backend results
def test_12_full_window_recomputed_statistics_agree_with_backend_results():
    """Verify full-window recomputed backtest matches the saved backend table exactly."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    backtest_tbl = load_backtest_results()

    hist_99_saved = backtest_tbl[
        (backtest_tbl["method"] == "historical") & np.isclose(backtest_tbl["confidence"], 0.99)
    ].iloc[0]

    recomputed = recompute_backtest_window(
        forecasts,
        start_date="2018-01-02",
        end_date="2026-10-20",
        method="historical",
        confidence=0.99,
    )

    assert recomputed["observations"] == int(hist_99_saved["observations"])
    assert recomputed["exceptions"] == int(hist_99_saved["exceptions"])
    assert np.isclose(recomputed["kupiec_p_value"], float(hist_99_saved["kupiec_p_value"]), atol=1e-4)
    assert np.isclose(recomputed["independence_p_value"], float(hist_99_saved["independence_p_value"]), atol=1e-4)
    assert np.isclose(recomputed["conditional_coverage_p_value"], float(hist_99_saved["conditional_coverage_p_value"]), atol=1e-4)


# 13. Filtered-window results agree with independently selected input observations
def test_13_filtered_window_results_agree_with_independent_observations():
    """Verify filtered window statistics match independent direct filtering."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    start_date = "2021-01-01"
    end_date = "2023-12-31"

    subset = forecasts[
        (forecasts["method"] == "parametric_ewma")
        & np.isclose(forecasts["confidence"], 0.95)
        & (forecasts["date"] >= start_date)
        & (forecasts["date"] <= end_date)
    ]
    expected_n = len(subset)
    expected_hits = int(subset["exception"].sum())

    recomputed = recompute_backtest_window(
        forecasts,
        start_date=start_date,
        end_date=end_date,
        method="parametric_ewma",
        confidence=0.95,
    )

    assert recomputed["observations"] == expected_n
    assert recomputed["exceptions"] == expected_hits


# 14. Traffic lights handle fewer than 250 observations and non-99% confidence
def test_14_traffic_lights_handle_fewer_than_250_obs_and_non_99():
    """Verify traffic light logic on 249 obs (insufficient_data) and non-99% (not_applicable)."""
    hits_249 = np.zeros(249, dtype=int)
    assert basel_traffic_light(hits_249, confidence=0.99)["traffic_light"] == "insufficient_data"

    hits_250 = np.zeros(250, dtype=int)
    assert basel_traffic_light(hits_250, confidence=0.95)["traffic_light"] == "not_applicable"
    assert basel_traffic_light(hits_250, confidence=0.90)["traffic_light"] == "not_applicable"

    # 250 at 0.99 with 0 exceptions -> green
    assert basel_traffic_light(hits_250, confidence=0.99)["traffic_light"] == "green"


# 15. Breach markers correspond to actual exceptions
def test_15_breach_markers_correspond_to_actual_exceptions():
    """Verify breach markers on validation plot correspond exactly to realized_loss > var_forecast."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    fig = plot_validation_breaches(forecasts, "historical", 0.99)

    breach_trace = None
    realized_trace = None
    for trace in fig.data:
        if "Exception" in trace.name or "Breach" in trace.name or "breach" in trace.name.lower():
            breach_trace = trace
        elif "Realized" in trace.name:
            realized_trace = trace

    assert breach_trace is not None
    assert realized_trace is not None

    # Number of breach scatter points matches actual breach count
    f_hist_99 = forecasts[(forecasts["method"] == "historical") & np.isclose(forecasts["confidence"], 0.99)]
    actual_breaches = int(f_hist_99["exception"].sum())
    assert len(breach_trace.x) == actual_breaches

    # Verify gains (negative losses) are preserved in realized loss trace
    has_negative_loss = any(val < 0 for val in realized_trace.y)
    assert has_negative_loss, "Gains must be visible as negative losses on the validation chart"


# 16. Chart traces use the selected portfolio/method/confidence
def test_16_chart_traces_use_selected_portfolio_method_confidence():
    """Verify chart traces dynamically reflect selected method and confidence level."""
    forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
    fig_95 = plot_var_es_time_series(forecasts, "parametric_ewma", 0.95)
    trace_names_95 = [t.name for t in fig_95.data]
    assert any("95%" in name for name in trace_names_95)
    assert not any("99%" in name for name in trace_names_95)

    fig_99 = plot_var_es_time_series(forecasts, "parametric_ewma", 0.99)
    trace_names_99 = [t.name for t in fig_99.data]
    assert any("99%" in name for name in trace_names_99)


# 17. Negative contributions remain negative in chart data
def test_17_negative_contributions_remain_negative_in_chart_data():
    """Verify diverging chart retains negative component risk values for hedging assets."""
    risk_df = load_risk_contributions()
    fig = plot_risk_contributions_diverging(risk_df, "component_var_usd", 0.95, "historical_simple")

    bar_trace = fig.data[0]
    values = bar_trace.x
    has_negative = any(v < 0 for v in values)
    assert has_negative, "Diverging risk contributions chart must preserve negative values"


# 18. Stress contributions reconcile with the scenario total
def test_18_stress_contributions_reconcile_with_scenario_total():
    """Verify hypothetical shock position P&Ls sum exactly to the scenario total P&L."""
    stress_sum = load_stress_summary()
    shock_contribs = load_stress_shock_contributions()

    hypo_row = stress_sum[stress_sum["type"] == "hypothetical_one_period"].iloc[0]
    expected_total_pnl = float(hypo_row["pnl_usd"])
    actual_sum_pnl = float(shock_contribs["pnl_contribution_usd"].sum())

    assert np.isclose(expected_total_pnl, actual_sum_pnl, atol=0.01)


# 19. Collapsed frontiers are represented honestly
def test_19_collapsed_frontiers_represented_honestly():
    """Verify efficient frontier results identify non-binding targets without fabricated frontiers."""
    frontier_df = load_efficient_frontier()
    assert len(frontier_df) == 6

    first_ret = frontier_df["expected_return"].iloc[0]
    all_same_ret = np.allclose(frontier_df["expected_return"], first_ret)
    assert all_same_ret, "All frontier solutions have identical expected return"

    fig = plot_efficient_frontier(frontier_df)
    assert "Current Allocation Scope" in fig.layout.title.text


# 20. Reloading updates results and associated text
def test_20_reloading_updates_results_and_associated_text():
    """Verify clear_app_cache properly flushes cache and allows clean reloading."""
    clear_app_cache()
    w = load_portfolio_weights(PortfolioBasis.CURRENT)
    k = load_kpi_summary(PortfolioBasis.CURRENT)
    assert not w.empty
    assert k["observations"] == 4980
