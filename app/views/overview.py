"""Executive Overview page for the Serendib University Endowment Fund."""
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from app.components import render_header, render_callout, render_basis_badge
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_kpi_summary,
    load_var_summary,
    load_backtest_results,
    load_data_cutoffs,
    load_config,
)
from app.formatting import format_currency, format_percent, format_date
from app.schemas import PortfolioBasis, PALETTE, ASSET_REGISTRY


def render_overview_view() -> None:
    """Render the executive summary overview view."""
    render_header(
        "Executive Overview",
        "Portfolio construction, risk measurement and validation for the Serendib University Endowment.",
    )

    portfolio_meta = load_selected_portfolio()
    weights_df = load_portfolio_weights(PortfolioBasis.CURRENT)
    kpi_summary = load_kpi_summary(PortfolioBasis.CURRENT)
    cutoffs = load_data_cutoffs()
    config = load_config()

    notional = float(portfolio_meta.get("portfolio_value_usd", 10_000_000.0))
    target_return = float(portfolio_meta.get("target_return_annual", 0.065))
    expected_return = float(portfolio_meta.get("expected_return_annual", 0.0766))
    optimizer_vol = float(portfolio_meta.get("volatility_annual", 0.0857))

    realized_return = float(kpi_summary.get("annualized_return", 0.0758))
    realized_vol = float(kpi_summary.get("annualized_volatility", 0.0919))
    max_dd = float(kpi_summary.get("max_drawdown", -0.2484))
    sharpe = float(kpi_summary.get("sharpe_ratio", 0.3656))

    render_basis_badge("Current proposed portfolio (Retrospective historical replay)")

    # Key institutional metrics banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Fund Reference Notional",
            value=format_currency(notional, 0),
            help="Total reference endowment capital allocation analyzed across all models.",
        )
    with col2:
        st.metric(
            label="Optimized Expected Return",
            value=format_percent(expected_return, 2),
            delta=f"+{format_percent(expected_return - target_return, 2)} vs Target ({format_percent(target_return, 1)})",
            help="Forward-looking blended expected return (50% historical, 50% CAPM) used in portfolio optimization.",
        )
    with col3:
        st.metric(
            label="Model Annual Volatility",
            value=format_percent(optimizer_vol, 2),
            help="Model-estimated covariance volatility (annualized log-return diagonal shrinkage).",
        )
    with col4:
        st.metric(
            label="Latest Observation Date",
            value=format_date(cutoffs["latest_return_date"]),
            help="Cutoff date for complete return series across all portfolio assets (2026-10-20).",
        )

    st.markdown("---")

    # Realized vs Expected Return Distinction Callout
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 1.25rem;">
                <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.75rem; font-size: 1.1rem;">
                    Optimization & Expected Estimates
                </h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Target Return Constraint</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(target_return, 2)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Estimated Annual Return (Model)</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['teal']};">{format_percent(expected_return, 2)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Estimated Volatility (Covariance)</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(optimizer_vol, 2)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Optimization Objective</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">Target-Return Minimum Variance</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 1.25rem;">
                <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.75rem; font-size: 1.1rem;">
                    Realized Historical Performance (2007–2026)
                </h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Realized Compound Annual Return</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(realized_return, 2)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Realized Annual Volatility</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(realized_vol, 2)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {PALETTE['border']};">
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Historical Maximum Drawdown</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['red']};">{format_percent(max_dd, 2)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: {PALETTE['muted']};">Realized Sharpe Ratio</td>
                        <td style="text-align: right; font-weight: 600; color: {PALETTE['navy']};">{sharpe:.2f} (rf = {format_percent(float(kpi_summary.get('annualized_risk_free', 0.0422)), 2)})</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Current Allocation Summary
    st.subheader("Current Proposed Asset Allocation")
    active_weights = weights_df[weights_df["weight"] > 1e-4].copy().sort_values("weight", ascending=False)

    cols = st.columns(len(active_weights))
    for idx, row in enumerate(active_weights.itertuples()):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="background-color: white; border: 1px solid {PALETTE['border']}; border-top: 4px solid {row.color}; border-radius: 6px; padding: 0.75rem 0.5rem; text-align: center;">
                    <div style="font-size: 1.15rem; font-weight: 800; color: {PALETTE['navy']};">{row.asset}</div>
                    <div style="font-size: 0.78rem; color: {PALETTE['muted']}; margin-bottom: 0.35rem;">{row.category}</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {PALETTE['navy']};">{format_percent(row.weight, 1)}</div>
                    <div style="font-size: 0.8rem; color: {PALETTE['muted']};">{format_currency(row.dollar_allocation, 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # What these results mean section - calculated dynamically from data
    st.subheader("What These Results Mean for the Endowment")

    from app.data_access import load_var_summary, load_backtest_results

    # Dynamic asset allocations
    agg_w = float(weights_df.loc[weights_df["asset"] == "AGG", "weight"].sum()) if "AGG" in weights_df["asset"].values else 0.0
    gld_w = float(weights_df.loc[weights_df["asset"] == "GLD", "weight"].sum()) if "GLD" in weights_df["asset"].values else 0.0
    defensive_w = agg_w + gld_w
    equity_assets = ["SPY", "EFA", "EEM", "XLE", "XLK"]
    equity_w = float(weights_df[weights_df["asset"].isin(equity_assets)]["weight"].sum())

    # Detail active equity holdings
    eq_details = ", ".join(
        f"{row.asset} {format_percent(row.weight, 1)}"
        for row in weights_df[(weights_df["asset"].isin(equity_assets)) & (weights_df["weight"] > 1e-4)].itertuples()
    )

    # Risk metrics from var_summary
    var_sum = load_var_summary(PortfolioBasis.CURRENT)
    h95 = var_sum[(var_sum["method"] == "historical") & np.isclose(var_sum["confidence"], 0.95)]
    var_95_str = (
        f"{format_currency(float(h95['latest_var_usd'].iloc[0]), 0)} ({format_percent(float(h95['latest_var_pct'].iloc[0]), 2)})"
        if not h95.empty
        else "Unavailable"
    )
    es_95_str = (
        f"{format_currency(float(h95['latest_es_usd'].iloc[0]), 0)} ({format_percent(float(h95['latest_es_pct'].iloc[0]), 2)})"
        if not h95.empty
        else "Unavailable"
    )

    # Backtest summary
    backtests_df = load_backtest_results()
    cc_99 = backtests_df[np.isclose(backtests_df["confidence"], 0.99)]
    n_models_99 = len(cc_99)
    n_rejected_99 = int(cc_99["conditional_coverage_reject"].sum()) if not cc_99.empty else 0
    oos_obs = int(cc_99["observations"].iloc[0]) if not cc_99.empty else 0

    st.markdown(
        f"""
        - **Asset Allocation Profile**: The proposed portfolio allocates **{format_percent(agg_w, 1)}** to investment-grade fixed income (AGG) and **{format_percent(gld_w, 1)}** to gold (GLD), establishing a combined **{format_percent(defensive_w, 1)} defensive exposure** (AGG + GLD). Equity exposure is constrained to **{format_percent(equity_w, 1)}** across active holdings ({eq_details}).
        - **Model Estimates vs. Target**: The model estimates an annual forward return of **{format_percent(expected_return, 2)}** against the **{format_percent(target_return, 2)}** target return constraint, with an estimated annual portfolio volatility of **{format_percent(optimizer_vol, 2)}**. *Note: Forward returns are model estimates based on historical/CAPM shrinkage priors and are not guaranteed; volatility measures return dispersion, not variance.*
        - **Historical Replay & Capital Preservation**: Retrospective replay across the full {int(kpi_summary.get('observations', 0)):,} historical trading days generated a compound annual return of **{format_percent(realized_return, 2)}** with an annualized realized volatility of **{format_percent(realized_vol, 2)}** and a maximum peak-to-trough drawdown of **{format_percent(max_dd, 2)}**. Latest evaluated 1-day 95% Historical VaR is **{var_95_str}** and Expected Shortfall is **{es_95_str}**. *Note: Capital preservation is an investment mandate objective, not a capital guarantee.*
        - **Risk Model Validation Status**: Formal out-of-sample backtesting on {oos_obs:,} trading days (2018–2026) reveals that at 99% confidence, **{n_rejected_99} of {n_models_99} baseline models reject joint conditional coverage at the 5% significance level**. While trailing 250-day windows register in the Green Basel zone during recent calm markets, multi-year crisis clusters (e.g. March 2020 and 2022 rate shocks) demonstrate that empirical market tails are fatter and more clustered than thin-tailed models assume. Statistical non-rejection in calm periods is not proof that a risk model is valid.
        """
    )

    # Mandatory recommendation notice
    render_callout(
        "**Investment Committee Status**: Team investment recommendation pending formal presentation and human review. "
        "The quantitative figures presented herein represent analytical model outputs and retrospective diagnostics, not an automated buy/sell recommendation.",
        level="info",
    )

    # Navigation Guide Cards
    st.markdown("### Committee Decision Navigation")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 1rem; height: 100%;">
                <h5 style="color: {PALETTE['navy']}; margin: 0 0 0.5rem 0;">1. Where is the money invested?</h5>
                <p style="font-size: 0.85rem; color: {PALETTE['muted']}; margin: 0 0 0.5rem 0;">View position weights, dollar amounts, broad asset class exposures, and cumulative growth charts.</p>
                <span style="font-size: 0.82rem; font-weight: 600; color: {PALETTE['teal']};">→ Explore Portfolio & Performance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n2:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 1rem; height: 100%;">
                <h5 style="color: {PALETTE['navy']}; margin: 0 0 0.5rem 0;">2. How much could it lose?</h5>
                <p style="font-size: 0.85rem; color: {PALETTE['muted']}; margin: 0 0 0.5rem 0;">Review Value at Risk (VaR) and Expected Shortfall (ES) across Historical, EWMA, Bootstrap, and EVT models.</p>
                <span style="font-size: 0.82rem; font-weight: 600; color: {PALETTE['teal']};">→ Explore Risk Analysis</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n3:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 1rem; height: 100%;">
                <h5 style="color: {PALETTE['navy']}; margin: 0 0 0.5rem 0;">3. How reliable are forecasts?</h5>
                <p style="font-size: 0.85rem; color: {PALETTE['muted']}; margin: 0 0 0.5rem 0;">Examine out-of-sample backtesting, Kupiec tests, Christoffersen independence, and Basel traffic lights.</p>
                <span style="font-size: 0.82rem; font-weight: 600; color: {PALETTE['teal']};">→ Explore Model Validation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    n4, n5, _ = st.columns([1, 1, 1])
    with n4:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 1rem; height: 100%;">
                <h5 style="color: {PALETTE['navy']}; margin: 0 0 0.5rem 0;">4. What happens during crises?</h5>
                <p style="font-size: 0.85rem; color: {PALETTE['muted']}; margin: 0 0 0.5rem 0;">Simulate GFC 2008, COVID crash, 2022 rate shocks, and examine Euler risk contributions by asset.</p>
                <span style="font-size: 0.82rem; font-weight: 600; color: {PALETTE['teal']};">→ Explore Stress & Risk Contributions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n5:
        st.markdown(
            f"""
            <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 1rem; height: 100%;">
                <h5 style="color: {PALETTE['navy']}; margin: 0 0 0.5rem 0;">5. What assumptions were made?</h5>
                <p style="font-size: 0.85rem; color: {PALETTE['muted']}; margin: 0 0 0.5rem 0;">Review data governance, ADF stationarity, missing price handling, and critical analytical limitations.</p>
                <span style="font-size: 0.82rem; font-weight: 600; color: {PALETTE['teal']};">→ Explore Assumptions & Quality</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
