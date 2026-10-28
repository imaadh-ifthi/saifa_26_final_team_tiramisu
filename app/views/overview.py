"""Executive Overview page for the Serendib University Endowment Fund."""
from __future__ import annotations
import streamlit as st
import pandas as pd

from app.components import render_header, render_callout, render_basis_badge
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_kpi_summary,
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

    # What these results mean section
    st.subheader("What These Results Mean for the Endowment")
    st.markdown(
        """
        - **Asset Allocation Profile**: The proposed portfolio allocates **45.0%** to investment-grade fixed income (AGG) and **15.0%** to gold (GLD), establishing a **60.0% defensive and real-asset buffer**. The remaining **40.0%** is invested across equities (SPY 22.8%, EFA 9.6%, XLK 6.3%, XLE 1.3%).
        - **Return vs. Target**: The portfolio achieves an estimated forward return of **7.66%**, meeting and exceeding the university's 6.50% return requirement, while minimizing annual portfolio variance to **8.57%**.
        - **Risk & Capital Preservation**: Retrospective 1-day Value at Risk (95%) stands at **$96,938 (0.97%)**, and Expected Shortfall at **$133,231 (1.33%)**. In historical crisis replays, the portfolio experienced a maximum drawdown of **-24.8%** during the 2008 Financial Crisis, recovering faster than pure equity allocations.
        - **Model Validation Findings**: Formal out-of-sample testing on 2,212 post-2018 trading days shows that while recent 250-day windows earn a **Green Basel traffic light**, statistical tests over the full multi-year horizon reject conditional coverage at 99%, demonstrating that extreme tails remain fatter than thin-tailed parametric models predict.
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
