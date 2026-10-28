"""Portfolio construction, asset allocation, and performance view."""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

from app.components import (
    render_header,
    render_basis_badge,
    render_callout,
    render_download_button,
)
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_performance_series,
    load_kpi_summary,
    load_efficient_frontier,
    load_expected_returns,
    load_config,
)
from app.charts import (
    plot_allocation_donut,
    plot_cumulative_wealth,
    plot_drawdowns,
    plot_efficient_frontier,
)
from app.formatting import format_currency, format_percent, format_date, format_number
from app.schemas import PortfolioBasis, PALETTE, ASSET_REGISTRY


def render_portfolio_view(selected_basis: PortfolioBasis = PortfolioBasis.CURRENT) -> None:
    """Render the portfolio allocation and historical performance analysis."""
    basis_str = (
        "Current proposed portfolio (Retrospective replay)"
        if selected_basis == PortfolioBasis.CURRENT
        else "Frozen validation portfolio (Out-of-sample)"
    )

    render_header(
        "Portfolio & Performance",
        f"Asset allocation, endowment constraints, and benchmark performance ({basis_str}).",
    )
    render_basis_badge(basis_str)

    portfolio_meta = load_selected_portfolio()
    weights_df = load_portfolio_weights(selected_basis)
    notional = float(portfolio_meta.get("portfolio_value_usd", 10_000_000.0))
    config = load_config()

    # Broad exposure summary
    equity_assets = [a for a, info in ASSET_REGISTRY.items() if info.category == "Equity"]
    defensive_assets = [a for a, info in ASSET_REGISTRY.items() if info.category == "Fixed Income"]
    real_assets = [a for a, info in ASSET_REGISTRY.items() if info.category == "Real Assets"]

    equity_weight = float(weights_df[weights_df["asset"].isin(equity_assets)]["weight"].sum())
    defensive_weight = float(weights_df[weights_df["asset"].isin(defensive_assets)]["weight"].sum())
    real_weight = float(weights_df[weights_df["asset"].isin(real_assets)]["weight"].sum())

    # Tabs for clean institutional layout
    tab_alloc, tab_perf, tab_frontier = st.tabs([
        "📊 Asset Allocation & Holdings",
        "📈 Historical Performance & KPIs",
        "🎯 Efficient Frontier & Optimization",
    ])

    with tab_alloc:
        col_c, col_t = st.columns([1, 1.2])
        with col_c:
            st.subheader("Target Weight Distribution")
            fig_donut = plot_allocation_donut(weights_df)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_t:
            st.subheader("Holdings & Dollar Allocations")
            st.caption(f"Based on USD {notional:,.0f} Total Reference Endowment Notional")

            display_table = weights_df[["asset", "name", "category", "weight", "dollar_allocation"]].copy()
            display_table = display_table.rename(columns={
                "asset": "Ticker",
                "name": "Asset Description",
                "category": "Category",
                "weight": "Target Weight",
                "dollar_allocation": "Notional Value (USD)",
            })
            display_table["Target Weight"] = display_table["Target Weight"].apply(lambda w: format_percent(w, 2))
            display_table["Notional Value (USD)"] = display_table["Notional Value (USD)"].apply(lambda d: format_currency(d, 0))

            st.dataframe(display_table, use_container_width=True, hide_index=True)
            render_download_button(weights_df, f"portfolio_weights_{selected_basis.value}.csv", "Export Weights CSV")

        st.markdown("---")
        st.subheader("Broad Asset Class Exposure vs Mandate Bounds")
        e_col, d_col, r_col = st.columns(3)
        with e_col:
            st.metric(
                label="Total Equity Exposure",
                value=format_percent(equity_weight, 2),
                delta=f"Max Allowed: {format_percent(config['portfolio']['max_equity_exposure'], 0)}",
                help="Sum of SPY, EFA, EEM, XLE, and XLK weights. Target maximum is 60.0%.",
            )
        with d_col:
            st.metric(
                label="Defensive Fixed Income (AGG)",
                value=format_percent(defensive_weight, 2),
                delta=f"Min Required: {format_percent(config['portfolio']['min_defensive_exposure'], 0)}",
                help="Allocation to US Aggregate Bond ETF (AGG). Target minimum is 35.0%.",
            )
        with r_col:
            st.metric(
                label="Real Assets / Gold (GLD)",
                value=format_percent(real_weight, 2),
                delta=f"Mandate Bounds: 5% – 15%",
                help="Gold exposure provides inflation hedging and crisis diversification. Target bounds 5.0% - 15.0%.",
            )

        with st.expander("ℹ️ Investment Mandate Constraints & Target Return Details"):
            st.markdown(
                """
                - **Target Return (6.50% Annual)**: Configured as an optimization constraint, not a guaranteed return or return forecast.
                - **Capital Preservation Objective**: The optimizer minimizes annualized portfolio variance subject to reaching this required return threshold.
                - **Single Position Limit**: No individual asset may exceed **45.0%** of total portfolio value.
                - **Satellite Asset Caps**: Emerging Markets (EEM) ≤ 10%, Energy (XLE) ≤ 10%, Technology (XLK) ≤ 15%.
                - **Rebalancing Convention**: Daily constant-weight rebalancing without transaction costs.
                """
            )

    with tab_perf:
        st.subheader("Performance Trajectory & Benchmark Replay")
        perf_df = load_performance_series(selected_basis)
        kpi_dict = load_kpi_summary(selected_basis)

        obs_count = int(kpi_dict.get("observations", len(perf_df)))
        start_date = perf_df["date"].min()
        end_date = perf_df["date"].max()

        st.caption(
            f"Observation Window: **{format_date(start_date)}** to **{format_date(end_date)}** ({obs_count:,} trading days). "
            f"Benchmark: Policy Benchmark (35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD)."
        )

        fig_wealth = plot_cumulative_wealth(perf_df, notional)
        st.plotly_chart(fig_wealth, use_container_width=True)

        fig_dd = plot_drawdowns(perf_df)
        st.plotly_chart(fig_dd, use_container_width=True)

        # Primary Performance Cards
        st.subheader("Key Investment Performance Indicators")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric(
                label="Compound Annual Return",
                value=format_percent(float(kpi_dict.get("annualized_return", 0.0)), 2),
                help="Annualized geometric compound growth rate of daily simple returns.",
            )
        with p2:
            st.metric(
                label="Realized Annual Volatility",
                value=format_percent(float(kpi_dict.get("annualized_volatility", 0.0)), 2),
                help="Annualized standard deviation of daily simple returns (sample standard deviation * sqrt(252)).",
            )
        with p3:
            st.metric(
                label="Maximum Historical Drawdown",
                value=format_percent(float(kpi_dict.get("max_drawdown", 0.0)), 2),
                help="Peak-to-trough maximum observed portfolio decline.",
            )
        with p4:
            st.metric(
                label="Sharpe Ratio",
                value=format_number(float(kpi_dict.get("sharpe_ratio", 0.0)), 2),
                help=(
                    f"Excess compound return over risk-free rate divided by realized volatility. "
                    f"Risk-free rate: {format_percent(float(kpi_dict.get('annualized_risk_free', 0.0)), 2)}."
                ),
            )

        with st.expander("📋 Detailed Institutional Performance Metrics & Conventions"):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"**Sortino Ratio**: {format_number(float(kpi_dict.get('sortino_ratio', 0.0)), 2)}")
                st.caption("Penalizes downside volatility below daily risk-free threshold only.")
                st.markdown(f"**Calmar Ratio**: {format_number(float(kpi_dict.get('calmar_ratio', 0.0)), 2)}")
                st.caption("Annualized compound return divided by absolute maximum drawdown.")
            with d2:
                st.markdown(f"**Portfolio Beta**: {format_number(float(kpi_dict.get('beta', 0.0)), 2)}")
                st.caption("Sensitivity relative to market proxy returns (70% SPY, 20% EFA, 10% EEM).")
                st.markdown(f"**Jensen's Alpha**: {format_percent(float(kpi_dict.get('jensens_alpha', 0.0)), 2)}")
                st.caption("Annualized abnormal return beyond CAPM market beta expectation.")
            with d3:
                st.markdown(f"**Information Ratio**: {format_number(float(kpi_dict.get('information_ratio', 0.0)), 2)}")
                st.caption("Mean active return over policy benchmark divided by annual tracking error.")
                st.markdown(f"**Treynor Ratio**: {format_percent(float(kpi_dict.get('treynor_ratio', 0.0)), 2)}")
                st.caption("Excess return per unit of systematic market risk (beta).")

            st.markdown("---")
            if selected_basis == PortfolioBasis.CURRENT:
                render_callout(
                    "**Convention Disclosure**: Current-allocation retrospective KPIs use the latest available 3-month Treasury quote "
                    f"({format_percent(float(kpi_dict.get('annualized_risk_free', 0.0422)), 2)}) as a constant risk-free reference. "
                    "The backend Sharpe calculation utilizes a compound-return numerator convention.",
                    level="info",
                )
            else:
                render_callout(
                    "**Convention Disclosure**: Out-of-sample (OOS) validation KPIs use an as-of daily FRED DGS3MO historical risk-free series "
                    f"(average {format_percent(float(kpi_dict.get('annualized_risk_free', 0.0274)), 2)}), providing a time-varying benchmark.",
                    level="info",
                )

    with tab_frontier:
        st.subheader("Efficient Frontier Analysis & Target Feasibility")
        frontier_df = load_efficient_frontier()

        f_left, f_right = st.columns([1.2, 1])
        with f_left:
            fig_front = plot_efficient_frontier(frontier_df, selected_target=config["portfolio"]["target_return_annual"])
            st.plotly_chart(fig_front, use_container_width=True)

        with f_right:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 1.25rem;">
                    <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.5rem;">Frontier Diagnostic Finding</h4>
                    <p style="font-size: 0.9rem; color: {PALETTE['navy']}; line-height: 1.5;">
                        <b>The configured target range produces the same minimum-variance solution.</b>
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45;">
                        Across all evaluated return targets from 5.0% to 7.5% per annum, the portfolio constraints (notably the 45% AGG ceiling, 15% GLD bound, and equity sub-limits) produce the identical minimum-variance solution with an expected return of <b>7.66%</b> and annual volatility of <b>8.57%</b>.
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45;">
                        Because the unconstrained minimum-variance portfolio's return already satisfies the target constraint, higher target returns within this range do not force additional variance-increasing equity risk.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Configured Target Return Grid Results")
        f_display = frontier_df.copy()
        f_display["target_return"] = f_display["target_return"].apply(lambda r: format_percent(r, 2))
        f_display["expected_return"] = f_display["expected_return"].apply(lambda r: format_percent(r, 2))
        f_display["volatility"] = f_display["volatility"].apply(lambda v: format_percent(v, 2))
        st.dataframe(f_display, use_container_width=True, hide_index=True)
