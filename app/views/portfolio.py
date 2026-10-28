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
    fixed_income_assets = [a for a, info in ASSET_REGISTRY.items() if info.category == "Fixed Income"]
    real_assets = [a for a, info in ASSET_REGISTRY.items() if info.category == "Real Assets"]

    equity_weight = float(weights_df[weights_df["asset"].isin(equity_assets)]["weight"].sum())
    fixed_income_weight = float(weights_df[weights_df["asset"].isin(fixed_income_assets)]["weight"].sum())
    gold_weight = float(weights_df[weights_df["asset"] == "GLD"]["weight"].sum()) if "GLD" in weights_df["asset"].values else 0.0
    
    # Defensive exposure is explicitly defined by the backend optimizer as AGG + GLD
    defensive_exposure = fixed_income_weight + gold_weight

    # Individual position caps
    p_cfg = config["portfolio"]
    single_pos_max = float(weights_df["weight"].max())
    single_pos_asset = weights_df.loc[weights_df["weight"].idxmax(), "asset"]
    eem_w = float(weights_df.loc[weights_df["asset"] == "EEM", "weight"].sum()) if "EEM" in weights_df["asset"].values else 0.0
    xle_w = float(weights_df.loc[weights_df["asset"] == "XLE", "weight"].sum()) if "XLE" in weights_df["asset"].values else 0.0
    xlk_w = float(weights_df.loc[weights_df["asset"] == "XLK", "weight"].sum()) if "XLK" in weights_df["asset"].values else 0.0

    # Tabs for clean institutional layout
    tab_alloc, tab_perf, tab_frontier, tab_estimates = st.tabs([
        "📊 Asset Allocation & Mandate",
        "📈 Historical Performance & KPIs",
        "🎯 Efficient Frontier & Optimization",
        "📐 Supporting Estimates (Returns & Covariance)",
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
        st.subheader("Endowment Mandate Compliance Checks")
        st.caption(
            "Evaluation against configured institutional rules. Note: The analytical backend defines defensive exposure "
            "as **AGG + GLD** (Fixed Income + Gold Bullion). Floating-point tolerance: 1e-5."
        )

        # Build Mandate Verification Table
        mandate_records = [
            {
                "Rule / Constraint": "Total Equity Exposure (SPY+EFA+EEM+XLE+XLK)",
                "Configured Bound": f"≤ {format_percent(p_cfg['max_equity_exposure'], 0)}",
                "Actual Exposure": format_percent(equity_weight, 2),
                "Status": "COMPLIANT" if equity_weight <= p_cfg["max_equity_exposure"] + 1e-5 else "BREACH",
            },
            {
                "Rule / Constraint": "Defensive Minimum (AGG + GLD)",
                "Configured Bound": f"≥ {format_percent(p_cfg['min_defensive_exposure'], 0)}",
                "Actual Exposure": format_percent(defensive_exposure, 2),
                "Status": "COMPLIANT" if defensive_exposure >= p_cfg["min_defensive_exposure"] - 1e-5 else "BREACH",
            },
            {
                "Rule / Constraint": f"Single-Position Maximum (Largest: {single_pos_asset})",
                "Configured Bound": f"≤ {format_percent(p_cfg['max_single_position'], 0)}",
                "Actual Exposure": format_percent(single_pos_max, 2),
                "Status": "COMPLIANT" if single_pos_max <= p_cfg["max_single_position"] + 1e-5 else "BREACH",
            },
            {
                "Rule / Constraint": "Gold Bullion Bounds (GLD)",
                "Configured Bound": f"{format_percent(p_cfg['gld_bounds'][0], 0)} – {format_percent(p_cfg['gld_bounds'][1], 0)}",
                "Actual Exposure": format_percent(gold_weight, 2),
                "Status": "COMPLIANT" if (gold_weight >= p_cfg["gld_bounds"][0] - 1e-5 and gold_weight <= p_cfg["gld_bounds"][1] + 1e-5) else "BREACH",
            },
            {
                "Rule / Constraint": "Emerging Markets Equity Cap (EEM)",
                "Configured Bound": f"≤ {format_percent(p_cfg['max_eem'], 0)}",
                "Actual Exposure": format_percent(eem_w, 2),
                "Status": "COMPLIANT" if eem_w <= p_cfg["max_eem"] + 1e-5 else "BREACH",
            },
            {
                "Rule / Constraint": "Energy Sector Equity Cap (XLE)",
                "Configured Bound": f"≤ {format_percent(p_cfg['max_xle'], 0)}",
                "Actual Exposure": format_percent(xle_w, 2),
                "Status": "COMPLIANT" if xle_w <= p_cfg["max_xle"] + 1e-5 else "BREACH",
            },
            {
                "Rule / Constraint": "Technology Sector Equity Cap (XLK)",
                "Configured Bound": f"≤ {format_percent(p_cfg['max_xlk'], 0)}",
                "Actual Exposure": format_percent(xlk_w, 2),
                "Status": "COMPLIANT" if xlk_w <= p_cfg["max_xlk"] + 1e-5 else "BREACH",
            },
        ]
        mandate_df = pd.DataFrame(mandate_records)
        st.dataframe(mandate_df, use_container_width=True, hide_index=True)

        # Highlight Asset Class Breakdown vs Constraint Definitions
        st.markdown("#### Asset Class Breakdown vs. Mandate Definitions")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric(
                label="Total Equities (5 ETFs)",
                value=format_percent(equity_weight, 2),
                help="Sum of SPY, EFA, EEM, XLE, and XLK weights.",
            )
        with b2:
            st.metric(
                label="Defensive Exposure (AGG + GLD)",
                value=format_percent(defensive_exposure, 2),
                delta=f"AGG: {format_percent(fixed_income_weight, 1)} | GLD: {format_percent(gold_weight, 1)}",
                help="Combined fixed income and gold holdings fulfilling the endowment defensive mandate.",
            )
        with b3:
            st.metric(
                label="Single Max Position",
                value=format_percent(single_pos_max, 2),
                delta=f"Asset: {single_pos_asset} (Cap: {format_percent(p_cfg['max_single_position'], 0)})",
                help="Largest single asset weight in the portfolio.",
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
            f"Benchmark: Policy Benchmark (35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD — 50% equities, 40% bonds, 10% gold)."
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

        if selected_basis != PortfolioBasis.CURRENT:
            render_callout(
                "**Scope Notice**: The efficient frontier and target return grid below describe the **Current Proposed Portfolio** "
                "(fitted on the full 2007–2026 estimation sample). It is distinct from the frozen 2007–2017 validation allocation.",
                level="info",
            )

        f_left, f_right = st.columns([1.2, 1])
        with f_left:
            fig_front = plot_efficient_frontier(frontier_df, selected_target=config["portfolio"]["target_return_annual"])
            st.plotly_chart(fig_front, use_container_width=True)

        exp_ret_val = float(frontier_df["expected_return"].iloc[0])
        vol_val = float(frontier_df["volatility"].iloc[0])

        with f_right:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 1.25rem;">
                    <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.5rem;">Frontier Diagnostic Finding</h4>
                    <p style="font-size: 0.9rem; color: {PALETTE['navy']}; line-height: 1.5;">
                        <b>The configured target range produces the identical minimum-variance solution.</b>
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45;">
                        Across all evaluated return targets from 5.0% to 7.5% per annum, the portfolio mandate constraints 
                        (45% single-position ceiling, 15% GLD bound, 35% minimum defensive buffer, and equity limits) 
                        produce the identical minimum-variance solution with an expected return of <b>{format_percent(exp_ret_val, 2)}</b> 
                        and annual volatility of <b>{format_percent(vol_val, 2)}</b>.
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45;">
                        Because the constrained minimum-variance portfolio's expected return already exceeds the target constraint, 
                        the target return constraint does not bind across this range. Note: This portfolio is subject to mandate investment 
                        constraints; it is not an unconstrained minimum-variance portfolio.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Configured Target Return Grid Results (Current Portfolio)")
        f_display = frontier_df.copy()
        f_display["target_return"] = f_display["target_return"].apply(lambda r: format_percent(r, 2))
        f_display["expected_return"] = f_display["expected_return"].apply(lambda r: format_percent(r, 2))
        f_display["volatility"] = f_display["volatility"].apply(lambda v: format_percent(v, 2))
        st.dataframe(f_display, use_container_width=True, hide_index=True)

    with tab_estimates:
        st.subheader("Model Supporting Estimates: Expected Returns & Covariance")
        from app.data_access import load_expected_returns, load_covariance_matrix, load_model_implied_correlation

        exp_ret_df = load_expected_returns()
        cov_df = load_covariance_matrix()
        corr_df = load_model_implied_correlation()

        st.markdown(
            """
            - **Expected Returns**: Blended 50% historical simple return and 50% CAPM equilibrium prior.
            - **Covariance Matrix**: Annualized log-return sample covariance shrunk 25% toward its diagonal.
            - **Correlation Matrix**: Model-implied correlation derived from the shrunk covariance matrix.
            """
        )

        st.markdown("#### Blended Annual Expected Returns (Model Estimates)")
        e_disp = exp_ret_df.copy()
        e_disp["Asset Name"] = e_disp["asset"].map(lambda a: ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a)
        e_disp["Category"] = e_disp["asset"].map(lambda a: ASSET_REGISTRY[a].category if a in ASSET_REGISTRY else "Other")
        e_disp["Estimated Return"] = e_disp["expected_return"].apply(lambda r: format_percent(r, 2))
        st.dataframe(e_disp[["asset", "Asset Name", "Category", "Estimated Return"]].rename(columns={"asset": "Ticker"}), use_container_width=True, hide_index=True)

        st.markdown("#### Annualized Covariance Matrix (Log Returns, Shrunk 25% to Diagonal)")
        st.dataframe(cov_df.style.format("{:.6f}"), use_container_width=True)

        st.markdown("#### Model-Implied Correlation Matrix")
        st.dataframe(corr_df.style.format("{:.3f}"), use_container_width=True)
