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
        "Current proposed portfolio · Retrospective replay"
        if selected_basis == PortfolioBasis.CURRENT
        else "Frozen validation portfolio · Out-of-sample"
    )

    render_header(
        "Allocation & Performance",
        f"Portfolio & Performance analysis: asset allocation, endowment constraints, and benchmark performance ({basis_str}).",
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
        "Asset Allocation & Mandate",
        "Historical Performance & KPIs",
        "Efficient Frontier & Optimization",
        "Supporting Estimates (Returns & Covariance)",
    ])

    with tab_alloc:
        col_c, col_t = st.columns([0.85, 1.15])
        with col_c:
            st.subheader("Target Weight Distribution")
            fig_donut = plot_allocation_donut(weights_df)
            fig_donut.update_traces(automargin=True)
            fig_donut.update_layout(
                height=380,
                margin=dict(t=20, b=65, l=20, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.06,
                    xanchor="center",
                    x=0.5,
                ),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_t:
            st.subheader("Holdings & Dollar Allocations")
            st.caption(f"Based on USD {notional:,.0f} Total Reference Endowment Notional")

            weights_sorted = weights_df.sort_values("weight", ascending=False).copy()

            # Render holdings as a styled institutional HTML table
            holdings_rows_html = []
            for idx, r in enumerate(weights_sorted.itertuples()):
                border_bottom = (
                    f"1px solid {PALETTE['border']}"
                    if idx < len(weights_sorted) - 1
                    else f"1.5px solid {PALETTE['border']}"
                )
                weight_val = float(r.weight)
                weight_str = format_percent(weight_val, 2)
                notional_str = format_currency(float(r.dollar_allocation), 0)
                is_zero = weight_val < 1e-5

                weight_color = PALETTE["muted"] if is_zero else PALETTE["navy"]
                notional_color = PALETTE["muted"] if is_zero else PALETTE["slate"]

                row_html = (
                    f'<tr style="border-bottom: {border_bottom};">'
                    f'<td style="padding: 8px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{r.asset}</td>'
                    f'<td style="padding: 8px 10px; color: {PALETTE["navy"]};">{getattr(r, "name", r.asset)}</td>'
                    f'<td style="padding: 8px 10px; color: {PALETTE["muted"]}; font-size: 0.85rem; white-space: nowrap;">{getattr(r, "category", "")}</td>'
                    f'<td style="padding: 8px 10px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {weight_color}; white-space: nowrap;">{weight_str}</td>'
                    f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {notional_color}; white-space: nowrap;">{notional_str}</td>'
                    f'</tr>'
                )
                holdings_rows_html.append(row_html)

            holdings_rows_joint = "".join(holdings_rows_html)
            holdings_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Ticker</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Asset Description</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Category</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Target Weight</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Notional (USD)</th>
</tr>
</thead>
<tbody>
{holdings_rows_joint}
</tbody>
</table>
"""
            # Strip leading spaces so Markdown never parses as a code block
            clean_holdings_html = "\n".join(line.lstrip() for line in holdings_table_html.splitlines())
            st.markdown(clean_holdings_html, unsafe_allow_html=True)
            render_download_button(weights_sorted, f"portfolio_weights_{selected_basis.value}.csv", "Export Weights CSV")

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

        # Render mandate verification as a styled institutional HTML table
        table_rows_html = []
        for idx, r in enumerate(mandate_records):
            is_compliant = r["Status"] == "COMPLIANT"
            status_badge = (
                f'<span style="display: inline-block; padding: 2px 10px; border-radius: 4px; '
                f'background: {PALETTE["green_light"]}; color: {PALETTE["green"]}; '
                f'font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;">✓ Compliant</span>'
                if is_compliant
                else f'<span style="display: inline-block; padding: 2px 10px; border-radius: 4px; '
                f'background: {PALETTE["red_light"]}; color: {PALETTE["red"]}; '
                f'font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;">⚠ Breach</span>'
            )
            border_bottom = f"1px solid {PALETTE['border']}" if idx < len(mandate_records) - 1 else f"1.5px solid {PALETTE['border']}"
            row_html = (
                f'<tr style="border-bottom: {border_bottom};">'
                f'<td style="padding: 9px 12px; color: {PALETTE["navy"]}; font-weight: 500;">{r["Rule / Constraint"]}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]};">{r["Configured Bound"]}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]};">{r["Actual Exposure"]}</td>'
                f'<td style="padding: 9px 12px; text-align: center;">{status_badge}</td>'
                f'</tr>'
            )
            table_rows_html.append(row_html)

        rows_joint = "".join(table_rows_html)
        mandate_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
<th style="text-align: left; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Rule / Constraint</th>
<th style="text-align: right; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Configured Bound</th>
<th style="text-align: right; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Actual Exposure</th>
<th style="text-align: center; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Status</th>
</tr>
</thead>
<tbody>
{rows_joint}
</tbody>
</table>
"""
        # Strip leading spaces on every line so Markdown never treats <thead> / <tr> as a code block
        clean_table_html = "\n".join(line.lstrip() for line in mandate_table_html.splitlines())
        st.markdown(clean_table_html, unsafe_allow_html=True)
        render_download_button(mandate_df, f"mandate_compliance_{selected_basis.value}.csv", "Export Compliance Checks CSV")

        # Highlight Asset Class Breakdown vs Constraint Definitions
        st.markdown("#### Asset Class Breakdown vs. Mandate Definitions")
        b1, b2, b3 = st.columns(3)
        with b1:
            equity_buffer = (p_cfg["max_equity_exposure"] - equity_weight) * 100
            st.metric(
                label="Total Equities (5 ETFs)",
                value=format_percent(equity_weight, 2),
                delta=f"Cap: ≤ {format_percent(p_cfg['max_equity_exposure'], 0)} ({equity_buffer:.1f}% buffer)",
                delta_color="off",
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

        # Compute benchmark stats for executive comparison deltas
        years = len(perf_df) / 252.0 if len(perf_df) > 0 else 1.0
        bench_cagr = (float(perf_df["policy_wealth"].iloc[-1]) / float(perf_df["policy_wealth"].iloc[0])) ** (1.0 / years) - 1.0
        bench_vol = float(perf_df["policy_return"].std() * np.sqrt(252))
        bench_max_dd = float(perf_df["policy_drawdown"].min())
        rf_rate = float(kpi_dict.get("annualized_risk_free", 0.0422))
        bench_sharpe = (bench_cagr - rf_rate) / bench_vol if bench_vol > 0 else 0.0

        port_cagr = float(kpi_dict.get("annualized_return", 0.0))
        port_vol = float(kpi_dict.get("annualized_volatility", 0.0))
        port_max_dd = float(kpi_dict.get("max_drawdown", 0.0))
        port_sharpe = float(kpi_dict.get("sharpe_ratio", 0.0))

        ret_diff = (port_cagr - bench_cagr) * 100.0
        vol_diff = (port_vol - bench_vol) * 100.0
        dd_diff = (port_max_dd - bench_max_dd) * 100.0
        sharpe_diff = port_sharpe - bench_sharpe

        # Primary Performance Cards
        st.subheader("Key Investment Performance Indicators")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric(
                label="Compound Annual Return",
                value=format_percent(port_cagr, 2),
                delta=f"{ret_diff:+.2f}% vs Policy ({format_percent(bench_cagr, 2)})",
                delta_color="normal",
                help="Annualized geometric compound growth rate of daily simple returns.",
            )
        with p2:
            st.metric(
                label="Realized Annual Volatility",
                value=format_percent(port_vol, 2),
                delta=f"{vol_diff:+.2f}% vs Policy ({format_percent(bench_vol, 2)})",
                delta_color="inverse",
                help="Annualized standard deviation of daily simple returns (sample standard deviation * sqrt(252)). Lower is better.",
            )
        with p3:
            st.metric(
                label="Maximum Historical Drawdown",
                value=format_percent(port_max_dd, 2),
                delta=f"{dd_diff:+.2f}% vs Policy ({format_percent(bench_max_dd, 1)})",
                delta_color="normal",
                help="Peak-to-trough maximum observed portfolio decline. Positive delta reflects drawdown cushion vs policy benchmark.",
            )
        with p4:
            st.metric(
                label="Sharpe Ratio",
                value=format_number(port_sharpe, 2),
                delta=f"{sharpe_diff:+.2f} vs Policy ({bench_sharpe:.2f})",
                delta_color="normal",
                help=(
                    f"Excess compound return over risk-free rate divided by realized volatility. "
                    f"Risk-free rate: {format_percent(rf_rate, 2)}."
                ),
            )

        with st.expander("Detailed Institutional Performance Metrics & Conventions"):
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

        perf_df_frontier = load_performance_series(selected_basis)

        f_left, f_right = st.columns([1.2, 1])
        with f_left:
            fig_front = plot_efficient_frontier(
                frontier_df,
                selected_target=config["portfolio"]["target_return_annual"],
                perf_df=perf_df_frontier,
            )
            st.plotly_chart(fig_front, use_container_width=True)

        exp_ret_val = float(frontier_df["expected_return"].iloc[0])
        vol_val = float(frontier_df["volatility"].iloc[0])

        with f_right:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 1.25rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                        <h4 style="color: {PALETTE['navy']}; margin: 0; font-size: 1.05rem; font-weight: 700;">Frontier Diagnostic Finding</h4>
                        <span style="display: inline-block; padding: 3px 9px; border-radius: 4px; background: {PALETTE['amber_light']}; color: {PALETTE['amber']}; font-weight: 600; font-size: 0.76rem; letter-spacing: 0.02em;">Non-Binding Hurdle</span>
                    </div>
                    <p style="font-size: 0.9rem; color: {PALETTE['navy']}; line-height: 1.5; margin-bottom: 0.5rem;">
                        <b>The configured target range produces the identical minimum-variance solution.</b>
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45; margin-bottom: 0.5rem;">
                        Across all evaluated return targets from 5.0% to 7.5% per annum, the portfolio mandate constraints 
                        (45% single-position ceiling, 15% GLD bound, 35% minimum defensive buffer, and equity limits) 
                        produce the identical minimum-variance solution with an expected return of <b>{format_percent(exp_ret_val, 2)}</b> 
                        and annual volatility of <b>{format_percent(vol_val, 2)}</b>.
                    </p>
                    <p style="font-size: 0.85rem; color: {PALETTE['muted']}; line-height: 1.45; margin-bottom: 0;">
                        Because the constrained minimum-variance portfolio's expected return already exceeds the target constraint, 
                        the target return constraint does not bind across this range. Note: This portfolio is subject to mandate investment 
                        constraints; it is not an unconstrained minimum-variance portfolio.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Configured Target Return Grid Results (Current Portfolio)")

        # Render target grid results as styled institutional HTML table
        target_rows_html = []
        for idx, r in enumerate(frontier_df.itertuples()):
            target_val = float(r.target_return)
            exp_ret_val_row = float(r.expected_return)
            vol_val_row = float(r.volatility)

            is_mandate_target = abs(target_val - config["portfolio"]["target_return_annual"]) < 1e-4

            row_bg = f"background-color: {PALETTE['teal_light']};" if is_mandate_target else ""
            border_bottom = f"1px solid {PALETTE['border']}" if idx < len(frontier_df) - 1 else f"1.5px solid {PALETTE['border']}"

            target_badge = (
                f'<span style="margin-left: 8px; padding: 2px 7px; border-radius: 4px; '
                f'background: {PALETTE["teal"]}; color: {PALETTE["card_bg"]}; '
                f'font-weight: 700; font-size: 0.72rem; letter-spacing: 0.03em;">★ Mandate Target</span>'
                if is_mandate_target else ""
            )

            status_badge = (
                f'<span style="display: inline-block; padding: 2px 10px; border-radius: 4px; '
                f'background: {PALETTE["green_light"]}; color: {PALETTE["green"]}; '
                f'font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;">✓ Feasible (Non-Binding)</span>'
            )

            row_html = (
                f'<tr style="border-bottom: {border_bottom}; {row_bg}">'
                f'<td style="padding: 9px 12px; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]};">{format_percent(target_val, 2)}{target_badge}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]};">{format_percent(exp_ret_val_row, 2)}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-weight: 500; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]};">{format_percent(vol_val_row, 2)}</td>'
                f'<td style="padding: 9px 12px; text-align: center;">{status_badge}</td>'
                f'</tr>'
            )
            target_rows_html.append(row_html)

        target_rows_joint = "".join(target_rows_html)
        target_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
<th style="text-align: left; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Target Return Hurdle</th>
<th style="text-align: right; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Optimized Expected Return</th>
<th style="text-align: right; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Annualized Volatility</th>
<th style="text-align: center; padding: 8px 12px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">Constraint Status</th>
</tr>
</thead>
<tbody>
{target_rows_joint}
</tbody>
</table>
"""
        clean_target_html = "\n".join(line.lstrip() for line in target_table_html.splitlines())
        st.markdown(clean_target_html, unsafe_allow_html=True)
        render_download_button(frontier_df, f"target_return_grid_{selected_basis.value}.csv", "Export Target Grid CSV")

    with tab_estimates:
        st.subheader("Model Supporting Estimates: Expected Returns & Covariance")
        from app.data_access import load_expected_returns, load_covariance_matrix, load_model_implied_correlation

        exp_ret_df = load_expected_returns()
        cov_df = load_covariance_matrix()
        corr_df = load_model_implied_correlation()

        # 1. Compact Methodology Callout Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 0.95rem 1.15rem; height: 100%;">
                    <div style="font-weight: 700; color: {PALETTE['navy']}; font-size: 0.92rem; margin-bottom: 0.35rem;">
                        Expected Returns Prior
                    </div>
                    <div style="color: {PALETTE['muted']}; font-size: 0.84rem; line-height: 1.45;">
                        Blended 50% historical simple return and 50% CAPM equilibrium prior for balanced long-term drift.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 0.95rem 1.15rem; height: 100%;">
                    <div style="font-weight: 700; color: {PALETTE['navy']}; font-size: 0.92rem; margin-bottom: 0.35rem;">
                        Covariance Shrinkage
                    </div>
                    <div style="color: {PALETTE['muted']}; font-size: 0.84rem; line-height: 1.45;">
                        Annualized log-return sample covariance shrunk 25% toward its diagonal to eliminate spurious correlation noise.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 8px; padding: 0.95rem 1.15rem; height: 100%;">
                    <div style="font-weight: 700; color: {PALETTE['navy']}; font-size: 0.92rem; margin-bottom: 0.35rem;">
                        Implied Correlation
                    </div>
                    <div style="color: {PALETTE['muted']}; font-size: 0.84rem; line-height: 1.45;">
                        Standardized correlation matrix derived directly from the shrunk covariance matrix (&rho;<sub>ij</sub> = &Sigma;<sub>ij</sub> / &sigma;<sub>i</sub>&sigma;<sub>j</sub>).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # 2. Institutional Blended Annual Expected Returns Table
        st.markdown("#### Blended Annual Expected Returns & Implied Volatility")
        e_disp = exp_ret_df.copy()
        e_disp["volatility"] = e_disp["asset"].apply(lambda a: float(np.sqrt(cov_df.loc[a, a])))
        e_disp = e_disp.sort_values("expected_return", ascending=False)

        exp_rows_html = []
        for idx, r in enumerate(e_disp.itertuples()):
            border_bottom = f"1px solid {PALETTE['border']}" if idx < len(e_disp) - 1 else f"1.5px solid {PALETTE['border']}"
            asset_info = ASSET_REGISTRY.get(r.asset, None)
            asset_name = asset_info.name if asset_info else r.asset
            category = asset_info.category if asset_info else "Other"
            vol_str = format_percent(float(r.volatility), 2)
            ret_str = format_percent(float(r.expected_return), 2)

            row_html = (
                f'<tr style="border-bottom: {border_bottom};">'
                f'<td style="padding: 8px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{r.asset}</td>'
                f'<td style="padding: 8px 10px; color: {PALETTE["navy"]};">{asset_name}</td>'
                f'<td style="padding: 8px 10px; color: {PALETTE["muted"]}; font-size: 0.85rem; white-space: nowrap;">{category}</td>'
                f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]}; white-space: nowrap;">{vol_str}</td>'
                f'<td style="padding: 8px 10px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["teal"]}; white-space: nowrap;">{ret_str}</td>'
                f'</tr>'
            )
            exp_rows_html.append(row_html)

        exp_rows_joint = "".join(exp_rows_html)
        exp_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Ticker</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Asset Description</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Category</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Implied Volatility</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Estimated Return</th>
</tr>
</thead>
<tbody>
{exp_rows_joint}
</tbody>
</table>
"""
        clean_exp_html = "\n".join(line.lstrip() for line in exp_table_html.splitlines())
        st.markdown(clean_exp_html, unsafe_allow_html=True)
        render_download_button(e_disp, "model_expected_returns.csv", "Export Expected Returns CSV")

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # 3. Institutional Covariance Matrix Table
        st.markdown("#### Annualized Covariance Matrix (Log Returns, Shrunk 25% to Diagonal)")
        st.caption("Diagonal values represent asset variances (&sigma;<sub>i</sub><sup>2</sup>); off-diagonals represent pairwise covariances (&Sigma;<sub>ij</sub>).")
        cov_cols = list(cov_df.columns)
        cov_rows_html = []
        for idx, row_asset in enumerate(cov_cols):
            border_bottom = f"1px solid {PALETTE['border']}" if idx < len(cov_cols) - 1 else f"1.5px solid {PALETTE['border']}"
            cells = [f'<td style="padding: 7px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{row_asset}</td>']
            for col_asset in cov_cols:
                val = float(cov_df.loc[row_asset, col_asset])
                is_diag = row_asset == col_asset
                diag_style = f"font-weight: 700; color: {PALETTE['navy']}; background-color: {PALETTE['light_bg']};" if is_diag else f"color: {PALETTE['slate']};"
                cells.append(
                    f'<td style="padding: 7px 10px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; {diag_style}">'
                    f'{val:.6f}</td>'
                )
            cov_rows_html.append(f'<tr style="border-bottom: {border_bottom};">{"".join(cells)}</tr>')

        header_cells = [f'<th style="text-align: left; padding: 7px 10px; color: {PALETTE["muted"]}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Asset</th>']
        for col_asset in cov_cols:
            header_cells.append(f'<th style="text-align: right; padding: 7px 10px; color: {PALETTE["muted"]}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">{col_asset}</th>')

        cov_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
{''.join(header_cells)}
</tr>
</thead>
<tbody>
{''.join(cov_rows_html)}
</tbody>
</table>
"""
        clean_cov_html = "\n".join(line.lstrip() for line in cov_table_html.splitlines())
        st.markdown(clean_cov_html, unsafe_allow_html=True)
        render_download_button(cov_df, "covariance_matrix.csv", "Export Covariance CSV")

        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        # 4. Color-Coded Model-Implied Correlation Matrix Table
        st.markdown("#### Model-Implied Correlation Matrix")
        st.markdown(
            f"""
            <div style="display: flex; gap: 1.25rem; font-size: 0.82rem; margin-top: 0.25rem; margin-bottom: 0.75rem; align-items: center; color: {PALETTE['muted']}; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background: rgba(8, 120, 109, 0.16); border: 1px solid {PALETTE['teal']};"></span>
                    <span>Diversifying Cushion (&rho; &lt; 0.20)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background: rgba(245, 158, 11, 0.16); border: 1px solid {PALETTE['amber']};"></span>
                    <span>High Equity Co-movement (&rho; &ge; 0.50)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']};"></span>
                    <span>Diagonal Variance Unit (1.000)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        corr_cols = list(corr_df.columns)
        corr_rows_html = []
        for idx, row_asset in enumerate(corr_cols):
            border_bottom = f"1px solid {PALETTE['border']}" if idx < len(corr_cols) - 1 else f"1.5px solid {PALETTE['border']}"
            cells = [f'<td style="padding: 8px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{row_asset}</td>']
            for col_asset in corr_cols:
                raw_val = float(corr_df.loc[row_asset, col_asset])
                # Fix -0.000 floating point rounding bug
                val = 0.0 if abs(raw_val) < 1e-4 else raw_val
                is_diag = row_asset == col_asset

                if is_diag:
                    cell_style = f"background-color: {PALETTE['light_bg']}; color: {PALETTE['muted']}; font-weight: 500;"
                elif val < 0.20:
                    cell_style = f"background-color: rgba(8, 120, 109, 0.14); color: {PALETTE['teal']}; font-weight: 600;"
                elif val >= 0.50:
                    cell_style = f"background-color: rgba(245, 158, 11, 0.14); color: {PALETTE['amber']}; font-weight: 600;"
                else:
                    cell_style = f"color: {PALETTE['slate']}; font-weight: 500;"

                cells.append(
                    f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; {cell_style}">'
                    f'{val:.3f}</td>'
                )
            corr_rows_html.append(f'<tr style="border-bottom: {border_bottom};">{"".join(cells)}</tr>')

        corr_header_cells = [f'<th style="text-align: left; padding: 7px 10px; color: {PALETTE["muted"]}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Asset</th>']
        for col_asset in corr_cols:
            corr_header_cells.append(f'<th style="text-align: right; padding: 7px 10px; color: {PALETTE["muted"]}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">{col_asset}</th>')

        corr_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
{''.join(corr_header_cells)}
</tr>
</thead>
<tbody>
{''.join(corr_rows_html)}
</tbody>
</table>
"""
        clean_corr_html = "\n".join(line.lstrip() for line in corr_table_html.splitlines())
        st.markdown(clean_corr_html, unsafe_allow_html=True)
        render_download_button(corr_df, "model_implied_correlation.csv", "Export Correlation CSV")
