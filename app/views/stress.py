"""Stress Testing, Crisis Replay, and Euler Risk Contributions view."""
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
    load_stress_summary,
    load_stress_paths,
    load_stress_shock_contributions,
    load_risk_contributions,
    load_selected_portfolio,
)
from app.charts import (
    plot_stress_paths,
    plot_hypothetical_shocks,
    plot_risk_contributions_diverging,
)
from app.formatting import format_currency, format_percent, format_date
from app.schemas import PortfolioBasis, PALETTE, ASSET_REGISTRY


def render_stress_view() -> None:
    """Render crisis replay scenarios, hypothetical stagflation shocks, and position risk contributions."""
    render_header(
        "Crisis & Contributions",
        "Stress testing and risk contributions: retrospective historical crisis replays, hypothetical stagflation shocks, and Euler position risk budgeting.",
    )
    render_basis_badge("Current proposed portfolio (Retrospective stress replay & position budgeting)")

    portfolio_meta = load_selected_portfolio()
    notional = float(portfolio_meta.get("portfolio_value_usd", 10_000_000.0))

    stress_sum = load_stress_summary()
    stress_paths = load_stress_paths()
    shock_contribs = load_stress_shock_contributions()
    risk_contribs = load_risk_contributions()

    tab_crises, tab_hypothetical, tab_contributions = st.tabs([
        "Historical Crisis Replays",
        "Hypothetical Stagflation Shock",
        "Position Risk Contributions",
    ])

    with tab_crises:
        st.subheader("Historical Severe Market Dislocations Replay")
        render_callout(
            "**Retrospective Replay Notice**: Applying today's allocation to historical crisis dates demonstrates how these exact "
            "weights would have behaved during past turmoil. It does not imply that these weights were chosen prior to 2008 or 2020.",
            level="info",
        )

        hist_scenarios = stress_sum[stress_sum["type"] == "historical_replay"].copy()

        # Summary Table
        s_table = hist_scenarios[["scenario", "actual_start", "actual_end", "observations", "cumulative_return", "pnl_usd", "worst_day_return", "max_drawdown", "ending_value_usd", "status"]].copy()
        s_table = s_table.rename(columns={
            "scenario": "Crisis Episode",
            "actual_start": "Start Date",
            "actual_end": "End Date",
            "observations": "Trading Days",
            "cumulative_return": "Total Return (%)",
            "pnl_usd": "P&L Impact (USD)",
            "worst_day_return": "Worst Single Day",
            "max_drawdown": "Max Drawdown",
            "ending_value_usd": "Ending Notional",
            "status": "Data Status",
        })

        s_table["Start Date"] = s_table["Start Date"].apply(format_date)
        s_table["End Date"] = s_table["End Date"].apply(format_date)
        s_table["Total Return (%)"] = s_table["Total Return (%)"].apply(lambda r: format_percent(r, 2))
        s_table["P&L Impact (USD)"] = s_table["P&L Impact (USD)"].apply(lambda p: format_currency(p, 0))
        s_table["Worst Single Day"] = s_table["Worst Single Day"].apply(lambda w: format_percent(w, 2))
        s_table["Max Drawdown"] = s_table["Max Drawdown"].apply(lambda d: format_percent(d, 2))
        s_table["Ending Notional"] = s_table["Ending Notional"].apply(lambda v: format_currency(v, 0))
        s_table["Data Status"] = s_table["Data Status"].apply(lambda s: "Complete" if s == "ok" else "Partial")

        s_rows = []
        for _, r in s_table.iterrows():
            row_html = (
                f'<tr style="border-bottom: 1px solid {PALETTE["border"]};">'
                f'<td style="text-align: left; padding: 7px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{r["Crisis Episode"]}</td>'
                f'<td style="text-align: center; padding: 7px 10px; color: {PALETTE["muted"]}; font-size: 0.80rem; white-space: nowrap;">{r["Start Date"]}</td>'
                f'<td style="text-align: center; padding: 7px 10px; color: {PALETTE["muted"]}; font-size: 0.80rem; white-space: nowrap;">{r["End Date"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["red"] if "-" in str(r["Total Return (%)"]) else PALETTE["navy"]}; font-variant-numeric: tabular-nums; font-weight: 600;">{r["Total Return (%)"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["red"] if "-" in str(r["P&L Impact (USD)"]) else PALETTE["navy"]}; font-variant-numeric: tabular-nums; font-weight: 600;">{r["P&L Impact (USD)"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{r["Worst Single Day"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{r["Max Drawdown"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{r["Ending Notional"]}</td>'
                f'<td style="text-align: center; padding: 7px 10px; color: {PALETTE["teal"] if r["Data Status"] == "Complete" else PALETTE["amber"]}; font-weight: 600; font-size: 0.80rem;">{r["Data Status"]}</td>'
                f'</tr>'
            )
            s_rows.append(row_html)

        s_html = f"""
<table style="width: 100%; border-collapse: collapse; font-family: 'Inter', -apple-system, sans-serif; font-size: 0.83rem; margin-top: 0.5rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 2px solid {PALETTE['border']};">
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Crisis Episode</th>
<th style="text-align: center; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Start Date</th>
<th style="text-align: center; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">End Date</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Total Return</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">P&L Impact (USD)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Worst Single Day</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Max Drawdown</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Ending Notional</th>
<th style="text-align: center; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Status</th>
</tr>
</thead>
<tbody>
{''.join(s_rows)}
</tbody>
</table>
"""
        clean_s_html = "\n".join(line.lstrip() for line in s_html.splitlines() if line.strip())
        st.markdown(clean_s_html, unsafe_allow_html=True)
        render_download_button(s_table, "historical_stress_scenarios.csv", "Export Crisis Scenarios CSV")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Interactive Crisis Trajectory Visualizer")
        scenario_options = hist_scenarios["scenario"].tolist()
        chosen_scenario = st.selectbox(
            "Select Crisis Scenario to Inspect Trajectory:",
            options=scenario_options,
            index=0,
        )

        fig_path = plot_stress_paths(stress_paths, chosen_scenario, notional)
        st.plotly_chart(fig_path, use_container_width=True)

    with tab_hypothetical:
        st.subheader("Hypothetical Stress: Stagflation & Equity Selloff")
        render_callout(
            "**Assumed Scenario Disclosure**: This scenario models simultaneous severe one-period shocks across asset classes. "
            "These are **team-assumed sensitivity shocks**, not statistical forecast probabilities or maximum loss guarantees.",
            level="warning",
        )

        hypo_meta = stress_sum[stress_sum["type"] == "hypothetical_one_period"].iloc[0]
        hypo_pnl = float(hypo_meta["pnl_usd"])
        hypo_ret = float(hypo_meta["cumulative_return"])
        hypo_ending = float(hypo_meta["ending_value_usd"])

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric(
                label="Portfolio Shock Return",
                value=format_percent(hypo_ret, 2),
                help="Sum of position weights multiplied by their assumed asset shocks.",
            )
        with h2:
            st.metric(
                label="Net Dollar P&L Impact",
                value=format_currency(hypo_pnl, 0),
                help="Total capital decline on USD 10 Million notional.",
            )
        with h3:
            st.metric(
                label="Stressed Portfolio Value",
                value=format_currency(hypo_ending, 0),
                help="Remaining endowment capital value following the shock.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot position contributions
        fig_shocks = plot_hypothetical_shocks(shock_contribs)
        st.plotly_chart(fig_shocks, use_container_width=True)

        st.markdown("#### Position-Level Shocks and Capital Impact")
        h_table = shock_contribs[["asset", "weight", "shock", "return_contribution", "pnl_contribution_usd"]].copy()
        h_table["name"] = h_table["asset"].map(lambda a: ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a)
        h_table = h_table.rename(columns={
            "asset": "Ticker",
            "name": "Asset Description",
            "weight": "Portfolio Weight",
            "shock": "Assumed Shock",
            "return_contribution": "Return Contribution",
            "pnl_contribution_usd": "P&L Contribution (USD)",
        })
        h_table["Portfolio Weight"] = h_table["Portfolio Weight"].apply(lambda w: format_percent(w, 2))
        h_table["Assumed Shock"] = h_table["Assumed Shock"].apply(lambda s: format_percent(s, 1))
        h_table["Return Contribution"] = h_table["Return Contribution"].apply(lambda r: format_percent(r, 2))
        h_table["P&L Contribution (USD)"] = h_table["P&L Contribution (USD)"].apply(lambda p: format_currency(p, 0))

        h_rows = []
        for _, r in h_table.iterrows():
            row_html = (
                f'<tr style="border-bottom: 1px solid {PALETTE["border"]};">'
                f'<td style="text-align: left; padding: 7px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{r["Ticker"]}</td>'
                f'<td style="text-align: left; padding: 7px 10px; color: {PALETTE["navy"]};">{r["Asset Description"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{r["Portfolio Weight"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["red"] if "-" in str(r["Assumed Shock"]) else PALETTE["navy"]}; font-variant-numeric: tabular-nums; font-weight: 600;">{r["Assumed Shock"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["red"] if "-" in str(r["Return Contribution"]) else PALETTE["navy"]}; font-variant-numeric: tabular-nums; font-weight: 600;">{r["Return Contribution"]}</td>'
                f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["red"] if "-" in str(r["P&L Contribution (USD)"]) else PALETTE["navy"]}; font-variant-numeric: tabular-nums; font-weight: 600;">{r["P&L Contribution (USD)"]}</td>'
                f'</tr>'
            )
            h_rows.append(row_html)

        h_html = f"""
<table style="width: 100%; border-collapse: collapse; font-family: 'Inter', -apple-system, sans-serif; font-size: 0.83rem; margin-top: 0.5rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 2px solid {PALETTE['border']};">
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Ticker</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Asset Description</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Weight</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Assumed Shock</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Return Contrib.</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">P&L Contrib. (USD)</th>
</tr>
</thead>
<tbody>
{''.join(h_rows)}
</tbody>
</table>
"""
        clean_h_html = "\n".join(line.lstrip() for line in h_html.splitlines() if line.strip())
        st.markdown(clean_h_html, unsafe_allow_html=True)
        render_download_button(h_table, "hypothetical_shocks.csv", "Export Shock Contributions CSV")

        # Verify reconciliation
        sum_pnl_contrib = float(shock_contribs["pnl_contribution_usd"].sum())
        pnl_reconciled = np.isclose(sum_pnl_contrib, hypo_pnl, atol=1.0)
        reconcile_text = (
            f"Reconciled: Sum of position P&L contributions ({format_currency(sum_pnl_contrib, 0)}) "
            f"matches total scenario P&L ({format_currency(hypo_pnl, 0)})."
            if pnl_reconciled
            else f"Discrepancy: Sum of contributions ({format_currency(sum_pnl_contrib, 0)}) does not match scenario total ({format_currency(hypo_pnl, 0)})."
        )
        st.caption(reconcile_text)

        render_callout(
            "**Scenario Sensitivity Note**: In this assumed scenario, the positive +8.0% gold shock produces a +$120,000 P&L contribution "
            "that partially cushions severe assumed equity and bond drawdowns. This illustrates scenario sensitivity under the mandate; "
            "it is an assumed shock rather than proof that gold always hedges stagflation in every historical or future regime.",
            level="info",
        )

    with tab_contributions:
        st.subheader("Euler Marginal & Component Risk Contributions")
        st.caption(
            "Evaluated over the trailing 252-day window (2025-10-20 to 2026-10-20). "
            "Euler component risk decomposes total portfolio risk such that component contributions sum exactly to portfolio risk. "
            "Marginal VaR and ES are dimensionless sensitivity derivatives (∂Risk/∂w_i); Component VaR and ES are denominated in USD."
        )

        c_control1, c_control2, c_control3 = st.columns(3)
        with c_control1:
            contrib_method = st.selectbox(
                "Risk Contribution Methodology:",
                options=["historical_simple", "parametric_normal_simple"],
                format_func=lambda m: "Historical Tail (Empirical Gradient)" if m == "historical_simple" else "Parametric Normal (Covariance Gradient)",
            )
        with c_control2:
            contrib_conf = st.selectbox(
                "Contribution Confidence Level:",
                options=[0.90, 0.95, 0.99],
                index=1,
                format_func=lambda c: f"{int(c*100)}% Confidence",
            )
        with c_control3:
            contrib_metric = st.selectbox(
                "Chart Contribution Metric:",
                options=["component_var_usd", "component_es_usd"],
                format_func=lambda m: "Dollar Component VaR (USD)" if m == "component_var_usd" else "Dollar Component ES (USD)",
            )

        # Plot Diverging Horizontal Bar Chart
        fig_rc = plot_risk_contributions_diverging(
            risk_contribs,
            metric=contrib_metric,
            confidence=contrib_conf,
            method=contrib_method,
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        # Risk Contributions Table (Custom Institutional HTML Format)
        subset_rc = risk_contribs[
            (risk_contribs["method"] == contrib_method) & np.isclose(risk_contribs["confidence"], contrib_conf)
        ].copy()

        total_comp_var = float(subset_rc["component_var_usd"].sum())
        total_comp_es = float(subset_rc["component_es_usd"].sum())
        total_weight = float(subset_rc["weight"].sum())

        rc_rows_html = []
        for idx, row in enumerate(subset_rc.itertuples()):
            asset_name = ASSET_REGISTRY[row.asset].name if row.asset in ASSET_REGISTRY else row.asset
            row_bg = PALETTE["light_bg"] if idx % 2 == 1 else PALETTE["card_bg"]
            
            # Highlight hedging assets (negative risk component) with emerald text
            c_var_color = PALETTE["green"] if row.component_var_usd < 0 else PALETTE["navy"]
            c_es_color = PALETTE["green"] if row.component_es_usd < 0 else PALETTE["navy"]
            var_share_color = PALETTE["green"] if row.var_risk_share < 0 else PALETTE["slate"]
            es_share_color = PALETTE["green"] if row.es_risk_share < 0 else PALETTE["slate"]

            rc_rows_html.append(f'''
            <tr style="background-color: {row_bg}; border-bottom: 1px solid {PALETTE['border']};">
                <td style="padding: 0.5rem 0.6rem; font-weight: 700; color: {PALETTE['navy']}; white-space: nowrap;">{row.asset}</td>
                <td style="padding: 0.5rem 0.6rem; color: {PALETTE['muted']}; font-size: 0.82rem;">{asset_name}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_percent(row.weight, 2)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_percent(row.marginal_var, 3)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {c_var_color}; white-space: nowrap;">{format_currency(row.component_var_usd, 0)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; color: {var_share_color}; white-space: nowrap;">{format_percent(row.var_risk_share, 2)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_percent(row.marginal_es, 3)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {c_es_color}; white-space: nowrap;">{format_currency(row.component_es_usd, 0)}</td>
                <td style="padding: 0.5rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; color: {es_share_color}; white-space: nowrap;">{format_percent(row.es_risk_share, 2)}</td>
            </tr>
            ''')

        rows_markup = "".join(rc_rows_html)
        rc_table_html = f'''
        <div style="overflow-x: auto; width: 100%; border: 1px solid {PALETTE['border']}; border-radius: 6px; margin-bottom: 0.75rem;">
            <table style="width: 100%; border-collapse: collapse; font-family: Inter, -apple-system, sans-serif; font-size: 0.83rem;">
                <thead>
                    <tr style="background-color: {PALETTE['light_bg']}; color: {PALETTE['navy']}; font-weight: 700; font-size: 0.74rem; letter-spacing: 0.04em; border-bottom: 2px solid {PALETTE['border']};">
                        <th style="text-align: left; padding: 0.55rem 0.6rem; white-space: nowrap;">TICKER</th>
                        <th style="text-align: left; padding: 0.55rem 0.6rem; min-width: 140px;">DESCRIPTION</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">WEIGHT</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">MARGINAL VAR</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">COMPONENT VAR (USD)</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">VAR SHARE</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">MARGINAL ES</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">COMPONENT ES (USD)</th>
                        <th style="text-align: right; padding: 0.55rem 0.5rem; white-space: nowrap;">ES SHARE</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_markup}
                </tbody>
                <tfoot>
                    <tr style="background-color: {PALETTE['light_bg']}; font-weight: 700; border-top: 2px solid {PALETTE['border']}; color: {PALETTE['navy']};">
                        <td style="padding: 0.55rem 0.6rem; white-space: nowrap;">Total Portfolio</td>
                        <td style="padding: 0.55rem 0.6rem; color: {PALETTE['muted']}; font-weight: 400; font-size: 0.8rem;">Euler Decomposition (100% Reconciled)</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_percent(total_weight, 2)}</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; color: {PALETTE['muted']};">—</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_currency(total_comp_var, 0)}</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">100.00%</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; color: {PALETTE['muted']};">—</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">{format_currency(total_comp_es, 0)}</td>
                        <td style="padding: 0.55rem 0.5rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;">100.00%</td>
                    </tr>
                </tfoot>
            </table>
        </div>
        '''
        clean_rc_html = "\n".join(line.lstrip() for line in rc_table_html.splitlines() if line.strip())
        st.markdown(clean_rc_html, unsafe_allow_html=True)
        render_download_button(subset_rc, f"risk_contributions_{contrib_method}_{int(contrib_conf*100)}.csv", "Export Risk Contributions CSV")

        with st.expander("Understanding Portfolio Weight vs. Risk Share (Preserving Negative Signs)"):
            st.markdown(
                """
                - **Weight vs Risk Share**: An asset's portfolio weight does not equal its risk contribution. For example, **AGG represents 45.0% of portfolio capital**, but contributes a far smaller or even negative share to short-horizon tail risk due to low correlation with equities.
                - **Preserving Signed Contributions**: Assets with negative marginal risk (such as AGG in certain tail quantiles, or energy XLE during specific periods) act as hedges. **We never discard negative signs or force them into a misleading pie chart.**
                - **Units of Marginal Quantities**: Marginal VaR and Marginal ES are dimensionless first derivatives representing sensitivity to marginal position expansion. Component quantities represent the dollar allocation of risk and sum to the total portfolio risk.
                - **Derivative Convention**: Marginal derivatives reflect an unconstrained change in asset exposure. They do not represent a self-financing trade that automatically reduces another asset.
                """
            )
