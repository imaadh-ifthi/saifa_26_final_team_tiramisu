"""Risk Analysis view: Value at Risk (VaR) and Expected Shortfall (ES)."""
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
    load_var_summary,
    load_var_forecasts,
    load_selected_portfolio,
    load_evt_diagnostics,
)
from app.charts import (
    plot_var_es_time_series,
    plot_var_method_comparison_bar,
)
from app.formatting import format_currency, format_percent, format_date
from app.schemas import PortfolioBasis, PALETTE, METHOD_LABELS, METHOD_DESCRIPTIONS


def render_risk_view(
    selected_basis: PortfolioBasis = PortfolioBasis.CURRENT,
    selected_confidence: float = 0.95,
    selected_method: str = "historical",
) -> None:
    """Render the risk analysis dashboard."""
    basis_str = (
        "Current proposed portfolio (Retrospective replay)"
        if selected_basis == PortfolioBasis.CURRENT
        else "Frozen validation portfolio (Out-of-sample)"
    )

    render_header(
        "Risk Explorer",
        "Comprehensive Risk Analysis: Probabilistic 1-day loss thresholds (VaR) and expected loss beyond the threshold (Expected Shortfall).",
    )
    render_basis_badge(basis_str)

    # Core concept definition banner
    st.markdown(
        f"""
        <div style="background-color: {PALETTE['light_bg']}; border-left: 3px solid {PALETTE['teal']}; border-radius: 0 6px 6px 0; padding: 0.65rem 0.95rem; margin-bottom: 1rem;">
            <div style="font-size: 0.90rem; font-weight: 700; color: {PALETTE['navy']}; margin-bottom: 0.2rem;">
                Core Risk Metric Distinction:
            </div>
            <div style="font-size: 0.88rem; color: {PALETTE['navy']}; line-height: 1.45;">
                <b>Value at Risk (VaR)</b> estimates a loss threshold: a loss this large occurs with probability <i>1 - c</i>.<br>
                <b>Expected Shortfall (ES / CVaR)</b> estimates the <i>average loss</i> conditional on exceeding that threshold.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    portfolio_meta = load_selected_portfolio()
    notional = float(portfolio_meta.get("portfolio_value_usd", 10_000_000.0))

    summary_df = load_var_summary(selected_basis)
    forecasts_df = load_var_forecasts(selected_basis)

    available_methods = summary_df["method"].unique().tolist()
    latest_common_date = summary_df["latest_date"].max()

    # Metric Cards for currently selected method and confidence
    matching_summary = summary_df[
        (summary_df["method"] == selected_method) & np.isclose(summary_df["confidence"], selected_confidence)
    ]

    if not matching_summary.empty:
        curr_row = matching_summary.iloc[0]
        v_pct = float(curr_row["latest_var_pct"]) if pd.notna(curr_row.get("latest_var_pct")) else None
        v_usd = float(curr_row["latest_var_usd"]) if pd.notna(curr_row.get("latest_var_usd")) else None
        e_pct = float(curr_row["latest_es_pct"]) if pd.notna(curr_row.get("latest_es_pct")) else None
        e_usd = float(curr_row["latest_es_usd"]) if pd.notna(curr_row.get("latest_es_usd")) else None
        eval_date = curr_row["latest_date"]
    else:
        v_pct = v_usd = e_pct = e_usd = None
        eval_date = latest_common_date

    st.markdown(f'<h3 style="color: {PALETTE["navy"]}; font-size: 1.40rem; font-weight: 700; margin: 0.5rem 0 0.15rem 0;">Latest Evaluated 1-Day Loss Forecasts ({format_date(eval_date)})</h3>', unsafe_allow_html=True)
    st.caption(
        f"Evaluated forecast for trading day **{format_date(eval_date)}**, estimated using market information through the preceding close. "
        f"Horizon: 1 Trading Day | Reference Notional: {format_currency(notional, 0)}. *Note: This is the latest evaluated forecast in the saved run; it is not a live forecast for tomorrow.*"
    )

    # Compute context deltas for the top 4 KPI cards
    method_label = METHOD_LABELS.get(selected_method, selected_method)
    tail_diff_str = f"+{format_currency(e_usd - v_usd, 0)} Tail Severity vs VaR" if (e_usd is not None and v_usd is not None) else "Tail Severity vs VaR"
    ratio_str = f"ES / VaR Ratio: {e_usd / v_usd:.2f}x" if (e_usd is not None and v_usd is not None and v_usd > 0) else "Tail Expectation"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label=f"1-Day VaR ({int(selected_confidence*100)}% Conf.)",
            value=format_currency(v_usd, 0) if v_usd is not None else "Unavailable",
            delta=f"Method: {method_label}",
            delta_color="off",
            delta_arrow="off",
            help="Dollar loss threshold expected to be exceeded on only (1 - c) of trading days under model assumptions.",
        )
    with m2:
        st.metric(
            label="VaR (% of Portfolio)",
            value=format_percent(v_pct, 2) if v_pct is not None else "Unavailable",
            delta=f"1-Day {int(selected_confidence*100)}% Tail Threshold",
            delta_color="off",
            delta_arrow="off",
            help="Estimated loss quantile as a percentage of total portfolio value. VaR is an estimated quantile, not a guaranteed maximum loss.",
        )
    with m3:
        st.metric(
            label=f"Expected Shortfall ({int(selected_confidence*100)}%)",
            value=format_currency(e_usd, 0) if e_usd is not None else "Unavailable",
            delta=tail_diff_str,
            delta_color="off",
            help="Expected loss conditional on exceeding the VaR threshold (average tail loss).",
        )
    with m4:
        st.metric(
            label="ES (% of Portfolio)",
            value=format_percent(e_pct, 2) if e_pct is not None else "Unavailable",
            delta=ratio_str,
            delta_color="off",
            delta_arrow="off",
            help="Expected loss percentage of total portfolio value within the extreme tail.",
        )

    # Method comparison table at current confidence level
    st.markdown(f'<h3 style="color: {PALETTE["navy"]}; font-size: 1.40rem; font-weight: 700; margin: 1.1rem 0 0.2rem 0;">Methodology Comparison Table ({int(selected_confidence*100)}% Confidence)</h3>', unsafe_allow_html=True)

    comp_df = summary_df[np.isclose(summary_df["confidence"], selected_confidence)].copy()

    # Render methodology comparison as custom institutional HTML table
    comp_rows_html = []
    for idx, r in enumerate(comp_df.itertuples()):
        is_active = r.method == selected_method
        method_name = METHOD_LABELS.get(r.method, r.method)
        row_bg = f"background-color: {PALETTE['teal_light']};" if is_active else ""
        border_bottom = f"1px solid {PALETTE['border']}" if idx < len(comp_df) - 1 else f"1.5px solid {PALETTE['border']}"

        active_badge = (
            f'<span style="margin-left: 8px; padding: 2px 7px; border-radius: 4px; '
            f'background: {PALETTE["teal"]}; color: {PALETTE["card_bg"]}; '
            f'font-weight: 700; font-size: 0.72rem; letter-spacing: 0.03em;">★ Active Selection</span>'
            if is_active else ""
        )

        var_pct_str = format_percent(float(r.latest_var_pct), 2) if pd.notna(r.latest_var_pct) else "Unavailable"
        var_usd_str = format_currency(float(r.latest_var_usd), 0) if pd.notna(r.latest_var_usd) else "Unavailable"
        es_pct_str = format_percent(float(r.latest_es_pct), 2) if pd.notna(r.latest_es_pct) else "Unavailable"
        es_usd_str = format_currency(float(r.latest_es_usd), 0) if pd.notna(r.latest_es_usd) else "Unavailable"

        if pd.notna(r.number_of_exceptions) and pd.notna(r.observations) and float(r.observations) > 0:
            exc_count = int(r.number_of_exceptions)
            exc_rate = (exc_count / float(r.observations)) * 100.0
            exc_str = f"{exc_count:,} ({exc_rate:.2f}%)"
        elif pd.notna(r.number_of_exceptions):
            exc_str = f"{int(r.number_of_exceptions):,}"
        else:
            exc_str = "Unavailable"

        obs_str = f"{int(r.observations):,}" if pd.notna(r.observations) else "Unavailable"

        row_html = (
            f'<tr style="border-bottom: {border_bottom}; {row_bg}">'
            f'<td style="padding: 8px 10px; font-weight: 600; color: {PALETTE["navy"]}; white-space: nowrap;">{method_name}{active_badge}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]}; white-space: nowrap;">{var_pct_str}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]}; white-space: nowrap;">{var_usd_str}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]}; white-space: nowrap;">{es_pct_str}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]}; white-space: nowrap;">{es_usd_str}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["navy"]}; white-space: nowrap;">{exc_str}</td>'
            f'<td style="padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; color: {PALETTE["slate"]}; white-space: nowrap;">{obs_str}</td>'
            f'</tr>'
        )
        comp_rows_html.append(row_html)

    comp_rows_joint = "".join(comp_rows_html)
    comp_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 0.75rem;">
<thead>
<tr style="border-bottom: 1.5px solid {PALETTE['border']};">
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Methodology</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">1-Day VaR (%)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">1-Day VaR (USD)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">1-Day ES (%)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">1-Day ES (USD)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Exceptions (Rate)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Sample Days</th>
</tr>
</thead>
<tbody>
{comp_rows_joint}
</tbody>
</table>
"""
    clean_comp_html = "\n".join(line.lstrip() for line in comp_table_html.splitlines())
    st.markdown(clean_comp_html, unsafe_allow_html=True)
    render_download_button(comp_df, f"var_methodology_comparison_{int(selected_confidence*100)}_{selected_basis.value}.csv", "Export Methodology Comparison CSV")

    # Disclosure regarding EVT availability on Current vs Validation portfolio
    if selected_basis == PortfolioBasis.CURRENT:
        render_callout(
            "**EVT Model Scope Notice**: Extreme Value Theory (EVT-GPD) is calibrated strictly for the "
            "**Frozen validation portfolio** across 1,000-day rolling evaluation windows. "
            "To view EVT tail fits alongside baseline methods, select the **'Frozen validation portfolio'** basis in the sidebar.",
            level="info",
        )

    # Visualizations: Bar comparison & Time Series
    tab_chart_time, tab_chart_bar, tab_evt = st.tabs([
        "Risk Forecasts Over Time",
        "Cross-Method Bar Comparison",
        "EVT Threshold Diagnostics",
    ])

    with tab_chart_time:
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Daily VaR & ES Evolution — {METHOD_LABELS.get(selected_method, selected_method)}</h4>', unsafe_allow_html=True)
        if selected_method in available_methods:
            fig_ts = plot_var_es_time_series(forecasts_df, selected_method, selected_confidence)
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning(f"Selected method '{selected_method}' is not available in the {selected_basis.value} dataset.")

    with tab_chart_bar:
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Cross-Methodology Loss Comparison ({format_date(latest_common_date)})</h4>', unsafe_allow_html=True)
        # Ensure all methods share the latest evaluation date
        shared_date_mask = summary_df["latest_date"] == latest_common_date
        if shared_date_mask.all():
            fig_bar = plot_var_method_comparison_bar(summary_df, selected_confidence)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption(f"Comparing methods aligned on common date **{format_date(latest_common_date)}**.")
            aligned_summary = summary_df[shared_date_mask].copy()
            fig_bar = plot_var_method_comparison_bar(aligned_summary, selected_confidence)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_evt:
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Extreme Value Theory (EVT-GPD) Tail Diagnostics</h4>', unsafe_allow_html=True)
        render_callout(
            "**Portfolio Basis Disclosure**: EVT-GPD is calibrated on the **Frozen Validation Portfolio** "
            "(2018–2026 forward evaluation). Diagnostics reflect rolling 1,000-day standard-loss residual fits.",
            level="info",
        )
        evt_df = load_evt_diagnostics()
        if not evt_df.empty:
            st.markdown(
                """
                EVT fits a Generalized Pareto Distribution (GPD) to positive tail loss exceedances above an eligible threshold:
                - **Refit Frequency**: Refit every 21 trading days over trailing 1,000 observations.
                - **Candidate Thresholds**: Evaluated across 85th, 90th, 92.5th, and 95th percentiles.
                - **Selection Rule**: Lowest stable threshold based on shape and scale stability.
                """
            )
            # Show latest refit summary
            latest_refit_date = evt_df["fit_date"].max()
            latest_refit = evt_df[evt_df["fit_date"] == latest_refit_date].copy()

            st.markdown(f"**Latest Refit Diagnostics (Date: {format_date(latest_refit_date)}):**")
            avail_cols = [c for c in ["threshold_quantile", "threshold_log", "exceedances", "shape", "scale", "modified_scale", "stable_with_next", "selection_status", "selected"] if c in latest_refit.columns]
            d_display = latest_refit[avail_cols].drop_duplicates().copy()

            evt_rows = []
            for _, r in d_display.iterrows():
                is_sel = bool(r.get("selected", False))
                is_stable = bool(r.get("stable_with_next", False))
                row_bg = PALETTE["teal_light"] if is_sel else "transparent"
                border_b = f"1px solid {PALETTE['border']}"

                q_val = f"{float(r['threshold_quantile'])*100:.1f}%"
                u_val = f"{float(r['threshold_log']):.4f}"
                exc_val = f"{int(r['exceedances'])}"
                shape_val = f"{float(r['shape']):.4f}"
                scale_val = f"{float(r['scale']):.4f}"
                mod_scale_val = f"{float(r['modified_scale']):.4f}"

                stable_badge = f'<span style="color: {PALETTE["teal"]}; font-weight: 700;">✓</span>' if is_stable else f'<span style="color: {PALETTE["muted"]};">—</span>'
                status_raw = str(r.get("selection_status", ""))
                status_val = "Stable adjacent thresholds" if "stable" in status_raw.lower() else status_raw.replace("_", " ").capitalize()

                sel_badge = (
                    f'<span style="display: inline-block; padding: 2px 7px; border-radius: 4px; background: {PALETTE["teal_light"]}; '
                    f'color: {PALETTE["teal"]}; font-weight: 700; font-size: 0.76rem; border: 1px solid {PALETTE["teal"]};">★ Selected</span>'
                    if is_sel else f'<span style="color: {PALETTE["muted"]};">—</span>'
                )

                row_html = (
                    f'<tr style="background-color: {row_bg}; border-bottom: {border_b};">'
                    f'<td style="text-align: right; padding: 7px 10px; font-weight: 600; color: {PALETTE["navy"]};">{q_val}</td>'
                    f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{u_val}</td>'
                    f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{exc_val}</td>'
                    f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{shape_val}</td>'
                    f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{scale_val}</td>'
                    f'<td style="text-align: right; padding: 7px 10px; color: {PALETTE["navy"]}; font-variant-numeric: tabular-nums;">{mod_scale_val}</td>'
                    f'<td style="text-align: center; padding: 7px 10px;">{stable_badge}</td>'
                    f'<td style="text-align: left; padding: 7px 10px; color: {PALETTE["navy"]}; font-size: 0.82rem;">{status_val}</td>'
                    f'<td style="text-align: center; padding: 7px 10px;">{sel_badge}</td>'
                    f'</tr>'
                )
                evt_rows.append(row_html)

            evt_table_html = f"""
<table style="width: 100%; border-collapse: collapse; font-family: 'Inter', -apple-system, sans-serif; font-size: 0.83rem; margin-top: 0.5rem; margin-bottom: 0.85rem;">
<thead>
<tr style="border-bottom: 2px solid {PALETTE['border']};">
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Quantile</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Log Threshold (u)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Exceedances</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Shape (ξ)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Scale (β)</th>
<th style="text-align: right; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Mod. Scale</th>
<th style="text-align: center; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Stable?</th>
<th style="text-align: left; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Selection Status</th>
<th style="text-align: center; padding: 7px 10px; color: {PALETTE['muted']}; font-weight: 600; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;">Selected?</th>
</tr>
</thead>
<tbody>
{''.join(evt_rows)}
</tbody>
</table>
"""
            clean_evt_html = "\n".join(line.lstrip() for line in evt_table_html.splitlines() if line.strip())
            st.markdown(clean_evt_html, unsafe_allow_html=True)
            render_download_button(d_display, f"evt_threshold_diagnostics_{latest_refit_date}.csv", "Export EVT Diagnostics CSV")

            # Count distinct refit dates with warning status
            total_refit_dates = int(evt_df["fit_date"].nunique())
            warning_refit_dates = int(evt_df[evt_df["selection_status"] == "warning_no_stable_threshold"]["fit_date"].nunique())
            if warning_refit_dates > 0:
                render_callout(
                    f"**EVT Fit Warning**: {warning_refit_dates} of {total_refit_dates} distinct refit events triggered the "
                    "lowest-KS fallback heuristic because no threshold satisfied the strict shape/scale stability criterion. "
                    "This diagnostic heuristic requires analytical review.",
                    level="warning",
                )
            else:
                render_callout(
                    f"**EVT Stability Check**: All {total_refit_dates} refit events met shape/scale stability criteria without fallback.",
                    level="success",
                )
        else:
            st.info("EVT diagnostics table is available for the Out-of-Sample validation run.")

    # Methodology Explanations Expander
    with st.expander("Risk Methodology Assumptions & Theoretical Framework"):
        for m_key, m_desc in METHOD_DESCRIPTIONS.items():
            if m_key in available_methods or m_key == "evt_gpd":
                st.markdown(f"**{METHOD_LABELS.get(m_key, m_key)}**")
                st.markdown(f'<p style="color: {PALETTE["navy"]}; font-size: 0.90rem; line-height: 1.55; margin-top: 0.2rem; margin-bottom: 0.85rem;">{m_desc}</p>', unsafe_allow_html=True)
