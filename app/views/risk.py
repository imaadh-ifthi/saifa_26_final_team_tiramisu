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
        "Risk Analysis: VaR & Expected Shortfall",
        "Probabilistic 1-day loss thresholds (VaR) and expected loss beyond the threshold (Expected Shortfall).",
    )
    render_basis_badge(basis_str)

    # Core concept definition banner
    st.markdown(
        f"""
        <div style="background-color: {PALETTE['light_bg']}; border-left: 4px solid {PALETTE['teal']}; border-radius: 4px; padding: 0.9rem 1.2rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.95rem; font-weight: 600; color: {PALETTE['navy']}; margin-bottom: 0.25rem;">
                Core Risk Metric Distinction:
            </div>
            <div style="font-size: 0.88rem; color: {PALETTE['slate']}; line-height: 1.45;">
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
        v_pct = float(curr_row["latest_var_pct"])
        v_usd = float(curr_row["latest_var_usd"])
        e_pct = float(curr_row["latest_es_pct"])
        e_usd = float(curr_row["latest_es_usd"])
        eval_date = curr_row["latest_date"]
    else:
        v_pct = v_usd = e_pct = e_usd = 0.0
        eval_date = latest_common_date

    st.markdown(f"### Latest 1-Day Loss Forecasts ({format_date(eval_date)})")
    st.caption(
        f"Forecast for **{format_date(eval_date)}**, estimated using market information through preceding trading day. "
        f"Horizon: 1 Trading Day | Reference Notional: {format_currency(notional, 0)}"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label=f"1-Day VaR ({int(selected_confidence*100)}% Conf.)",
            value=format_currency(v_usd, 0),
            help=f"Dollar loss expected to be exceeded on only {format_percent(1 - selected_confidence, 0)} of trading days.",
        )
    with m2:
        st.metric(
            label="VaR (% of Portfolio)",
            value=format_percent(v_pct, 2),
            help="Loss as a percentage of total portfolio value.",
        )
    with m3:
        st.metric(
            label=f"Expected Shortfall ({int(selected_confidence*100)}%)",
            value=format_currency(e_usd, 0),
            help="Average dollar loss within the worst (1 - c) tail of outcomes.",
        )
    with m4:
        st.metric(
            label="ES (% of Portfolio)",
            value=format_percent(e_pct, 2),
            help="Expected loss as a percentage of total portfolio value within the extreme tail.",
        )

    st.markdown("---")

    # Method comparison table at current confidence level
    st.subheader(f"Methodology Comparison Table ({int(selected_confidence*100)}% Confidence)")

    comp_df = summary_df[np.isclose(summary_df["confidence"], selected_confidence)].copy()
    comp_df["Methodology"] = comp_df["method"].map(lambda m: METHOD_LABELS.get(m, m))
    comp_df["1-Day VaR (%)"] = comp_df["latest_var_pct"].apply(lambda v: format_percent(v, 2))
    comp_df["1-Day VaR (USD)"] = comp_df["latest_var_usd"].apply(lambda v: format_currency(v, 0))
    comp_df["Expected Shortfall (%)"] = comp_df["latest_es_pct"].apply(lambda e: format_percent(e, 2))
    comp_df["Expected Shortfall (USD)"] = comp_df["latest_es_usd"].apply(lambda e: format_currency(e, 0))
    comp_df["Historical Exceptions"] = comp_df["number_of_exceptions"].apply(lambda x: f"{int(x):,}")
    comp_df["Evaluated Days"] = comp_df["observations"].apply(lambda o: f"{int(o):,}")

    table_cols = [
        "Methodology",
        "1-Day VaR (%)",
        "1-Day VaR (USD)",
        "Expected Shortfall (%)",
        "Expected Shortfall (USD)",
        "Historical Exceptions",
        "Evaluated Days",
    ]
    st.dataframe(comp_df[table_cols], use_container_width=True, hide_index=True)

    # Disclosure regarding EVT availability on Current vs Validation portfolio
    if selected_basis == PortfolioBasis.CURRENT:
        render_callout(
            "**EVT Model Availability Notice**: Extreme Value Theory (EVT-GPD) is calibrated strictly for the "
            "**Frozen validation portfolio** across 1,000-day rolling windows (outputs/tables/risk_summary_oos.csv). "
            "To view EVT tail fits alongside baseline methods, select the **'Frozen validation portfolio'** basis in the sidebar.",
            level="info",
        )

    # Visualizations: Bar comparison & Time Series
    tab_chart_time, tab_chart_bar, tab_evt = st.tabs([
        "📉 Risk Forecasts Over Time",
        "📊 Cross-Method Bar Comparison",
        "🔍 EVT Threshold Diagnostics",
    ])

    with tab_chart_time:
        st.subheader(f"Daily VaR & ES Evolution — {METHOD_LABELS.get(selected_method, selected_method)}")
        if selected_method in available_methods:
            fig_ts = plot_var_es_time_series(forecasts_df, selected_method, selected_confidence)
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning(f"Selected method '{selected_method}' is not available in the {selected_basis.value} dataset.")

    with tab_chart_bar:
        st.subheader("Cross-Methodology Loss Comparison at Latest Date")
        fig_bar = plot_var_method_comparison_bar(summary_df, selected_confidence)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_evt:
        st.subheader("Extreme Value Theory (EVT-GPD) Tail Fits")
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
            d_display = d_display.rename(columns={
                "threshold_quantile": "Quantile",
                "threshold_log": "Log Loss Threshold (u)",
                "exceedances": "Exceedances",
                "shape": "Shape (ξ)",
                "scale": "Scale (β)",
                "modified_scale": "Modified Scale",
                "stable_with_next": "Stable?",
                "selection_status": "Selection Status",
                "selected": "Selected?",
            })
            st.dataframe(d_display, use_container_width=True, hide_index=True)

            warnings_count = int(evt_df["selection_status"].eq("warning_no_stable_threshold").sum())
            if warnings_count > 0:
                render_callout(
                    f"**EVT Fit Warning**: {warnings_count} candidate fits used the lowest-KS fallback because no threshold "
                    "met the strict shape/scale stability criteria. This heuristic requires analytical review.",
                    level="warning",
                )
        else:
            st.info("EVT diagnostics table is available for the Out-of-Sample validation run.")

    # Methodology Explanations Expander
    with st.expander("📘 Risk Methodology Assumptions & Theoretical Framework"):
        for m_key, m_desc in METHOD_DESCRIPTIONS.items():
            if m_key in available_methods or m_key == "evt_gpd":
                st.markdown(f"**{METHOD_LABELS.get(m_key, m_key)}**")
                st.markdown(f"_{m_desc}_")
                st.markdown("<br>", unsafe_allow_html=True)
