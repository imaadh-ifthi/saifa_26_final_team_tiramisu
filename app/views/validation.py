"""Formal Model Validation and Statistical Backtesting view."""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

from app.components import (
    render_header,
    render_basis_badge,
    render_callout,
    render_traffic_light_badge,
    render_download_button,
)
from app.data_access import (
    load_backtest_results,
    load_var_forecasts,
    load_analytical_run,
    recompute_backtest_window,
)
from app.charts import plot_validation_breaches
from app.formatting import format_currency, format_percent, format_date, format_number
from app.schemas import PortfolioBasis, PALETTE, METHOD_LABELS


def render_validation_view(
    selected_confidence: float = 0.99,
    selected_method: str = "parametric_ewma",
    eval_start_date: pd.Timestamp | str | None = None,
    eval_end_date: pd.Timestamp | str | None = None,
) -> None:
    """Render formal model validation, likelihood-ratio tests, and Basel traffic light."""
    render_header(
        "Model Validation & Statistical Backtesting",
        "Formal out-of-sample (OOS) statistical calibration tests and regulatory exception monitoring.",
    )

    # Strictly locked to Frozen validation allocation
    render_basis_badge("Frozen validation portfolio (Out-of-sample evaluation: 2018–2026)")

    analytical_meta = load_analytical_run()
    train_start = analytical_meta.get("training_start", "2007-01-04")
    train_end = analytical_meta.get("training_end", "2017-12-29")
    oos_weights = analytical_meta.get("weights", {})

    st.markdown(
        f"""
        <div style="background-color: {PALETTE['light_bg']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.88rem;">
            <b>Frozen Training Protocol</b>: Portfolio weights were optimized strictly on <b>{train_start} to {train_end}</b> 
            (SPY {format_percent(oos_weights.get('SPY', 0.321), 1)}, AGG {format_percent(oos_weights.get('AGG', 0.379), 1)}, 
            GLD {format_percent(oos_weights.get('GLD', 0.15), 1)}, XLK {format_percent(oos_weights.get('XLK', 0.15), 1)}). 
            Risk models are evaluated strictly out-of-sample beginning January 2018 with zero parameter leakage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecasts_df = load_var_forecasts(PortfolioBasis.VALIDATION)
    baseline_backtests = load_backtest_results()

    # Determine evaluation window
    full_min_date = forecasts_df["date"].min()
    full_max_date = forecasts_df["date"].max()

    curr_start = pd.Timestamp(eval_start_date) if eval_start_date else full_min_date
    curr_end = pd.Timestamp(eval_end_date) if eval_end_date else full_max_date

    is_filtered_window = (curr_start > full_min_date) or (curr_end < full_max_date)

    # Recompute statistics dynamically for the chosen window using backend test functions
    test_results = recompute_backtest_window(
        forecasts_df=forecasts_df,
        start_date=curr_start,
        end_date=curr_end,
        method=selected_method,
        confidence=selected_confidence,
        significance=0.05,
    )

    if "error" in test_results:
        st.error(test_results["error"])
        return

    n_obs = test_results["observations"]
    actual_exceptions = test_results["exceptions"]
    observed_rate = test_results["exception_rate"]
    expected_rate = test_results["expected_exception_rate"]
    expected_exceptions = n_obs * expected_rate

    kupiec_p = test_results["kupiec_p_value"]
    kupiec_reject = test_results["kupiec_reject"]
    indep_p = test_results["independence_p_value"]
    indep_reject = test_results["independence_reject"]
    indep_status = test_results.get("independence_status", "ok")
    cc_p = test_results.get("conditional_coverage_p_value", np.nan)
    cc_reject = test_results.get("conditional_coverage_reject", None)

    traffic_status = test_results["traffic_light"]
    traffic_obs = test_results["traffic_light_observations"]
    traffic_exceptions = test_results["traffic_light_exceptions"]

    # Regulatory & Statistical Summary Banner
    st.subheader(f"Evaluation Window Summary: {format_date(curr_start)} to {format_date(curr_end)}")
    if is_filtered_window:
        render_callout(
            f"**Filtered Date Window Active**: Displayed statistics are dynamically recomputed from {n_obs:,} forecast observations. "
            "Model parameters are fixed; only the evaluation sample is filtered.",
            level="info",
        )

    # Core Visual: Realized Losses vs VaR Threshold (Positive Loss Convention)
    fig_breach = plot_validation_breaches(
        forecasts_df[
            (forecasts_df["date"] >= curr_start) & (forecasts_df["date"] <= curr_end)
        ],
        selected_method,
        selected_confidence,
    )
    st.plotly_chart(fig_breach, use_container_width=True)

    # Basel Traffic Light Indicator
    st.subheader("Basel Regulatory Traffic Light (99% Confidence / Trailing 250 Days)")
    render_traffic_light_badge(traffic_status, traffic_exceptions, traffic_obs)

    if np.isclose(selected_confidence, 0.99):
        st.caption(
            f"Traffic light window covers the last {traffic_obs} evaluated trading days within the selected sample. "
            f"Zone thresholds: Green (0–4 exceptions), Yellow (5–9 exceptions), Red (10+ exceptions)."
        )
    else:
        st.info("The Basel traffic light framework is strictly defined for 99% confidence over 250 days; it is not applicable at 90% or 95% confidence.")

    st.markdown("---")

    # Detailed Statistical Backtest Cards
    st.subheader("Formal Likelihood-Ratio Hypothesis Tests (5% Significance Level)")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(
            label="Actual Exceptions",
            value=f"{actual_exceptions:,}",
            delta=f"{actual_exceptions - expected_exceptions:+.1f} vs Expected ({expected_exceptions:.1f})",
            help="Number of days where realized portfolio loss exceeded forecast VaR.",
        )
    with s2:
        st.metric(
            label="Exception Rate",
            value=format_percent(observed_rate, 2),
            delta=f"{format_percent(observed_rate - expected_rate, 2, include_sign=True)} vs Expected ({format_percent(expected_rate, 2)})",
            help="Realized breach proportion (exceptions / total observations).",
        )
    with s3:
        kupiec_status = "REJECTS at 5%" if kupiec_reject else "DOES NOT REJECT"
        kupiec_color = PALETTE["red"] if kupiec_reject else PALETTE["green"]
        st.markdown(
            f"""
            <div style="background-color: white; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.8rem; color: {PALETTE['muted']};">Kupiec POF Test (Coverage)</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {kupiec_color}; margin: 0.25rem 0;">{kupiec_status}</div>
                <div style="font-size: 0.78rem; color: {PALETTE['muted']};">LR = {test_results['kupiec_lr']:.2f} | p = {kupiec_p:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        if indep_status != "ok":
            indep_text = "INSUFFICIENT TRANSITIONS"
            indep_color = PALETTE["muted"]
            indep_sub = "Zero transition counts"
        else:
            indep_text = "REJECTS at 5%" if indep_reject else "DOES NOT REJECT"
            indep_color = PALETTE["red"] if indep_reject else PALETTE["green"]
            indep_sub = f"LR = {test_results['independence_lr']:.2f} | p = {indep_p:.4f}"
        st.markdown(
            f"""
            <div style="background-color: white; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 0.6rem; text-align: center;">
                <div style="font-size: 0.8rem; color: {PALETTE['muted']};">Christoffersen Independence</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {indep_color}; margin: 0.25rem 0;">{indep_text}</div>
                <div style="font-size: 0.78rem; color: {PALETTE['muted']};">{indep_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Full Backtesting Results Across All Methods and Confidences Table
    st.subheader("Baseline Model Calibration Master Table (Full OOS Sample: 2018–2026)")
    b_display = baseline_backtests.copy()
    b_display["Method"] = b_display["method"].map(lambda m: METHOD_LABELS.get(m, m))
    b_display["Confidence"] = b_display["confidence"].apply(lambda c: f"{int(c*100)}%")
    b_display["Obs"] = b_display["observations"].apply(lambda o: f"{int(o):,}")
    b_display["Breaches"] = b_display["exceptions"].apply(lambda x: f"{int(x):,}")
    b_display["Breach Rate"] = b_display["exception_rate"].apply(lambda r: format_percent(r, 2))
    b_display["Kupiec p-val"] = b_display["kupiec_p_value"].apply(lambda p: f"{p:.4f}")
    b_display["Kupiec Outcome"] = b_display["kupiec_reject"].apply(lambda r: "Reject (5%)" if r else "Pass")
    b_display["Independence p-val"] = b_display["independence_p_value"].apply(lambda p: f"{p:.4f}" if np.isfinite(p) else "N/A")
    b_display["Joint CC Outcome"] = b_display["conditional_coverage_reject"].apply(lambda r: "Reject (5%)" if r else "Pass")
    b_display["Basel Zone"] = b_display["traffic_light"].apply(lambda t: t.upper())

    master_cols = [
        "Method",
        "Confidence",
        "Obs",
        "Breaches",
        "Breach Rate",
        "Kupiec p-val",
        "Kupiec Outcome",
        "Independence p-val",
        "Joint CC Outcome",
        "Basel Zone",
    ]
    st.dataframe(b_display[master_cols], use_container_width=True, hide_index=True)
    render_download_button(baseline_backtests, "backtest_results_master.csv", "Export Validation Table CSV")

    # Critical Honest Disclosure Expander
    st.markdown("---")
    with st.expander("🚨 Crucial Governance Insight: Reconciling Green Traffic Lights with Statistical Rejections"):
        st.markdown(
            f"""
            <div style="font-size: 0.92rem; color: {PALETTE['navy']}; line-height: 1.55;">
                <p><b>Why does the Basel Traffic Light show 'GREEN' while formal tests REJECT at 99%?</b></p>
                <ul>
                    <li><b>Different Time Horizons</b>: The Basel Traffic Light evaluates <i>only the most recent 250 trading days</i> (late 2025 to late 2026). During this recent period of lower equity volatility, only 1–4 breaches occurred across models, putting all models into the regulatory Green Zone.</li>
                    <li><b>Multi-Year Volatility Clusters</b>: Over the full 2,212 out-of-sample days (which include the massive market turbulence of the COVID-19 crash in March 2020 and the 2022 interest rate shock), exceptions clustered heavily.</li>
                    <li><b>Statistical Calibration Failure</b>: Over 2,212 days at 99% confidence, we expect ~22 exceptions. Historical had 38, Monte Carlo 32, EVT 31, and EWMA Normal had 48. Because these breaches occurred in tight clusters rather than independently spaced occurrences, the joint conditional coverage test <b>rejects all four baseline models at 99%</b>.</li>
                    <li><b>Takeaway for the Investment Committee</b>: A recent green traffic light does <i>not</i> mean the risk model is mathematically proven or immune to tail risk. The endowment must maintain substantial defensive reserves and supplement daily VaR with stress testing.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
