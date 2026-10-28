"""Assumptions, Data Governance, Quality Diagnostics, and Limitations view."""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

from app.components import (
    render_header,
    render_callout,
    render_download_button,
)
from app.data_access import (
    load_data_quality_report,
    load_adf_report,
    load_data_cutoffs,
    load_config,
)
from app.formatting import format_currency, format_percent, format_date
from app.schemas import PALETTE, ASSET_REGISTRY


def render_assumptions_view() -> None:
    """Render comprehensive assumptions, data quality diagnostics, and methodology governance."""
    render_header(
        "Assumptions, Data Quality & Governance",
        "Data sources, date cutoffs, missing data handling, stationarity tests, and model limitations.",
    )

    cutoffs = load_data_cutoffs()
    quality_df = load_data_quality_report()
    adf_df = load_adf_report()
    config = load_config()

    # Three Explicit Timestamps to prevent misleading freshness claims
    st.subheader("Data Cutoffs and Observation Windows")
    st.caption("A file timestamp must not imply market freshness beyond the actual underlying price observations.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="1. Latest Raw Price Date",
            value=format_date(cutoffs["latest_price_date"]),
            help="Latest date with any asset price in the dataset (2026-10-22).",
        )
    with c2:
        st.metric(
            label="2. Latest Complete Return Date",
            value=format_date(cutoffs["latest_return_date"]),
            help="Latest date where ALL 7 portfolio assets have complete, valid simple returns (2026-10-20).",
        )
    with c3:
        st.metric(
            label="3. Latest Evaluated Risk Date",
            value=format_date(cutoffs["latest_risk_date"]),
            help="Latest date with evaluated one-day forward VaR/ES forecasts (2026-10-20).",
        )

    # Discrepancy Detection Callout (Dynamically generated)
    missing_assets = {k: v for k, v in cutoffs["missing_by_asset"].items() if v > 0}
    if missing_assets:
        missing_desc = ", ".join(f"**{k}** ({v} day{'s' if v > 1 else ''})" for k, v in missing_assets.items())
        all_missing_dates = sorted(set(d for dates in cutoffs.get("missing_dates", {}).values() for d in dates))
        missing_dates_str = ", ".join(f"**{d}**" for d in all_missing_dates) if all_missing_dates else "the detected dates"
        latest_complete_date = format_date(cutoffs["latest_return_date"])
        render_callout(
            f"**Data Discrepancy Detected**: {missing_desc} missing on {missing_dates_str}. "
            "Because percentage return calculations require consecutive trading day pairs, missing prices on any date make subsequent returns incomplete. "
            f"Consequently, portfolio-level analysis terminates cleanly on **{latest_complete_date}** without forward-filling or imputation.",
            level="warning",
        )

    st.markdown("---")

    # Tabs for Governance, Diagnostics, and Limitations
    tab_quality, tab_adf, tab_assumptions, tab_limitations = st.tabs([
        "🔍 Data Quality Report",
        "📊 Stationarity Tests (ADF)",
        "📜 Core Modeling Assumptions",
        "⚠️ Limitations & AI Disclosure",
    ])

    with tab_quality:
        st.subheader("Asset Data Quality & Outlier Screening")
        st.markdown(
            """
            - **Data Sources**: Adjusted close prices sourced via Yahoo Finance (`yfinance`); 3-month Treasury Bill rate via Federal Reserve Economic Data (`FRED: DGS3MO`).
            - **Outlier Protocol**: Potential extreme returns are flagged using an **8-MAD (Median Absolute Deviation)** robust threshold. Flagged returns are retained in the analysis without silent deletion to preserve empirical crisis tails.
            """
        )

        q_display = quality_df.copy()
        q_display["Asset Name"] = q_display["asset"].map(lambda a: ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a)
        q_display["First Price Date"] = q_display["first_price_date"].apply(format_date)
        q_display["Last Price Date"] = q_display["last_price_date"].apply(format_date)
        q_display["Missing Price Days"] = q_display["missing_price_days"].apply(lambda m: f"{int(m):,}")
        q_display["Extreme Return Flags"] = q_display["extreme_return_flags"].apply(lambda e: f"{int(e):,}")
        q_display["Min Log Return"] = q_display["min_log_return"].apply(lambda r: format_percent(r, 2))
        q_display["Max Log Return"] = q_display["max_log_return"].apply(lambda r: format_percent(r, 2))

        cols_q = [
            "asset",
            "Asset Name",
            "First Price Date",
            "Last Price Date",
            "Missing Price Days",
            "Extreme Return Flags",
            "Min Log Return",
            "Max Log Return",
        ]
        st.dataframe(q_display[cols_q].rename(columns={"asset": "Ticker"}), use_container_width=True, hide_index=True)
        render_download_button(quality_df, "data_quality_report.csv", "Download Data Quality Report")

    with tab_adf:
        st.subheader("Augmented Dickey-Fuller (ADF) Unit Root Tests")
        st.markdown(
            """
            The ADF test evaluates the null hypothesis of a **unit root (non-stationarity)** in log return series.
            Rejection at 1% confirms that return series are mean-reverting and stationary.
            """
        )

        adf_display = adf_df.copy()
        adf_display["Asset Name"] = adf_display["asset"].map(lambda a: ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a)
        adf_display["ADF Statistic"] = adf_display["adf_statistic"].apply(lambda s: f"{s:.4f}")
        adf_display["p-value"] = adf_display["p_value"].apply(lambda p: f"{p:.4e}")
        adf_display["Lags Used"] = adf_display["used_lag"].apply(lambda l: str(int(l)))
        adf_display["Observations"] = adf_display["n_obs"].apply(lambda n: f"{int(n):,}")
        adf_display["Stationary @ 1%?"] = adf_display["stationary_at_1pct"].apply(lambda s: "YES (p < 0.01)" if s else "NO")

        cols_adf = [
            "asset",
            "Asset Name",
            "ADF Statistic",
            "p-value",
            "Lags Used",
            "Observations",
            "Stationary @ 1%?",
        ]
        st.dataframe(adf_display[cols_adf].rename(columns={"asset": "Ticker"}), use_container_width=True, hide_index=True)

        render_callout(
            "**Statistical Scope Note**: Rejecting a unit-root null hypothesis provides evidence against a unit root "
            "under the test's assumptions. It does not prove all forms of stationarity, constant volatility, independence, "
            "normality, or absence of structural breaks.",
            level="info",
        )

    with tab_assumptions:
        st.subheader("Summary of Methodological Assumptions")
        st.markdown(
            f"""
            1. **Portfolio Optimization**:
               - **Objective**: Target-return minimum variance subject to long-only mandate constraints.
               - **Expected Return Blending**: 50% historical simple return and 50% CAPM equilibrium return (shrinkage to prior = {config['estimation']['expected_return_shrinkage_to_prior']}).
               - **Covariance Shrinkage**: Annualized log-return sample covariance shrunk 25% toward its diagonal to stabilize correlation estimates.
            2. **Timing and Allocation Split**:
               - **Training Horizon**: 2007-01-04 through 2017-12-29 (2,768 trading days).
               - **Out-of-Sample Evaluation**: 2018-01-02 through 2026-10-20 (2,212 trading days).
               - **One-Step-Ahead Forecasting**: All forecasts for day *t* utilize strictly information available on or before day *t - 1*.
            3. **Execution & Rebalancing**:
               - Daily constant-weight rebalancing without execution lag or liquidity slippage.
            4. **Risk-Free Rate Benchmarking**:
               - Retrospective KPIs use the latest available 3-month Treasury quote ({format_percent(0.0422, 2)}).
               - OOS Validation KPIs align an as-of daily FRED series (average {format_percent(0.0274, 2)}).
            """
        )

    with tab_limitations:
        st.subheader("Explicit Limitations & Disclosures for Investment Committee")
        st.markdown(
            """
            - **No Friction or Costs**: Results exclude management fees, performance fees, custodial expenses, trading commissions, bid-ask spread slippage, and market impact costs.
            - **Constant Notional Assumption**: Dollar VaR and ES figures reflect a static USD 10 Million notional rather than dynamic NAV compounding.
            - **Model Distribution Limits**:
              - Normal EWMA underestimates fat-tail probability during liquidity crises.
              - Empirical historical and bootstrap models cannot extrapolate beyond worst observed historical drops.
              - EVT Generalized Pareto models depend on subjective threshold stability heuristics.
            - **No Holding Out Confirmation on Alert Experiments**: Experimental CUSUM alert variants were inspected on the 2018–2026 development period and must be logged prospectively before being treated as proven.
            - **Human Review & AI Disclosure**:
              - Quantitative pipeline algorithms and this dashboard interface were implemented with AI pair programming (Codex / Antigravity IDE).
              - Team investment interpretation and fiduciary sign-off must be independently completed by human team members prior to submission.
            """
        )

        st.markdown("---")
        with st.expander("🔬 Experimental Candidate Models (Development Results Disclosure)"):
            st.markdown(
                """
                **Scope & Provenance Limitation**:
                - Advanced experiments under `experiments/` explore filtered risk models, GJR-GARCH conditional EVT, and regime-switching crisis responses.
                - These models were evaluated on the already-inspected 2018–2026 development sample. Their favorable retrospective
                  diagnostics do **not** constitute independent prospective validation.
                - Baseline models in the main pipeline are **not** replaced or altered by these candidate experiments.
                """
            )
