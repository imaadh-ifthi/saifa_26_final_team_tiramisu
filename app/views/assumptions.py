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
        "Evidence & Assumptions",
        "Assumptions and data governance: data sources, observation windows, missing price handling, stationarity tests, and model limitations.",
    )

    cutoffs = load_data_cutoffs()
    quality_df = load_data_quality_report()
    adf_df = load_adf_report()
    config = load_config()

    # Three Explicit Timestamps to prevent misleading freshness claims
    st.markdown(f'<h3 style="color: {PALETTE["navy"]}; font-size: 1.45rem; font-weight: 700; margin: 0.4rem 0 0.15rem 0;">Data Cutoffs & Observation Windows</h3>', unsafe_allow_html=True)
    st.caption("A file timestamp must not imply market freshness beyond the actual underlying price observations.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="1. Latest Raw Price Date",
            value=format_date(cutoffs["latest_price_date"]),
            delta="Raw price feed (before alignment)",
            delta_color="off",
            delta_arrow="off",
            help="Latest date with any asset price in the dataset (2026-10-22).",
        )
    with c2:
        st.metric(
            label="2. Latest Complete Return Date",
            value=format_date(cutoffs["latest_return_date"]),
            delta="All 7 assets present & aligned",
            delta_color="off",
            delta_arrow="off",
            help="Latest date where ALL 7 portfolio assets have complete, valid simple returns (2026-10-20).",
        )
    with c3:
        st.metric(
            label="3. Latest Evaluated Risk Date",
            value=format_date(cutoffs["latest_risk_date"]),
            delta="VaR / ES model evaluation cutoff",
            delta_color="off",
            delta_arrow="off",
            help="Latest date with evaluated one-day forward VaR/ES forecasts (2026-10-20).",
        )

    # Discrepancy Detection Callout (Quiet, calibrated institutional note with body-size text and left accent bar)
    missing_assets = {k: v for k, v in cutoffs["missing_by_asset"].items() if v > 0}
    if missing_assets:
        latest_complete_date = format_date(cutoffs["latest_return_date"])
        st.markdown(
            f'''
            <div style="background-color: {PALETTE['light_bg']}; border-left: 3px solid {PALETTE['muted']}; padding: 0.65rem 0.95rem; border-radius: 0 6px 6px 0; margin: 0.85rem 0 1.15rem 0; font-size: 0.88rem; color: {PALETTE['navy']}; line-height: 1.5;">
                <span style="font-weight: 600; color: {PALETTE['navy']};">Data Gap Protocol:</span> 
                EFA and AGG are missing on <b>2026-10-21</b>. Returns need consecutive prices, so portfolio analysis ends on <b>{latest_complete_date}</b> with no forward-filling.
            </div>
            ''',
            unsafe_allow_html=True,
        )

    # Tabs for Governance, Diagnostics, and Limitations
    tab_quality, tab_adf, tab_assumptions, tab_limitations = st.tabs([
        "Data Quality Report",
        "Stationarity Tests (ADF)",
        "Core Modeling Assumptions",
        "Limitations & AI Disclosure",
    ])

    with tab_quality:
        h_col, btn_col = st.columns([3, 1])
        with h_col:
            st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Asset Data Quality & Outlier Screening</h4>', unsafe_allow_html=True)
        with btn_col:
            render_download_button(quality_df, "data_quality_report.csv", "Download Report (CSV)")

        st.markdown(
            """
            - **Data Sources**: Adjusted close prices sourced via Yahoo Finance (`yfinance`); 3-month Treasury Bill rate via Federal Reserve Economic Data (`FRED: DGS3MO`).
            - **Outlier Protocol**: Potential extreme returns are flagged using an **8-MAD (Median Absolute Deviation)** robust threshold. Flagged returns are retained in the analysis without silent deletion to preserve empirical crisis tails.
            """
        )

        q_rows_html = []
        n_q = len(quality_df)
        for idx, row in enumerate(quality_df.itertuples()):
            asset_name = ASSET_REGISTRY[row.asset].name if row.asset in ASSET_REGISTRY else row.asset
            row_bg = PALETTE["light_bg"] if idx % 2 == 1 else PALETTE["card_bg"]
            
            missing_val = int(row.missing_price_days)
            missing_color = PALETTE["amber"] if missing_val > 0 else PALETTE["navy"]
            missing_weight = "650" if missing_val > 0 else "400"
            
            # Last row has no bottom border
            b_bottom = "none" if idx == n_q - 1 else f"1px solid {PALETTE['border']}"
            
            q_rows_html.append(f'''
            <tr style="background-color: {row_bg};">
                <td style="padding: 0.55rem 0.65rem; text-align: left; white-space: nowrap; border: none; border-bottom: {b_bottom};">
                    <span style="font-weight: 700; color: {PALETTE['navy']};">{row.asset}</span>
                    &nbsp;<span style="color: {PALETTE['muted']}; font-size: 0.8rem;">({asset_name})</span>
                </td>
                <td style="padding: 0.55rem 0.55rem; text-align: left; font-variant-numeric: tabular-nums; color: {PALETTE['slate']}; white-space: nowrap; border: none; border-bottom: {b_bottom};">{format_date(row.first_price_date)}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: left; font-variant-numeric: tabular-nums; color: {PALETTE['slate']}; white-space: nowrap; border: none; border-bottom: {b_bottom};">{format_date(row.last_price_date)}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; color: {missing_color}; font-weight: {missing_weight}; white-space: nowrap; border: none; border-bottom: {b_bottom};">{missing_val:,}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{int(row.extreme_return_flags):,}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{format_percent(row.min_log_return, 2)}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{format_percent(row.max_log_return, 2)}</td>
            </tr>
            ''')

        q_rows_str = "".join(q_rows_html)
        q_table_html = f'''
        <div style="overflow-x: auto; width: 100%; margin-bottom: 0.25rem;">
            <table style="width: 100%; border-collapse: collapse; font-family: Inter, -apple-system, sans-serif; font-size: 0.83rem; border: none;">
                <thead>
                    <tr style="background-color: {PALETTE['light_bg']}; color: {PALETTE['muted']}; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase;">
                        <th style="text-align: left; padding: 0.55rem 0.65rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">TICKER & ASSET</th>
                        <th style="text-align: left; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">FIRST DATE</th>
                        <th style="text-align: left; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">LAST DATE</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">MISSING DAYS</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">EXTREME FLAGS</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">MIN LOG RETURN</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">MAX LOG RETURN</th>
                    </tr>
                </thead>
                <tbody>
                    {q_rows_str}
                </tbody>
            </table>
        </div>
        '''
        clean_q_html = "\n".join(line.lstrip() for line in q_table_html.splitlines() if line.strip())
        st.markdown(clean_q_html, unsafe_allow_html=True)

    with tab_adf:
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Augmented Dickey-Fuller (ADF) Unit Root Tests</h4>', unsafe_allow_html=True)
        st.markdown(
            """
            The ADF test evaluates the null hypothesis of a **unit root (non-stationarity)** in log return series.
            Rejection at 1% confirms that return series are mean-reverting and stationary.
            """
        )

        adf_rows_html = []
        n_adf = len(adf_df)
        for idx, row in enumerate(adf_df.itertuples()):
            asset_name = ASSET_REGISTRY[row.asset].name if row.asset in ASSET_REGISTRY else row.asset
            row_bg = PALETTE["light_bg"] if idx % 2 == 1 else PALETTE["card_bg"]
            
            if row.stationary_at_1pct:
                stat_pill = f'<span style="background-color: {PALETTE["green_light"]}; color: {PALETTE["green"]}; border: 1px solid rgba(8, 120, 109, 0.25); padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 600; font-size: 0.74rem; white-space: nowrap;">✓ Stationary (p &lt; 0.01)</span>'
            else:
                stat_pill = f'<span style="color: {PALETTE["red"]}; font-weight: 600;">NO</span>'
            
            # Last row has no bottom border
            b_bottom = "none" if idx == n_adf - 1 else f"1px solid {PALETTE['border']}"
            
            adf_rows_html.append(f'''
            <tr style="background-color: {row_bg};">
                <td style="padding: 0.55rem 0.65rem; text-align: left; white-space: nowrap; border: none; border-bottom: {b_bottom};">
                    <span style="font-weight: 700; color: {PALETTE['navy']};">{row.asset}</span>
                    &nbsp;<span style="color: {PALETTE['muted']}; font-size: 0.8rem;">({asset_name})</span>
                </td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{row.adf_statistic:.4f}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{row.p_value:.4e}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{int(row.used_lag)}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border: none; border-bottom: {b_bottom};">{int(row.n_obs):,}</td>
                <td style="padding: 0.55rem 0.55rem; text-align: center; white-space: nowrap; border: none; border-bottom: {b_bottom};">{stat_pill}</td>
            </tr>
            ''')

        adf_rows_str = "".join(adf_rows_html)
        adf_table_html = f'''
        <div style="overflow-x: auto; width: 100%; margin-bottom: 0.85rem;">
            <table style="width: 100%; border-collapse: collapse; font-family: Inter, -apple-system, sans-serif; font-size: 0.83rem; border: none;">
                <thead>
                    <tr style="background-color: {PALETTE['light_bg']}; color: {PALETTE['muted']}; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase;">
                        <th style="text-align: left; padding: 0.55rem 0.65rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">TICKER & ASSET</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">ADF STATISTIC</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">P-VALUE</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">LAGS USED</th>
                        <th style="text-align: right; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">OBSERVATIONS</th>
                        <th style="text-align: center; padding: 0.55rem 0.55rem; white-space: nowrap; border: none; border-top: 1px solid {PALETTE['border']}; border-bottom: 2px solid {PALETTE['border']};">STATIONARITY (@ 1%)</th>
                    </tr>
                </thead>
                <tbody>
                    {adf_rows_str}
                </tbody>
            </table>
        </div>
        '''
        clean_adf_html = "\n".join(line.lstrip() for line in adf_table_html.splitlines() if line.strip())
        st.markdown(clean_adf_html, unsafe_allow_html=True)

        st.markdown(
            f'''
            <div style="background-color: {PALETTE['light_bg']}; border-left: 3px solid {PALETTE['teal']}; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin-top: 0.5rem; font-size: 0.90rem; line-height: 1.55;">
                <span style="font-weight: 700; color: {PALETTE['teal']};">Statistical Scope Note:</span> 
                <span style="color: {PALETTE['navy']};">Rejecting a unit-root null hypothesis provides evidence against a unit root under the test's assumptions. It does not prove all forms of stationarity, constant volatility, independence, normality, or absence of structural breaks.</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with tab_assumptions:
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Summary of Methodological Assumptions</h4>', unsafe_allow_html=True)
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
        st.markdown(f'<h4 style="color: {PALETTE["navy"]}; font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0 0.1rem 0;">Explicit Limitations & Disclosures for Investment Committee</h4>', unsafe_allow_html=True)
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
        with st.expander("Experimental Candidate Models (Development Results Disclosure)"):
            st.markdown(
                """
                **Scope & Provenance Limitation**:
                - Advanced experiments under `experiments/` explore filtered risk models, GJR-GARCH conditional EVT, and regime-switching crisis responses.
                - These models were evaluated on the already-inspected 2018–2026 development sample. Their favorable retrospective
                  diagnostics do **not** constitute independent prospective validation.
                - Baseline models in the main pipeline are **not** replaced or altered by these candidate experiments.
                """
            )
