"""Main Streamlit application entry point for Quant Edge 1.0."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add repo root to python path to allow importing src and app modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.schemas import PortfolioBasis, CONFIDENCE_LEVELS, METHOD_LABELS, PALETTE
from app.data_access import (
    load_data_cutoffs,
    clear_app_cache,
    load_var_summary,
)
from app.views.overview import render_overview_view
from app.views.portfolio import render_portfolio_view
from app.views.risk import render_risk_view
from app.views.validation import render_validation_view
from app.views.stress import render_stress_view
from app.views.assumptions import render_assumptions_view

# Streamlit Page Configuration
st.set_page_config(
    page_title="Serendib Endowment | Quant Edge 1.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for clean institutional typography and layout
st.markdown(
    f"""
    <style>
        /* Main background & typography */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {PALETTE['navy']};
        }}
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #f1f5f9;
            border-right: 1px solid {PALETTE['border']};
        }}
        /* Clean tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            color: {PALETTE['slate']};
            border-radius: 4px 4px 0 0;
        }}
        .stTabs [aria-selected="true"] {{
            color: {PALETTE['teal']} !important;
            border-bottom: 2px solid {PALETTE['teal']} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # Sidebar Header
    with st.sidebar:
        st.markdown(
            f"""
            <div style="margin-bottom: 1.25rem;">
                <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {PALETTE['teal']};">
                    Quant Edge 1.0
                </div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {PALETTE['navy']}; line-height: 1.2;">
                    Serendib Endowment
                </div>
                <div style="font-size: 0.8rem; color: {PALETTE['muted']}; margin-top: 0.2rem;">
                    Portfolio Value: USD 10,000,000
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Navigation
        page = st.radio(
            "Navigation",
            options=[
                "1. Overview",
                "2. Portfolio & Performance",
                "3. Risk Analysis",
                "4. Model Validation",
                "5. Stress & Risk Contributions",
                "6. Assumptions & Data Quality",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Global Portfolio Basis Selector
        # Certain pages strictly require or lock a particular basis
        if page in ("2. Portfolio & Performance", "3. Risk Analysis"):
            st.markdown("#### Analysis Portfolio Basis")
            basis_choice = st.radio(
                "Select Portfolio Basis:",
                options=[
                    "Current proposed portfolio",
                    "Frozen validation portfolio",
                ],
                index=0,
                help="Switch between current 2007-2026 proposed allocation (retrospective replay) and the frozen 2007-2017 training allocation (out-of-sample).",
            )
            selected_basis = (
                PortfolioBasis.CURRENT
                if basis_choice == "Current proposed portfolio"
                else PortfolioBasis.VALIDATION
            )
        elif page == "4. Model Validation":
            selected_basis = PortfolioBasis.VALIDATION
            st.caption("🔒 **Locked Basis**: Frozen validation portfolio (OOS evaluation).")
        else:
            selected_basis = PortfolioBasis.CURRENT
            st.caption("🔒 **Locked Basis**: Current proposed allocation (retrospective).")

        st.markdown("---")

        # Risk Forecast Controls (Confidence and Methodology)
        st.markdown("#### Risk & Forecast Parameters")
        selected_confidence = st.select_slider(
            "Confidence Level:",
            options=CONFIDENCE_LEVELS,
            value=0.95,
            format_func=lambda c: f"{int(c*100)}%",
            help="Confidence level c for 1-day Value at Risk and Expected Shortfall.",
        )

        # Determine available methods for this basis
        try:
            summary_for_basis = load_var_summary(selected_basis)
            available_methods = summary_for_basis["method"].unique().tolist()
        except Exception:
            available_methods = ["historical", "parametric_ewma", "monte_carlo_bootstrap"]

        method_names = {m: METHOD_LABELS.get(m, m) for m in available_methods}
        selected_method = st.selectbox(
            "Primary Risk Methodology:",
            options=available_methods,
            index=0,
            format_func=lambda m: method_names.get(m, m),
            help="Select forecasting model. Changing methodology filters risk estimates; it does not alter portfolio weights.",
        )

        # Evaluation Date Filtering (Used especially in Model Validation)
        eval_start_date = None
        eval_end_date = None
        if page == "4. Model Validation":
            st.markdown("---")
            st.markdown("#### OOS Evaluation Window Filter")
            with st.form("validation_filter_form"):
                d_start = st.date_input(
                    "Evaluation Start Date:",
                    value=pd.to_datetime("2018-01-02"),
                    min_value=pd.to_datetime("2018-01-02"),
                    max_value=pd.to_datetime("2026-10-20"),
                )
                d_end = st.date_input(
                    "Evaluation End Date:",
                    value=pd.to_datetime("2026-10-20"),
                    min_value=pd.to_datetime("2018-01-02"),
                    max_value=pd.to_datetime("2026-10-20"),
                )
                apply_filter = st.form_submit_button("Apply Evaluation Window")
                if apply_filter:
                    eval_start_date = pd.Timestamp(d_start)
                    eval_end_date = pd.Timestamp(d_end)
                else:
                    eval_start_date = pd.Timestamp(d_start)
                    eval_end_date = pd.Timestamp(d_end)

        st.markdown("---")

        # Data Status & Cutoffs Panel
        try:
            cutoffs = load_data_cutoffs()
            st.markdown(
                f"""
                <div style="font-size: 0.8rem; color: {PALETTE['muted']}; line-height: 1.45;">
                    <b>Data Cutoff Status</b><br>
                    • Latest Prices: <b>{cutoffs['latest_price_date'].strftime('%Y-%m-%d')}</b><br>
                    • Complete Returns: <b>{cutoffs['latest_return_date'].strftime('%Y-%m-%d')}</b><br>
                    • Evaluated Risk: <b>{cutoffs['latest_risk_date'].strftime('%Y-%m-%d')}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reload Results Cache", help="Clears cached tables and reloads output CSV/JSON files from disk."):
            clear_app_cache()
            st.success("Cache cleared! Reloaded analytical outputs.")
            st.rerun()

    # Route to selected page view
    if page == "1. Overview":
        render_overview_view()
    elif page == "2. Portfolio & Performance":
        render_portfolio_view(selected_basis=selected_basis)
    elif page == "3. Risk Analysis":
        render_risk_view(
            selected_basis=selected_basis,
            selected_confidence=selected_confidence,
            selected_method=selected_method,
        )
    elif page == "4. Model Validation":
        render_validation_view(
            selected_confidence=selected_confidence,
            selected_method=selected_method,
            eval_start_date=eval_start_date,
            eval_end_date=eval_end_date,
        )
    elif page == "5. Stress & Risk Contributions":
        render_stress_view()
    elif page == "6. Assumptions & Data Quality":
        render_assumptions_view()


if __name__ == "__main__":
    main()
