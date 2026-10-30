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
    page_title="TiraMisU | University of Peradeniya",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.design import apply_design, render_brand, PAGES, PAGE_LABELS


def main() -> None:
    apply_design()
    # Sidebar Header
    with st.sidebar:
        render_brand()
        st.caption("Thoughtful allocation. Transparent risk.")

        # Check query params for initial page selection
        default_nav_idx = 0
        qp = st.query_params.get("page", "")
        page_map = {
            "1": 0, "overview": 0,
            "2": 1, "portfolio": 1, "allocation": 1,
            "3": 2, "risk": 2,
            "4": 3, "validation": 3, "reliability": 3,
            "5": 4, "stress": 4, "crisis": 4,
            "6": 5, "assumptions": 5, "evidence": 5,
        }
        if qp and qp.lower() in page_map:
            target_page = PAGES[page_map[qp.lower()]]
            if st.session_state.get("tm_navigation") != target_page:
                st.session_state["tm_navigation"] = target_page
            default_nav_idx = page_map[qp.lower()]

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
            index=default_nav_idx,
            key="tm_navigation",
            format_func=lambda value: PAGE_LABELS[PAGES.index(value)],
            label_visibility="collapsed",
        )

        # Global Portfolio Basis Selector
        # Certain pages strictly require or lock a particular basis
        if page in ("2. Portfolio & Performance", "3. Risk Analysis"):
            st.markdown(
                f'<div style="font-size: 0.78rem; font-weight: 700; color: {PALETTE["navy"]}; margin-top: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;">Analysis Portfolio Basis</div>',
                unsafe_allow_html=True,
            )
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
            st.caption("**Locked Basis**: Frozen validation portfolio (OOS evaluation).")
        else:
            selected_basis = PortfolioBasis.CURRENT

        # Check for discrepancies between config.yaml and saved metadata
        from app.data_access import check_config_metadata_discrepancies, load_var_forecasts
        discrepancies = check_config_metadata_discrepancies()
        if discrepancies:
            st.warning("**Configuration Notice**:\n" + "\n".join(f"• {d}" for d in discrepancies))

        # Risk Forecast Controls (Confidence and Methodology)

        if page in ("3. Risk Analysis", "4. Model Validation"):
            def _format_confidence(c: float | str) -> str:
                try:
                    s = str(c).replace("%", "").strip()
                    val = float(s)
                    if val <= 1.0:
                        val *= 100.0
                    return f"{int(round(val))}%"
                except Exception:
                    return str(c)

            selected_confidence = st.select_slider(
                "Confidence Level:",
                options=CONFIDENCE_LEVELS,
                value=0.95 if page == "3. Risk Analysis" else 0.99,
                format_func=_format_confidence,
                help="Confidence level c for 1-day Value at Risk and Expected Shortfall.",
                key="global_confidence_slider",
            )

            # Determine available methods for this basis
            try:
                summary_for_basis = load_var_summary(selected_basis)
                available_methods = summary_for_basis["method"].unique().tolist()
            except Exception:
                available_methods = ["historical", "parametric_ewma", "monte_carlo_bootstrap"]

            method_names = {m: METHOD_LABELS.get(m, m) for m in available_methods}
            default_idx = 0
            if "parametric_ewma" in available_methods and page == "4. Model Validation":
                default_idx = available_methods.index("parametric_ewma")

            selected_method = st.selectbox(
                "Primary Risk Methodology:",
                options=available_methods,
                index=default_idx,
                format_func=lambda m: method_names.get(m, m),
                help="Select forecasting model. Changing methodology filters risk estimates; it does not alter portfolio weights.",
                key="global_method_selector",
            )
        else:
            selected_confidence = 0.95
            selected_method = "historical"

        # Evaluation Date Filtering (Used on Model Validation page)
        eval_start_date = None
        eval_end_date = None
        if page == "4. Model Validation":
            st.markdown("---")
            st.markdown("#### OOS Evaluation Window Filter")
            try:
                val_forecasts = load_var_forecasts(PortfolioBasis.VALIDATION)
                oos_min_date = val_forecasts["date"].min().date()
                oos_max_date = val_forecasts["date"].max().date()
            except Exception:
                oos_min_date = pd.to_datetime("2018-01-02").date()
                oos_max_date = pd.to_datetime("2026-10-20").date()

            with st.form("validation_filter_form"):
                d_start = st.date_input(
                    "Evaluation Start Date:",
                    value=oos_min_date,
                    min_value=oos_min_date,
                    max_value=oos_max_date,
                )
                d_end = st.date_input(
                    "Evaluation End Date:",
                    value=oos_max_date,
                    min_value=oos_min_date,
                    max_value=oos_max_date,
                )
                apply_filter = st.form_submit_button("Apply Evaluation Window")

                if d_start > d_end:
                    st.error(f"Reversed window: Start date ({d_start}) cannot be after end date ({d_end}).")
                    eval_start_date = pd.Timestamp(d_start)
                    eval_end_date = pd.Timestamp(d_end)
                else:
                    eval_start_date = pd.Timestamp(d_start)
                    eval_end_date = pd.Timestamp(d_end)

        st.markdown(
            f'<div style="margin: 14px 0 10px 0; border-top: 1px solid {PALETTE["border"]};"></div>',
            unsafe_allow_html=True,
        )

        # Data Status & Cutoffs Panel with clear date explanation
        try:
            cutoffs = load_data_cutoffs()
            lp = cutoffs['latest_price_date'].strftime('%Y-%m-%d')
            lr = cutoffs['latest_return_date'].strftime('%Y-%m-%d')
            lk = cutoffs['latest_risk_date'].strftime('%Y-%m-%d')
            with st.expander("Data Status & Freshness", expanded=False):
                st.markdown(
                    f"""
                    <div style="font-size: 0.78rem; color: {PALETTE['muted']}; line-height: 1.55;">
                        <div>• Prices: <b>{lp}</b> <span style="font-size: 0.72rem;">(raw)</span></div>
                        <div>• Returns: <b>{lr}</b> <span style="font-size: 0.72rem;">(complete)</span></div>
                        <div>• Risk Cutoff: <b>{lk}</b> <span style="font-size: 0.72rem;">(eval)</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        st.markdown(
            f'<div style="font-size: 0.78rem; font-weight: 600; color: {PALETTE["navy"]}; margin-top: 16px; margin-bottom: 3px;">Appearance</div>'
            '<div style="font-size: 12px; color: #a8a29a; line-height: 1.4; margin-bottom: 12px;">Theme toggle: ⋮ → Settings → Theme</div>',
            unsafe_allow_html=True,
        )

        if st.button("Reload Results Cache", help="Clears cached tables and reloads output CSV/JSON files from disk.", key="btn_reload_cache", width="stretch"):
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
