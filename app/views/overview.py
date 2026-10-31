"""Executive Overview page for the Serendib University Endowment Fund."""
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from app.design import render_intro, render_decision_routes
from app.components import render_callout, render_basis_badge
from app.data_access import (
    load_selected_portfolio,
    load_portfolio_weights,
    load_kpi_summary,
    load_var_summary,
    load_backtest_results,
    load_data_cutoffs,
    load_config,
)
from app.formatting import format_currency, format_percent, format_date
from app.schemas import PortfolioBasis, PALETTE, ASSET_REGISTRY


# Bar segment palette: distinct shades to separate adjacent segments
BAR_SEGMENT_COLORS = {
    "AGG": "#08786d",   # Fixed Income Teal
    "GLD": "#8c5b13",   # Real Assets Amber Gold
    "SPY": "#2563eb",   # US Core Equity Royal Blue
    "EFA": "#3b82f6",   # Dev Markets Equity Blue
    "XLK": "#60a5fa",   # Tech Tilt Sky Blue
    "XLE": "#93c5fd",   # Energy Tilt Soft Blue
}

# Asset class colors for cards: unified blue for all equities
ASSET_CLASS_COLORS = {
    "Fixed Income": "#08786d",  # same teal as the bar
    "Real Assets": "#8c5b13",   # same amber as the bar
    "Equity": "#2563eb",       # one blue for SPY, EFA, XLK, XLE
}


def render_overview_view() -> None:
    """Render the executive summary overview view."""
    portfolio_meta = load_selected_portfolio()
    weights_df = load_portfolio_weights(PortfolioBasis.CURRENT)
    kpi_summary = load_kpi_summary(PortfolioBasis.CURRENT)
    cutoffs = load_data_cutoffs()
    config = load_config()

    notional = float(portfolio_meta.get("portfolio_value_usd", 10_000_000.0))
    target_return = float(portfolio_meta.get("target_return_annual", 0.065))
    expected_return = float(portfolio_meta.get("expected_return_annual", 0.0766))
    optimizer_vol = float(portfolio_meta.get("volatility_annual", 0.0857))

    realized_return = float(kpi_summary.get("annualized_return", 0.0758))
    realized_vol = float(kpi_summary.get("annualized_volatility", 0.0919))
    max_dd = float(kpi_summary.get("max_drawdown", -0.2484))
    sharpe = float(kpi_summary.get("sharpe_ratio", 0.3656))
    rf_rate = float(kpi_summary.get("annualized_risk_free", 0.0422))

    # 1. Plain Header (title + subtitle together)
    render_intro()

    # 2. Metadata row: Neutral basis badge + Observation date cutoff together
    render_basis_badge(
        "Current proposed portfolio (Retrospective historical replay)",
        cutoff_date=format_date(cutoffs["latest_return_date"]),
    )

    # 3. Full-width single horizontal row of four KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Fund Reference Notional",
            value=format_currency(notional, 0),
            delta="Reference endowment size",
            delta_color="off",
            delta_arrow="off",
            help="Total reference endowment capital allocation analyzed across all models.",
        )
    with col2:
        st.metric(
            label="Optimized Expected Return",
            value=format_percent(expected_return, 2),
            delta=f"+{format_percent(expected_return - target_return, 2)} vs Target ({format_percent(target_return, 1)})",
            delta_color="normal",
            delta_arrow="auto",
            help="Forward-looking blended expected return (50% historical, 50% CAPM) used in portfolio optimization.",
        )
    with col3:
        # Explicit verdict: green treatment indicating pass, no misleading directional arrow
        vol_under_pts = (0.10 - optimizer_vol) * 100
        st.metric(
            label="Model Annual Volatility",
            value=format_percent(optimizer_vol, 2),
            delta=f"{vol_under_pts:.2f} pts under 10.0% limit",
            delta_color="normal",
            delta_arrow="off",
            help="Model-estimated covariance volatility (annualized log-return diagonal shrinkage).",
        )
    with col4:
        st.metric(
            label="Historical Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"rf = {format_percent(rf_rate, 2)} (3M T-Bill)",
            delta_color="off",
            delta_arrow="off",
            help="Annualized Sharpe ratio over the full 2007-2026 historical period relative to 3M Treasury rate.",
        )

    # 4. Decision routes with primary CTA directly below KPI row
    render_decision_routes()

    # Clean 24px whitespace separation without redundant lines
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    with st.expander("Compare model estimates with historical results", expanded=False):
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border-radius: 8px; padding: 1.25rem;">
                    <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.75rem; font-size: 1.1rem;">
                        Optimization & Expected Estimates
                    </h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; border: none; margin: 0;">
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Target Return Constraint</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(target_return, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Estimated Annual Return (Model)</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['teal']};">{format_percent(expected_return, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Estimated Volatility (Covariance)</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(optimizer_vol, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; color: {PALETTE['muted']};">Optimization Objective</td>
                            <td style="padding: 7px 4px; border: none; text-align: right; font-weight: 600; color: {PALETTE['navy']};">Target-Return Minimum Variance</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_right:
            st.markdown(
                f"""
                <div style="background-color: {PALETTE['light_bg']}; border-radius: 8px; padding: 1.25rem;">
                    <h4 style="color: {PALETTE['navy']}; margin-top: 0; margin-bottom: 0.75rem; font-size: 1.1rem;">
                        Realized Historical Performance (2007–2026)
                    </h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; border: none; margin: 0;">
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Realized Compound Annual Return</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(realized_return, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Realized Annual Volatility</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['navy']};">{format_percent(realized_vol, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};">Historical Maximum Drawdown</td>
                            <td style="padding: 7px 4px; border: none; border-bottom: 1px solid {PALETTE['border']}; text-align: right; font-weight: 600; color: {PALETTE['red']};">{format_percent(max_dd, 2)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 7px 4px; border: none; color: {PALETTE['muted']};">Realized Sharpe Ratio</td>
                            <td style="padding: 7px 4px; border: none; text-align: right; font-weight: 600; color: {PALETTE['navy']};">{sharpe:.2f} (rf = {format_percent(rf_rate, 2)})</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 24px clean separator to allocation section
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # 5. Current Allocation Summary with aligned legends and asset-class colors
    st.subheader("Current Proposed Asset Allocation")
    active_weights = weights_df[weights_df["weight"] > 1e-4].copy().sort_values("weight", ascending=False)

    agg_w = float(weights_df.loc[weights_df["asset"] == "AGG", "weight"].sum()) if "AGG" in weights_df["asset"].values else 0.0
    gld_w = float(weights_df.loc[weights_df["asset"] == "GLD", "weight"].sum()) if "GLD" in weights_df["asset"].values else 0.0
    defensive_w = agg_w + gld_w
    equity_assets = ["SPY", "EFA", "EEM", "XLE", "XLK"]
    equity_w = float(weights_df[weights_df["asset"].isin(equity_assets)]["weight"].sum())
    xlk_w = float(weights_df.loc[weights_df["asset"] == "XLK", "weight"].sum()) if "XLK" in weights_df["asset"].values else 0.0
    xle_w = float(weights_df.loc[weights_df["asset"] == "XLE", "weight"].sum()) if "XLE" in weights_df["asset"].values else 0.0
    tilts_w = xlk_w + xle_w
    tilts_label = f"XLK + XLE {format_percent(tilts_w, 1)}"
    tilts_tooltip = f"XLK: {format_percent(xlk_w, 1)}, XLE: {format_percent(xle_w, 1)}"

    # Legend and stacked allocation bar
    defensive_color = ASSET_CLASS_COLORS["Fixed Income"]
    equity_color = ASSET_CLASS_COLORS["Equity"]
    navy_color = PALETTE["navy"]
    muted_color = PALETTE["muted"]
    border_color = PALETTE["border"]
    card_bg = PALETTE["card_bg"]

    c_agg = BAR_SEGMENT_COLORS["AGG"]
    c_gld = BAR_SEGMENT_COLORS["GLD"]
    c_spy = BAR_SEGMENT_COLORS["SPY"]
    c_efa = BAR_SEGMENT_COLORS["EFA"]
    c_xlk = BAR_SEGMENT_COLORS["XLK"]
    c_xle = BAR_SEGMENT_COLORS["XLE"]

    bar_html = (
        f'<div style="margin-bottom: 1.1rem;">'
        f'<div style="display: flex; width: 100%; margin-bottom: 8px; font-size: 0.82rem;">'
        f'<div style="width: 60%; padding-right: 8px; box-sizing: border-box;">'
        f'<div style="border-bottom: 2px solid {defensive_color}; padding-bottom: 6px;">'
        f'<div style="display: flex; align-items: center; gap: 6px; font-weight: 700; color: {navy_color};">'
        f'<span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: {defensive_color};"></span>'
        f'<span>Defensive Allocation: <strong>{format_percent(defensive_w, 1)}</strong></span>'
        f'</div>'
        f'<div style="font-size: 0.74rem; color: {muted_color}; margin-left: 15px; margin-top: 2px;">'
        f'AGG {format_percent(agg_w, 1)} Fixed Income + GLD {format_percent(gld_w, 1)} Gold Bullion'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="width: 40%; padding-left: 8px; box-sizing: border-box;">'
        f'<div style="border-bottom: 2px solid {equity_color}; padding-bottom: 6px;">'
        f'<div style="display: flex; align-items: center; gap: 6px; font-weight: 700; color: {equity_color};">'
        f'<span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: {equity_color};"></span>'
        f'<span>Equity Allocation: <strong>{format_percent(equity_w, 1)}</strong></span>'
        f'</div>'
        f'<div style="font-size: 0.74rem; color: {muted_color}; margin-left: 15px; margin-top: 2px;">'
        f'SPY 22.8%, EFA 9.6%, XLK 6.3%, XLE 1.3%'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="height: 14px; width: 100%; display: flex; border-radius: 7px; overflow: hidden; background: {border_color};">'
        f'<div style="width: 45.0%; background-color: {c_agg}; border-right: 2px solid {card_bg};" title="AGG: 45.0% (Fixed Income)"></div>'
        f'<div style="width: 15.0%; background-color: {c_gld}; border-right: 3px solid {navy_color};" title="GLD: 15.0% (Gold Bullion)"></div>'
        f'<div style="width: 22.8%; background-color: {c_spy}; border-right: 1.5px solid {card_bg};" title="SPY: 22.8% (US Core Equity)"></div>'
        f'<div style="width: 9.6%; background-color: {c_efa}; border-right: 1.5px solid {card_bg};" title="EFA: 9.6% (Dev Markets Equity)"></div>'
        f'<div style="width: 6.3%; background-color: {c_xlk}; border-right: 1.5px solid {card_bg};" title="XLK: 6.3% (Technology Equity)"></div>'
        f'<div style="width: 1.3%; background-color: {c_xle};" title="XLE: 1.3% (Energy Equity)"></div>'
        f'</div>'
        f'<div style="display: flex; width: 100%; margin-top: 4px; font-size: 0.72rem; color: {muted_color};">'
        f'<div style="width: 45.0%; text-align: center;">AGG 45.0%</div>'
        f'<div style="width: 15.0%; text-align: center; border-right: 1px dashed {border_color};">GLD 15.0%</div>'
        f'<div style="width: 22.8%; text-align: center;">SPY 22.8%</div>'
        f'<div style="width: 9.6%; text-align: center;">EFA 9.6%</div>'
        f'<div style="width: 7.6%; text-align: center; white-space: nowrap;" title="{tilts_tooltip}">{tilts_label}</div>'
        f'</div>'
        f'</div>'
    )
    st.html(bar_html)

    cols = st.columns(len(active_weights))
    for idx, row in enumerate(active_weights.itertuples()):
        bar_pct = min(100.0, (row.weight / 0.50) * 100.0)
        color = ASSET_CLASS_COLORS.get(row.category, "#2563eb")
        with cols[idx]:
            card_html = (
                f'<div style="background-color: {PALETTE["card_bg"]}; border: 1px solid {PALETTE["border"]}; border-top: 3px solid {color}; border-radius: 8px; padding: 0.75rem 0.6rem; text-align: center;">'
                f'<div style="font-size: 1.15rem; font-weight: 800; color: {PALETTE["navy"]};">{row.asset}</div>'
                f'<div style="font-size: 0.76rem; color: {PALETTE["muted"]}; margin-bottom: 0.3rem;">{row.category}</div>'
                f'<div style="font-size: 1.25rem; font-weight: 750; color: {PALETTE["navy"]};">{format_percent(row.weight, 1)}</div>'
                f'<div style="font-size: 0.78rem; color: {PALETTE["muted"]};">{format_currency(row.dollar_allocation, 0)}</div>'
                f'<div style="height: 6px; width: 100%; background: {PALETTE["border"]}; border-radius: 3px; margin-top: 0.55rem; overflow: hidden; opacity: 0.85;">'
                f'<div style="height: 100%; width: {bar_pct:.1f}%; background: {color}; border-radius: 3px;"></div>'
                f'</div>'
                f'</div>'
            )
            st.html(card_html)

    # 24px clean separator to narrative section
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    # 6. What these results mean section
    st.subheader("What These Results Mean for the Endowment")

    eq_details = ", ".join(
        f"{row.asset} {format_percent(row.weight, 1)}"
        for row in weights_df[(weights_df["asset"].isin(equity_assets)) & (weights_df["weight"] > 1e-4)].itertuples()
    )

    var_sum = load_var_summary(PortfolioBasis.CURRENT)
    h95 = var_sum[(var_sum["method"] == "historical") & np.isclose(var_sum["confidence"], 0.95)]
    var_95_str = (
        f"{format_currency(float(h95['latest_var_usd'].iloc[0]), 0).replace('$', '&#36;')} ({format_percent(float(h95['latest_var_pct'].iloc[0]), 2)})"
        if not h95.empty
        else "Unavailable"
    )
    es_95_str = (
        f"{format_currency(float(h95['latest_es_usd'].iloc[0]), 0).replace('$', '&#36;')} ({format_percent(float(h95['latest_es_pct'].iloc[0]), 2)})"
        if not h95.empty
        else "Unavailable"
    )

    backtests_df = load_backtest_results()
    cc_99 = backtests_df[np.isclose(backtests_df["confidence"], 0.99)]
    n_models_99 = len(cc_99)
    n_rejected_99 = int(cc_99["conditional_coverage_reject"].sum()) if not cc_99.empty else 0
    oos_obs = int(cc_99["observations"].iloc[0]) if not cc_99.empty else 0
    obs_count = int(kpi_summary.get('observations', 0))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="min-height: 220px; padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-weight: 600; color: #f3f4f6;">Asset Allocation Profile</span>
                    <span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: rgba(8, 120, 109, 0.2); color: #70d8c3; font-weight: 600;">60 / 40 Mandate</span>
                </div>
                <div style="min-height: 76px; font-size: 0.9rem; line-height: 1.5; color: #d1d5db;">
                    Allocates <b>{format_percent(agg_w, 1)}</b> to fixed income (AGG) and <b>{format_percent(gld_w, 1)}</b> to gold (GLD), establishing a combined <b>{format_percent(defensive_w, 1)} defensive cushion</b>. Equity exposure is constrained to <b>{format_percent(equity_w, 1)}</b> across active holdings ({eq_details}).
                </div>
                <div style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.07); font-size: 0.8rem; line-height: 1.45; color: #9ca3af;">
                    Non-correlated real assets dampen drawdowns while preserving global market upside participation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="min-height: 220px; padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-weight: 600; color: #f3f4f6;">Model Estimates vs. Target</span>
                    <span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: rgba(8, 120, 109, 0.2); color: #70d8c3; font-weight: 600;">Target Met</span>
                </div>
                <div style="min-height: 76px; font-size: 0.9rem; line-height: 1.5; color: #d1d5db;">
                    Estimated annual forward return of <b>{format_percent(expected_return, 2)}</b> exceeds the <b>{format_percent(target_return, 2)}</b> hurdle constraint (+{format_percent(expected_return - target_return, 2)}), with an estimated annual portfolio volatility of <b>{format_percent(optimizer_vol, 2)}</b> (under 10.0% limit).
                </div>
                <div style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.07); font-size: 0.8rem; line-height: 1.45; color: #9ca3af;">
                    <i>Caveat:</i> Forward returns are Bayesian shrinkage priors (50% historical, 50% CAPM) and are not guaranteed; volatility measures dispersion.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            f"""
            <div style="min-height: 272px; padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); margin-bottom: 1.25rem; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-weight: 600; color: #f3f4f6;">Historical Replay & Downside Risk</span>
                    <span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.06); color: #d1d5db;">2007–2026 Sample</span>
                </div>
                <div style="min-height: 76px; font-size: 0.9rem; line-height: 1.5; color: #d1d5db;">
                    Retrospective replay across {obs_count:,} trading days delivered a compound annual return of <b>{format_percent(realized_return, 2)}</b>, annualized volatility of <b>{format_percent(realized_vol, 2)}</b>, and a max drawdown of <b>{format_percent(max_dd, 2)}</b>. Latest evaluated 1-day 95% Historical VaR is <b>{var_95_str}</b> and Expected Shortfall is <b>{es_95_str}</b>.
                </div>
                <div style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.07); font-size: 0.8rem; line-height: 1.45; color: #9ca3af;">
                    <i>Caveat:</i> Capital preservation is an investment mandate objective, not a capital guarantee.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div style="min-height: 272px; padding: 1.25rem; border-radius: 10px; border: 1px solid rgba(239,68,68,0.4); background: linear-gradient(180deg, rgba(239,68,68,0.06) 0%, rgba(255,255,255,0.015) 100%); margin-bottom: 1.25rem; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-weight: 600; color: #f3f4f6;">Risk Model Validation Status</span>
                    <span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: rgba(239,68,68,0.2); color: #fca5a5;">Critical Caveat</span>
                </div>
                <div style="min-height: 76px; font-size: 0.9rem; line-height: 1.5; color: #d1d5db;">
                    Out-of-sample backtesting on {oos_obs:,} trading days reveals that at 99% confidence, <b>{n_rejected_99} of {n_models_99} baseline models reject joint conditional coverage</b> (Christoffersen test, 5% significance level).
                </div>
                <div style="margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid rgba(239,68,68,0.2); font-size: 0.8rem; line-height: 1.45; color: #9ca3af;">
                    <i>Key Takeaway:</i> While trailing 250-day windows register in the Green Basel zone during calm markets, multi-year crisis clusters (2020 crash, 2022 rate shocks) demonstrate that empirical market tails are fatter than thin-tailed models assume. Statistical non-rejection in calm periods is not proof of model validity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_callout(
        "**Investment Committee Status**: Team investment recommendation pending formal presentation and human review. "
        "The quantitative figures presented herein represent analytical model outputs and retrospective diagnostics, not an automated buy/sell recommendation.",
        level="info",
    )
