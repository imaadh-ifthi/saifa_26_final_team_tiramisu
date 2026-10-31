"""Interactive Plotly charts for the Serendib Endowment Dashboard."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .schemas import PALETTE, ASSET_REGISTRY, METHOD_LABELS
from .theme import theme_name
from .formatting import format_currency, format_percent


def apply_institutional_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent institutional styling to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": PALETTE["navy"], "family": "Inter, sans-serif"},
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_dark" if theme_name() == "dark" else "plotly_white",
        paper_bgcolor=PALETTE["card_bg"],
        plot_bgcolor=PALETTE["light_bg"],
        font={"family": "Inter, sans-serif", "color": PALETTE["navy"], "size": 12},
        height=height,
        margin={"l": 50, "r": 30, "t": 45 if title else 25, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11},
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["border"],
        zeroline=True,
        zerolinecolor=PALETTE["muted"],
        tickfont={"size": 11},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["border"],
        zeroline=True,
        zerolinecolor=PALETTE["muted"],
        tickfont={"size": 11},
    )
    return fig


def plot_allocation_donut(weights_df: pd.DataFrame) -> go.Figure:
    """Donut chart of asset weights with asset names and dollar values in hover."""
    display_df = weights_df[weights_df["weight"] > 1e-6].copy()
    colors = [ASSET_REGISTRY.get(a, None).color if a in ASSET_REGISTRY else "#64748b" for a in display_df["asset"]]

    hover_text = [
        f"<b>{row.asset}</b> - {getattr(row, 'name', row.asset)}<br>"
        f"Category: {getattr(row, 'category', 'Other')}<br>"
        f"Weight: {format_percent(row.weight, 2)}<br>"
        f"Dollar Notional: {format_currency(row.dollar_allocation, 0)}"
        for row in display_df.itertuples()
    ]

    fig = go.Figure(data=[
        go.Pie(
            labels=display_df["asset"],
            values=display_df["weight"],
            hole=0.55,
            marker={"colors": colors, "line": {"color": PALETTE["card_bg"], "width": 2}},
            textinfo="label+percent",
            hovertext=hover_text,
            hoverinfo="text",
            textfont={"size": 12},
            sort=False,
            automargin=True,
        )
    ])
    fig.update_layout(
        font={"color": PALETTE["navy"]},
        showlegend=True,
        template="plotly_dark" if theme_name() == "dark" else "plotly_white",
        paper_bgcolor=PALETTE["card_bg"],
        height=380,
        margin={"l": 20, "r": 20, "t": 20, "b": 65},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.06,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    return fig


def plot_cumulative_wealth(perf_df: pd.DataFrame, notional: float = 10_000_000.0) -> go.Figure:
    """Growth of $10M: Portfolio vs Policy Benchmark over time."""
    fig = go.Figure()

    port_wealth = perf_df["portfolio_wealth"] * notional
    fill_col = "rgba(112, 216, 195, 0.08)" if theme_name() == "dark" else "rgba(8, 120, 109, 0.06)"
    fig.add_trace(go.Scatter(
        x=perf_df["date"],
        y=port_wealth,
        mode="lines",
        name="Portfolio Value",
        fill="tozeroy",
        fillcolor=fill_col,
        line={"color": PALETTE["teal"], "width": 2.2},
        hovertemplate="Portfolio: %{y:$,.0f}<extra></extra>",
    ))

    max_wealth = float(port_wealth.max())
    min_wealth = float(port_wealth.min())

    if "policy_wealth" in perf_df.columns:
        policy_wealth = perf_df["policy_wealth"] * notional
        max_wealth = max(max_wealth, float(policy_wealth.max()))
        min_wealth = min(min_wealth, float(policy_wealth.min()))
        fig.add_trace(go.Scatter(
            x=perf_df["date"],
            y=policy_wealth,
            mode="lines",
            name="Policy Benchmark (50/40/10: 35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD)",
            line={"color": PALETTE["amber"], "width": 1.8, "dash": "dash"},
            hovertemplate="Benchmark: %{y:$,.0f}<extra></extra>",
        ))

    # Clamp y-axis bottom to eliminate $0.0 dead space from fill='tozeroy'
    y_bottom = min(notional * 0.75, min_wealth * 0.95)
    y_top = max_wealth * 1.05

    fig = apply_institutional_layout(fig, "Portfolio Wealth Trajectory (USD 10 Million Initial Notional)", height=390)
    fig.update_yaxes(
        title="Portfolio Value",
        tickformat="$.2s",
        range=[y_bottom, y_top],
    )
    fig.update_xaxes(title=None)
    return fig


def plot_drawdowns(perf_df: pd.DataFrame) -> go.Figure:
    """Historical drawdown (underwater) comparison."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf_df["date"],
        y=perf_df["portfolio_drawdown"] * 100.0,
        mode="lines",
        name="Portfolio Drawdown",
        fill="tozeroy",
        line={"color": PALETTE["red"], "width": 1.5},
        fillcolor="rgba(220, 38, 38, 0.15)",
        hovertemplate="Portfolio Drawdown: %{y:.2f}%<extra></extra>",
    ))

    if "policy_drawdown" in perf_df.columns:
        fig.add_trace(go.Scatter(
            x=perf_df["date"],
            y=perf_df["policy_drawdown"] * 100.0,
            mode="lines",
            name="Policy Benchmark Drawdown",
            line={"color": PALETTE["slate"], "width": 1.2, "dash": "dot"},
            hovertemplate="Benchmark Drawdown: %{y:.2f}%<extra></extra>",
        ))

    fig = apply_institutional_layout(fig, "Portfolio Drawdown from Historical Peak (%)", height=320)
    fig.update_yaxes(title="Drawdown (%)", ticksuffix="%")
    fig.update_xaxes(title=None)
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    selected_target: float = 0.065,
    perf_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Plot risk-return profile with constituent assets, policy benchmark, and mandate boundaries."""
    fig = go.Figure()

    # 1. Plot individual constituent assets for risk-return context
    try:
        from .data_access import load_expected_returns, load_covariance_matrix
        exp_ret_df = load_expected_returns()
        cov_df = load_covariance_matrix()
        asset_vols = [float(np.sqrt(cov_df.loc[a, a])) * 100.0 for a in exp_ret_df["asset"]]
        asset_rets = [float(r) * 100.0 for r in exp_ret_df["expected_return"]]
        asset_colors = [ASSET_REGISTRY[a].color if a in ASSET_REGISTRY else "#64748b" for a in exp_ret_df["asset"]]

        fig.add_trace(go.Scatter(
            x=asset_vols,
            y=asset_rets,
            mode="markers+text",
            text=exp_ret_df["asset"].tolist(),
            textposition=["top right", "bottom right", "top left", "bottom right", "top right", "top left", "top left"],
            marker={"size": 10, "color": asset_colors, "line": {"color": PALETTE["card_bg"], "width": 1.5}},
            name="Constituent Assets",
            hovertemplate="<b>%{text}</b><br>Expected Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>",
        ))
    except Exception:
        pass

    # 2. Plot Policy Benchmark if performance series is provided
    if perf_df is not None and len(perf_df) > 0 and "policy_wealth" in perf_df.columns:
        years = len(perf_df) / 252.0
        bench_cagr = (float(perf_df["policy_wealth"].iloc[-1]) / float(perf_df["policy_wealth"].iloc[0])) ** (1.0 / years) - 1.0
        bench_vol = float(perf_df["policy_return"].std() * np.sqrt(252))
        fig.add_trace(go.Scatter(
            x=[bench_vol * 100.0],
            y=[bench_cagr * 100.0],
            mode="markers+text",
            text=["Policy Benchmark"],
            textposition="top right",
            marker={"size": 12, "color": PALETTE["amber"], "symbol": "square", "line": {"color": PALETTE["card_bg"], "width": 1.5}},
            name="Policy Benchmark (50/40/10)",
            hovertemplate="<b>Policy Benchmark (50/40/10)</b><br>Historical Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>",
        ))

    vols = frontier_df["volatility"].values
    returns = frontier_df["expected_return"].values
    is_collapsed = np.allclose(vols, vols[0], atol=1e-5) and np.allclose(returns, returns[0], atol=1e-5)

    # 3. Plot Constrained Minimum-Variance Solution
    if is_collapsed:
        fig.add_trace(go.Scatter(
            x=[vols[0] * 100.0],
            y=[returns[0] * 100.0],
            mode="markers+text",
            marker={"size": 16, "color": PALETTE["teal"], "symbol": "diamond", "line": {"color": PALETTE["card_bg"], "width": 2}},
            text=["Optimal Portfolio"],
            textposition="top center",
            name="Optimal Constrained Portfolio",
            hovertemplate=(
                f"<b>Optimal Constrained Minimum-Variance Portfolio</b><br>"
                f"Expected Return: {returns[0]*100:.2f}%<br>"
                f"Annual Volatility: {vols[0]*100:.2f}%<extra></extra>"
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=frontier_df["volatility"] * 100.0,
            y=frontier_df["expected_return"] * 100.0,
            mode="lines+markers",
            name="Efficient Frontier",
            line={"color": PALETTE["teal"], "width": 2},
            marker={"size": 8, "color": PALETTE["teal"]},
            hovertemplate="Vol: %{x:.2f}%<br>Exp Return: %{y:.2f}%<extra></extra>",
        ))

    # 4. Reference lines for institutional mandate bounds
    fig.add_hline(
        y=selected_target * 100.0,
        line_dash="dash",
        line_color=PALETTE["amber"],
        line_width=1.5,
        annotation_text=f"Mandate Target ({selected_target*100:.1f}%)",
        annotation_position="top right",
        annotation_yshift=6,
        annotation_font={"size": 10, "color": PALETTE["amber"]},
    )
    fig.add_vline(
        x=10.0,
        line_dash="dash",
        line_color=PALETTE["red"],
        line_width=1.5,
        annotation_text="Volatility Limit (10.0%)",
        annotation_position="top left",
        annotation_xshift=-6,
        annotation_font={"size": 10, "color": PALETTE["red"]},
    )

    fig = apply_institutional_layout(fig, "Risk-Return Profile & Minimum Variance Solution", height=390)
    fig.update_layout(
        margin={"l": 50, "r": 20, "t": 40, "b": 65},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.update_xaxes(title="Annualized Volatility (%)", ticksuffix="%")
    fig.update_yaxes(title="Expected Annual Return (%)", ticksuffix="%")
    return fig


def plot_var_es_time_series(forecasts_df: pd.DataFrame, method: str, confidence: float) -> go.Figure:
    """VaR and ES over time for selected method and confidence level."""
    subset = forecasts_df[(forecasts_df["method"] == method) & np.isclose(forecasts_df["confidence"], confidence)].copy()
    subset = subset.sort_values("date").reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["var_usd"],
        mode="lines",
        name=f"1-Day VaR ({int(confidence*100)}%)",
        line={"color": PALETTE["teal"], "width": 2.0},
        hovertemplate="VaR: %{y:$,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["es_usd"],
        mode="lines",
        name=f"1-Day Expected Shortfall ({int(confidence*100)}%)",
        line={"color": PALETTE["red"], "width": 2.0},
        hovertemplate="ES: %{y:$,.0f}<extra></extra>",
    ))

    fig = apply_institutional_layout(fig, "", height=360)
    fig.update_layout(
        font={"family": "Inter, sans-serif", "size": 11, "color": PALETTE["navy"]},
        margin={"l": 55, "r": 25, "t": 25, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
    )
    fig.update_yaxes(
        title="1-Day Loss Forecast (USD)",
        tickformat="$,.0f",
        title_font={"size": 11, "family": "Inter, sans-serif", "color": PALETTE["navy"]},
        tickfont={"size": 10, "family": "Inter, sans-serif"},
    )
    fig.update_xaxes(
        title="Evaluation Date",
        title_font={"size": 11, "family": "Inter, sans-serif", "color": PALETTE["navy"]},
        tickfont={"size": 10, "family": "Inter, sans-serif"},
    )
    return fig


def plot_var_method_comparison_bar(summary_df: pd.DataFrame, confidence: float) -> go.Figure:
    """Side-by-side comparison of latest VaR and ES across baseline methods."""
    subset = summary_df[np.isclose(summary_df["confidence"], confidence)].copy()
    subset["display_name"] = subset["method"].map(lambda m: METHOD_LABELS.get(m, m))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=subset["display_name"],
        y=subset["latest_var_usd"],
        name="Value at Risk (VaR)",
        marker_color=PALETTE["teal"],
        hovertemplate="VaR: %{y:$,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=subset["display_name"],
        y=subset["latest_es_usd"],
        name="Expected Shortfall (ES)",
        marker_color=PALETTE["red"],
        hovertemplate="ES: %{y:$,.0f}<extra></extra>",
    ))

    fig = apply_institutional_layout(fig, "", height=360)
    fig.update_layout(
        barmode="group",
        font={"family": "Inter, sans-serif", "size": 11, "color": PALETTE["navy"]},
        margin={"l": 55, "r": 25, "t": 25, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"family": "Inter, sans-serif", "size": 11, "color": PALETTE["navy"]},
        },
    )
    fig.update_yaxes(
        title="1-Day Loss Forecast (USD)",
        tickformat="$,.0f",
        title_font={"size": 11, "family": "Inter, sans-serif", "color": PALETTE["navy"]},
        tickfont={"size": 10, "family": "Inter, sans-serif"},
    )
    fig.update_xaxes(
        title_font={"size": 11, "family": "Inter, sans-serif", "color": PALETTE["navy"]},
        tickfont={"size": 10, "family": "Inter, sans-serif"},
    )
    return fig


def plot_validation_breaches(forecasts_df: pd.DataFrame, method: str, confidence: float) -> go.Figure:
    """
    Formal Out-of-Sample model validation chart:
    - Positive loss convention: realised loss = actual_loss_usd (positive = loss, negative = gain)
    - VaR forecast line = var_usd (positive threshold)
    - Breach markers = red diamonds when actual_loss_usd > var_usd (exception == True)
    - Gains visible as negative losses below $0 line, NOT clipped away.
    """
    subset = forecasts_df[(forecasts_df["method"] == method) & np.isclose(forecasts_df["confidence"], confidence)].copy()
    subset = subset.sort_values("date").reset_index(drop=True)

    fig = go.Figure()

    # Realised P&L as losses (positive = loss, negative = gain)
    fig.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["actual_loss_usd"],
        mode="lines",
        name="Realized Daily Loss / (Gain)",
        line={"color": PALETTE["slate"], "width": 1.0},
        hovertemplate="Realized Loss: %{y:$,.0f}<extra></extra>",
    ))

    # VaR threshold line
    fig.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["var_usd"],
        mode="lines",
        name=f"Forecast VaR ({int(confidence*100)}%)",
        line={"color": PALETTE["teal"], "width": 2.0},
        hovertemplate="VaR Threshold: %{y:$,.0f}<extra></extra>",
    ))

    # Highlight breaches
    breaches = subset[subset["exception"] == True]
    if not breaches.empty:
        fig.add_trace(go.Scatter(
            x=breaches["date"],
            y=breaches["actual_loss_usd"],
            mode="markers",
            name=f"VaR Breaches ({len(breaches)} exceptions)",
            marker={"size": 8, "color": PALETTE["red"], "symbol": "diamond-open", "line": {"width": 2, "color": PALETTE["red"]}},
            hovertemplate="<b>BREACH</b><br>Date: %{x|%Y-%m-%d}<br>Realized Loss: %{y:$,.0f}<extra></extra>",
        ))

    method_label = METHOD_LABELS.get(method, method)
    fig = apply_institutional_layout(
        fig,
        f"Model Validation: Realized Losses vs VaR Threshold ({method_label}, {int(confidence*100)}%)",
        height=420,
    )
    fig.update_layout(
        margin=dict(t=70, b=45, l=50, r=30),
        title=dict(y=0.96),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255, 255, 255, 0.6)",
        ),
    )
    fig.update_yaxes(title="Realized Loss / (Gain) in USD", tickformat="$,.0f")
    fig.update_xaxes(title="Evaluation Date")
    return fig


def plot_stress_paths(paths_df: pd.DataFrame, scenario_name: str, notional: float = 10_000_000.0) -> go.Figure:
    """Replay trajectory of portfolio value and drawdown during historical crisis."""
    scenario_path = paths_df[paths_df["scenario"] == scenario_name].copy()
    if scenario_path.empty:
        fig = go.Figure()
        return apply_institutional_layout(fig, f"No path data available for {scenario_name}")

    scenario_path = scenario_path.sort_values("date").reset_index(drop=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Portfolio Value (USD)", "Drawdown (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=scenario_path["date"],
            y=scenario_path["wealth"] * notional,
            mode="lines",
            name="Portfolio Wealth",
            line={"color": PALETTE["navy"], "width": 2.0},
            hovertemplate="Value: %{y:$,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=scenario_path["date"],
            y=scenario_path["drawdown"] * 100.0,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"color": PALETTE["red"], "width": 1.5},
            fillcolor="rgba(220, 38, 38, 0.15)",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=450,
        template="plotly_dark" if theme_name() == "dark" else "plotly_white",
        paper_bgcolor=PALETTE["card_bg"],
        font={"family": "Inter, sans-serif", "color": PALETTE["navy"]},
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 40, "b": 30},
    )
    fig.update_yaxes(title_text="USD Value", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%", row=2, col=1)
    return fig


def plot_hypothetical_shocks(contributions_df: pd.DataFrame) -> go.Figure:
    """Position-level P&L contributions ($) for assumed hypothetical scenario."""
    df = contributions_df.copy()
    df["display_name"] = df["asset"].map(lambda a: f"{a} ({ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a})")
    colors = [PALETTE["green"] if val >= 0 else PALETTE["red"] for val in df["pnl_contribution_usd"]]

    fig = go.Figure(data=[
        go.Bar(
            y=df["display_name"],
            x=df["pnl_contribution_usd"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Asset: %{y}<br>"
                "P&L Contribution: %{x:$,.0f}<extra></extra>"
            ),
        )
    ])
    fig = apply_institutional_layout(fig, "Position-Level Dollar P&L Contribution Under Assumed Stagflation Shock", height=350)
    fig.update_layout(
        margin=dict(l=190, r=30, t=50, b=45),
    )
    fig.update_xaxes(title="P&L Contribution (USD)", tickformat="$,.0f")
    fig.update_yaxes(automargin=True, autorange="reversed")
    return fig


def plot_risk_contributions_diverging(contributions_df: pd.DataFrame, metric: str = "component_var_usd", confidence: float = 0.95, method: str = "historical_simple") -> go.Figure:
    """
    Diverging horizontal bar chart preserving negative risk contributions.
    Negative values indicate hedging assets (e.g. bonds or energy during certain horizons).
    """
    subset = contributions_df[(contributions_df["method"] == method) & np.isclose(contributions_df["confidence"], confidence)].copy()
    if subset.empty:
        fig = go.Figure()
        return apply_institutional_layout(fig, "No risk contribution data found")

    subset["display_name"] = subset["asset"].map(lambda a: f"{a} - {ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a}")
    values = subset[metric].values
    colors = [PALETTE["red"] if v > 0 else PALETTE["green"] for v in values]

    metric_title = "Dollar Component VaR (USD)" if "var" in metric else "Dollar Component ES (USD)"
    component_type = "Component VaR" if "var" in metric else "Component ES"

    fig = go.Figure(data=[
        go.Bar(
            y=subset["display_name"],
            x=values,
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Asset: %{y}<br>"
                f"{metric_title}: %{{x:$,.0f}}<extra></extra>"
            ),
        )
    ])
    method_title = "Historical Empirical Tail" if "historical" in method else "Parametric Normal (Simple Returns)"
    fig = apply_institutional_layout(
        fig,
        f"Euler {component_type} Risk Contribution by Position ({method_title}, {int(confidence*100)}% Confidence)",
        height=360,
    )
    fig.update_layout(
        margin=dict(l=190, r=30, t=50, b=45),
    )
    fig.update_xaxes(title=metric_title, tickformat="$,.0f")
    fig.update_yaxes(automargin=True, autorange="reversed")
    return fig
