"""Reusable UI components and cards for Streamlit institutional layout."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from .schemas import PALETTE


def render_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean institutional page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 0.75rem;">
            <h2 style="color: {PALETTE['navy']}; margin: 0; font-weight: 700; font-size: 1.75rem; letter-spacing: -0.02em;">{title}</h2>
            {f'<p style="color: {PALETTE["muted"]}; margin: 0.35rem 0 0 0; font-size: 0.95rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_basis_badge(basis_label: str) -> None:
    """Render a persistent badge indicating the analytical basis of displayed metrics."""
    is_oos = "Out-of-sample" in basis_label or "Frozen" in basis_label
    bg_color = PALETTE["blue_light"] if is_oos else PALETTE["teal_light"]
    text_color = PALETTE["blue"] if is_oos else PALETTE["teal"]
    border_color = PALETTE["blue"] if is_oos else PALETTE["teal"]

    st.markdown(
        f"""
        <div style="display: inline-block; background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 0.25rem 0.6rem; font-size: 0.82rem; font-weight: 600; margin-bottom: 1rem;">
            Analysis Basis: {basis_label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(text: str, level: str = "info") -> None:
    """Render an institutional callout alert box."""
    if level == "warning":
        bg, border, text_color, icon = PALETTE["amber_light"], PALETTE["amber"], PALETTE["amber"], "⚠️"
    elif level == "danger" or level == "error":
        bg, border, text_color, icon = PALETTE["red_light"], PALETTE["red"], PALETTE["red"], "🚨"
    elif level == "success":
        bg, border, text_color, icon = PALETTE["green_light"], PALETTE["green"], PALETTE["green"], "✓"
    else:  # info
        bg, border, text_color, icon = PALETTE["light_bg"], PALETTE["border"], PALETTE["navy"], "ℹ️"

    renderer = {"warning": st.warning, "danger": st.error, "error": st.error,
                "success": st.success}.get(level, st.info)
    renderer(text)


def render_traffic_light_badge(traffic_status: str, exceptions: int | None = None, observations: int | None = None) -> None:
    """Render Basel traffic light indicator with regulatory context."""
    status_lower = str(traffic_status).lower()
    if status_lower == "green":
        bg, color, label = PALETTE["green_light"], PALETTE["green"], "GREEN (0–4 Exceptions)"
    elif status_lower in ("yellow", "amber"):
        bg, color, label = PALETTE["amber_light"], PALETTE["amber"], "YELLOW / AMBER (5–9 Exceptions)"
    elif status_lower == "red":
        bg, color, label = PALETTE["red_light"], PALETTE["red"], "RED (10+ Exceptions)"
    elif status_lower == "insufficient_data":
        bg, color, label = PALETTE["light_bg"], PALETTE["muted"], "INSUFFICIENT DATA (<250 observations)"
    else:
        bg, color, label = PALETTE["light_bg"], PALETTE["muted"], f"NOT APPLICABLE ({traffic_status})"

    obs_text = f" ({exceptions} breaches in last {observations} days)" if exceptions is not None and observations is not None else ""

    st.markdown(
        f"""
        <div style="display: inline-block; background-color: {bg}; color: {color}; border: 1px solid {color}; border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.85rem; font-weight: 700; margin: 0.5rem 0;">
            Basel Traffic Light (99% / 250-Day): {label}{obs_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_button(df: pd.DataFrame, filename: str, label: str = "Download Table as CSV") -> None:
    """Render a clean CSV download button."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
