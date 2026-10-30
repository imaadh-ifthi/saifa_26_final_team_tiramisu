"""Reusable UI components and cards for Streamlit institutional layout."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from .schemas import PALETTE


def render_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean institutional page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 0.85rem; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 0.55rem;">
            <h1 style="color: {PALETTE['navy']}; margin: 0; font-weight: 750; font-size: clamp(1.6rem, 2.2vw, 2.1rem); letter-spacing: -0.03em; line-height: 1.2;">{title}</h1>
            {f'<p style="color: {PALETTE["muted"]}; margin: 0.35rem 0 0 0; font-size: 0.95rem; line-height: 1.45;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_basis_badge(basis_label: str, cutoff_date: str | None = None) -> None:
    """Render a persistent badge indicating the analytical basis of displayed metrics."""
    # Neutral muted gray fill, no bright outline, so only green on page means good performance
    if "(" in basis_label:
        main_part, paren = basis_label.split("(", 1)
        paren_html = f'<span style="color: {PALETTE["muted"]}; font-weight: 400;">({paren}</span>'
    else:
        main_part = basis_label
        paren_html = ''

    cutoff_html = (
        f'<div style="display: inline-flex; align-items: center; background-color: {PALETTE["light_bg"]}; '
        f'color: {PALETTE["muted"]}; border: 1px solid {PALETTE["border"]}; border-radius: 6px; '
        f'padding: 0.25rem 0.65rem; font-size: 0.8rem; font-weight: 500;">'
        f'Latest Data Cutoff:&nbsp;<span style="color: {PALETTE["navy"]}; font-weight: 600;">{cutoff_date}</span>'
        f'</div>'
        if cutoff_date else ''
    )
    badge_html = (
        f'<div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-bottom: 1.1rem;">'
        f'<div style="display: inline-flex; align-items: center; background-color: {PALETTE["light_bg"]}; color: {PALETTE["muted"]}; border: 1px solid {PALETTE["border"]}; border-radius: 6px; padding: 0.25rem 0.65rem; font-size: 0.8rem; font-weight: 500;">'
        f'Analysis Basis:&nbsp;<span style="color: {PALETTE["navy"]}; font-weight: 650;">{main_part.strip()}</span>&nbsp;{paren_html}'
        f'</div>'
        f'{cutoff_html}'
        f'</div>'
    )
    st.html(badge_html)


def render_callout(text: str, level: str = "info") -> None:
    """Render an institutional callout alert box with neutral fill, colored left accent bar, and normal body text."""
    accent_color = {
        "warning": PALETTE["amber"],
        "danger": PALETTE["red"],
        "error": PALETTE["red"],
        "success": PALETTE["green"],
    }.get(level, PALETTE["teal"])

    import re
    html_text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"<b>{m.group(1)}</b>", text)
    html_text = re.sub(r"\*(.+?)\*", lambda m: f"<i>{m.group(1)}</i>", html_text)
    html_text = re.sub(r"`(.+?)`", lambda m: f"<code style='background: rgba(0,0,0,0.06); padding: 1px 4px; border-radius: 4px; font-size: 0.85em;'>{m.group(1)}</code>", html_text)

    st.markdown(
        f"""
        <div style="background-color: {PALETTE['light_bg']}; border-left: 3px solid {accent_color}; padding: 0.65rem 0.95rem; border-radius: 0 6px 6px 0; margin: 0.65rem 0 0.85rem 0; font-size: 0.88rem; color: {PALETTE['navy']}; line-height: 1.5;">
            {html_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    elif status_lower == "not_applicable":
        bg, color, label = PALETTE["light_bg"], PALETTE["muted"], "NOT APPLICABLE (99% Only)"
    else:
        bg, color, label = PALETTE["light_bg"], PALETTE["muted"], traffic_status.upper()

    obs_text = f" · {exceptions} breaches in trailing {observations} days" if exceptions is not None and observations is not None else ""

    st.markdown(
        f"""
        <div style="display: inline-block; background-color: {bg}; color: {color}; border: 1px solid {color}; border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.85rem; font-weight: 700; margin: 0.5rem 0;">
            Basel Traffic Light (99% / 250-Day): {label}{obs_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_button(df: pd.DataFrame, filename: str, label: str = "Download Table as CSV", icon: str | None = ":material/download:") -> None:
    """Render a clean CSV download button."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        icon=icon,
    )
