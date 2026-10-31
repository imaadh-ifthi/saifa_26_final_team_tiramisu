"""TiraMisU identity and accessible presentation. No analytical changes."""
import streamlit as st
from .theme import PALETTE as P

TAGLINE = "Thoughtful allocation. Transparent risk."
PAGES = ["1. Overview", "2. Portfolio & Performance", "3. Risk Analysis",
         "4. Model Validation", "5. Stress & Risk Contributions", "6. Assumptions & Data Quality"]
PAGE_LABELS = ["01  Executive overview", "02  Allocation & performance", "03  Risk explorer",
               "04  Model reliability", "05  Crisis & contributions", "06  Evidence & assumptions"]

def apply_design():
    st.html(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;750&display=swap');
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}
    .stApp {{ --tm-ink:{P['navy']}; --tm-muted:{P['muted']}; --tm-line:{P['border']}; }}
    .stMainBlockContainer {{max-width:1360px;padding-top:3.5rem;padding-bottom:2.5rem;}}
    h1,h2,h3,h4 {{letter-spacing:-.025em;color:{P['navy']};}}
    p,li {{line-height:1.6;}}
    [data-testid="stSidebar"] {{border-right:1px solid {P['border']};}}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:.4rem;}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{padding:.45rem .55rem;border-radius:8px;transition:background 0.15s, border-left 0.15s;}}
    /* Hide the radio circle inside the label wrapper while keeping stMarkdownContainer visible */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child > div:first-child:not([data-testid="stMarkdownContainer"]),
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {{
      display: none !important;
    }}
    /* Active sidebar item with distinct 3px left bar */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
      background:{P['teal_light']} !important;
      border-left:3px solid {P['teal']} !important;
      padding-left:0.75rem !important;
      font-weight:650 !important;
    }}
    [data-testid="stMetric"] {{
      background:{P['light_bg']};
      border:1px solid {P['border']};
      border-radius:12px;
      padding:0.85rem 1rem;
      min-height:112px;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
      box-sizing:border-box;
    }}
    [data-testid="stMetricLabel"] {{
      font-size:0.82rem;
      color:{P['muted']};
      font-weight:500;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
    }}
    [data-testid="stMetricValue"] {{
      font-variant-numeric:tabular-nums;
      letter-spacing:-.035em;
      font-size:2.15rem !important;
      font-weight:750 !important;
      line-height:1.15 !important;
      color:{P['navy']};
    }}
    [data-testid="stMetricDelta"] {{
      font-size:0.82rem;
      font-weight:500;
      line-height:1.3;
    }}
    /* Primary button: Dark text (#1a1714) on mint */
    button[kind="primary"],
    button[kind="primary"] *,
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] * {{
      background-color:{P['teal']} !important;
      color:#1a1714 !important;
      border-color:{P['teal']} !important;
      font-weight:750 !important;
      border-radius:8px !important;
    }}
    button[kind="primary"]:hover,
    button[kind="primary"]:hover *,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover * {{
      background-color:{P['navy']} !important;
      border-color:{P['navy']} !important;
      color:{P['card_bg']} !important;
    }}
    button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{
      background-color:transparent !important;
      color:{P['navy']} !important;
      border:1px solid {P['border']} !important;
      font-weight:500 !important;
      border-radius:8px !important;
    }}
    button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
      border-color:{P['teal']} !important;
      color:{P['teal']} !important;
    }}
    [data-testid="stTabs"] [role="tablist"] {{gap:.5rem;border-bottom:1px solid {P['border']};}}
    [data-testid="stTabs"] [role="tab"] {{
      padding:.6rem 1rem;
      border:none !important;
      border-bottom:2px solid transparent !important;
      color:{P['muted']} !important;
      font-weight:500 !important;
      transition:color 0.15s ease-in-out;
    }}
    [data-testid="stTabs"] [role="tab"]:focus:not(:focus-visible) {{
      outline:none !important;
      box-shadow:none !important;
      border-left:none !important;
      border-right:none !important;
      border-top:none !important;
    }}
    [data-testid="stTabs"] [role="tab"]:focus-visible {{
      outline:2px solid {P['teal']} !important;
      outline-offset:2px !important;
      border-radius:4px !important;
    }}
    [data-testid="stTabs"] [role="tab"] p,
    [data-testid="stTabs"] [role="tab"] span {{
      color:{P['muted']} !important;
      font-weight:500 !important;
    }}
    [data-testid="stTabs"] [role="tab"]:hover p,
    [data-testid="stTabs"] [role="tab"]:hover span,
    [data-testid="stTabs"] [role="tab"]:hover {{
      color:{P['navy']} !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
      color:{P['teal']} !important;
      font-weight:650 !important;
      border-bottom:2px solid {P['teal']} !important;
      border-left:none !important;
      border-right:none !important;
      border-top:none !important;
      box-shadow:none !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] span {{
      color:{P['teal']} !important;
      font-weight:650 !important;
    }}
    /* Neutral gray chip for inline code - prevents bright green clash */
    code, [data-testid="stMarkdownContainer"] code {{
      color:{P['slate']} !important;
      background-color:{P['light_bg']} !important;
      border:1px solid {P['border']} !important;
      padding:0.12rem 0.35rem !important;
      border-radius:4px !important;
      font-size:0.85em !important;
      font-weight:500 !important;
    }}
    button:focus-visible,a:focus-visible {{outline:3px solid {P['teal']};outline-offset:3px;}}
    .tm-brand {{font-size:2rem;font-weight:750;letter-spacing:-.07em;line-height:1.2;color:{P['navy']};}}
    .tm-brand span {{color:{P['teal']};}}
    .tm-school {{font-size:.78rem;letter-spacing:.02em;color:{P['muted']};margin:.2rem 0 .6rem;}}
    .tm-page-header {{margin-bottom:0.75rem;}}
    .tm-eyebrow {{color:{P['teal']};font-size:.74rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:0.2rem;}}
    .tm-enter {{animation:tm-arrive .5s ease-out both;}}
    @keyframes tm-arrive {{from {{transform:translateY(-8px);opacity:0;}} to {{transform:none;opacity:1;}}}}
    @media(prefers-reduced-motion:reduce) {{.tm-enter {{animation:none;}}}}
    @media(max-width:700px) {{.stMainBlockContainer {{padding:.85rem;}}
      [data-testid="stMetric"] {{min-height:95px;padding:.65rem;}}}}
    [data-testid="stExpander"] table {{ border: none !important; }}
    [data-testid="stExpander"] td, [data-testid="stExpander"] th {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
    /* Default markdown tables: clean horizontal look without breaking custom borders */
    .stMarkdown table {{
      border: none;
      border-collapse: collapse;
    }}
    .stMarkdown td, .stMarkdown th {{
      border-left: none;
      border-right: none;
    }}
    a.header-anchor, [data-testid="stHeaderActionElements"] {{ display: none !important; }}
    </style>""")

def render_brand():
    st.html('<div class="tm-brand">Tira<span>MisU</span></div>'
            '<div class="tm-school">University of Peradeniya<br>Quant Edge 1.0</div>')

def render_intro():
    first = not st.session_state.get("tm_intro_seen", False)
    st.session_state.tm_intro_seen = True
    cls = "tm-page-header tm-enter" if first else "tm-page-header"
    st.markdown(f'<div class="{cls}">'
                f'<div class="tm-eyebrow">TiraMisU · University of Peradeniya</div>'
                f'<h1 style="color:{P["navy"]};font-size:clamp(1.5rem, 2.2vw, 2.1rem);font-weight:750;letter-spacing:-0.03em;margin:0.15rem 0 0.15rem 0;line-height:1.15;">Executive Overview</h1>'
                f'<p style="color:{P["muted"]};font-size:0.95rem;line-height:1.45;margin:0 0 0.75rem 0;max-width:850px;">Portfolio construction, downside risk measurement and validation for the Serendib University Endowment.</p>'
                f'</div>',
                unsafe_allow_html=True)

def _navigate(page):
    st.session_state.tm_navigation = page

def render_decision_routes():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Explore the allocation →", key="route_"+PAGES[1], on_click=_navigate, args=(PAGES[1],), type="primary", width="stretch")
    with c2:
        st.button("Examine crisis losses →", key="route_"+PAGES[4], on_click=_navigate, args=(PAGES[4],), type="secondary", width="stretch")
    with c3:
        st.button("Check model reliability →", key="route_"+PAGES[3], on_click=_navigate, args=(PAGES[3],), type="secondary", width="stretch")
    st.markdown(
        f'<div style="font-size:0.78rem;color:{P["muted"]};margin-top:8px;margin-bottom:0;">'
        'Saved analytical results · Current-allocation replay and frozen validation are separate analyses.'
        '</div>',
        unsafe_allow_html=True,
    )
