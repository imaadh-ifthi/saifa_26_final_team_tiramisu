"""TiraMisU identity and accessible presentation. No analytical changes."""
import streamlit as st
from .theme import PALETTE as P

TAGLINE = "Thoughtful allocation. Transparent risk."
PAGES = ["1. Overview", "2. Portfolio & Performance", "3. Risk Analysis",
         "4. Model Validation", "5. Stress & Risk Contributions", "6. Assumptions & Data Quality"]
PAGE_LABELS = ["01  Investment briefing", "02  Allocation & performance", "03  Risk explorer",
               "04  Model reliability", "05  Crisis & contributions", "06  Evidence & assumptions"]

def apply_design():
    st.html(f"""<style>
    .stApp {{ --tm-ink:{P['navy']}; --tm-muted:{P['muted']}; --tm-line:{P['border']}; }}
    .stMainBlockContainer {{max-width:1360px;padding-top:2.4rem;padding-bottom:3rem;}}
    h1,h2,h3,h4 {{letter-spacing:-.025em;color:{P['navy']};}}
    p,li {{line-height:1.65;}}
    [data-testid="stSidebar"] {{border-right:1px solid {P['border']};}}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap:.6rem;}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{padding:.45rem .55rem;border-radius:8px;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{background:{P['teal_light']};}}
    [data-testid="stMetric"] {{background:{P['light_bg']};border:1px solid {P['border']};
      border-radius:14px;padding:1rem 1.1rem;min-height:125px;}}
    [data-testid="stMetricValue"] {{font-variant-numeric:tabular-nums;letter-spacing:-.035em;}}
    [data-testid="stTabs"] [role="tablist"] {{gap:.35rem;border-bottom:1px solid {P['border']};}}
    [data-testid="stTabs"] [role="tab"] {{padding:.6rem .9rem;}}
    button:focus-visible,a:focus-visible {{outline:3px solid {P['teal']};outline-offset:3px;}}
    .tm-brand {{font-size:2rem;font-weight:750;letter-spacing:-.07em;line-height:1.2;color:{P['navy']};}}
    .tm-brand span {{color:{P['teal']};}}
    .tm-school {{font-size:.78rem;letter-spacing:.02em;color:{P['muted']};margin:.3rem 0 1rem;}}
    .tm-hero {{border:1px solid {P['border']};border-radius:20px;padding:2rem;
      background:linear-gradient(115deg,{P['light_bg']},{P['card_bg']});position:relative;overflow:hidden;margin-bottom:1.3rem;}}
    .tm-eyebrow {{color:{P['teal']};font-size:.72rem;text-transform:uppercase;letter-spacing:.15em;font-weight:700;}}
    .tm-hero h1 {{font-size:clamp(1.8rem,3.2vw,3rem);line-height:1.15;max-width:780px;margin:.7rem 0;}}
    .tm-hero p {{color:{P['muted']};max-width:680px;margin-bottom:0;}}
    .tm-layers {{display:flex;gap:5px;margin-bottom:1rem;}}
    .tm-layers i {{display:block;width:42px;height:7px;border-radius:4px;background:{P['teal']};}}
    .tm-layers i:nth-child(2) {{background:{P['amber']};width:64px;}}
    .tm-layers i:nth-child(3) {{background:{P['muted']};width:88px;}}
    .tm-enter .tm-layers i {{animation:tm-arrive .65s ease-out both;}}
    .tm-enter .tm-layers i:nth-child(2) {{animation-delay:.1s;}}
    .tm-enter .tm-layers i:nth-child(3) {{animation-delay:.2s;}}
    @keyframes tm-arrive {{from {{transform:translateX(-14px);opacity:0;}} to {{transform:none;opacity:1;}}}}
    @media(prefers-reduced-motion:reduce) {{.tm-enter .tm-layers i {{animation:none;}}}}
    @media(max-width:700px) {{.tm-hero {{padding:1.2rem;}} .stMainBlockContainer {{padding:1rem;}}
      [data-testid="stMetric"] {{min-height:105px;padding:.75rem;}}}}
    </style>""")

def render_brand():
    st.html('<div class="tm-brand">Tira<span>MisU</span></div>'
            '<div class="tm-school">University of Peradeniya<br>Quant Edge 1.0</div>')

def render_intro():
    first = not st.session_state.get("tm_intro_seen", False)
    st.session_state.tm_intro_seen = True
    cls = "tm-hero tm-enter" if first else "tm-hero"
    st.html(f'<section class="{cls}" aria-label="TiraMisU investment briefing">'
            '<div class="tm-layers" aria-hidden="true"><i></i><i></i><i></i></div>'
            '<div class="tm-eyebrow">TiraMisU · University of Peradeniya</div>'
            f'<h1>{TAGLINE}</h1>'
            '<p>Explore the Serendib Endowment proposal, examine its downside, '
            'and follow the evidence behind each result.</p></section>')

def _navigate(page):
    st.session_state.tm_navigation = page

def render_decision_routes():
    for col, page, label in zip(st.columns(3), [PAGES[1],PAGES[4],PAGES[3]],
        ["Explore the allocation →", "Examine crisis losses →", "Check model reliability →"]):
        with col:
            st.button(label, key="route_"+page, on_click=_navigate, args=(page,), width="stretch")
    st.caption("Saved analytical results · Current-allocation replay and frozen validation are separate analyses.")
