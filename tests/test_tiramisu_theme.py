"""Theme isolation, readable semantic tokens, and branded navigation."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from app import theme

APP = Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'

@pytest.mark.parametrize('mode', ['light', 'dark'])
def test_theme_pages_and_navigation(monkeypatch, mode):
    monkeypatch.setattr(theme, 'theme_name', lambda: mode)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.button(key='route_5. Stress & Risk Contributions').click().run()
    assert at.sidebar.radio[0].value == '5. Stress & Risk Contributions'
    for page in ['1. Overview','2. Portfolio & Performance','3. Risk Analysis',
                 '4. Model Validation','5. Stress & Risk Contributions','6. Assumptions & Data Quality']:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_palette_does_not_leak_between_sessions(monkeypatch):
    monkeypatch.setattr(theme, 'theme_name', lambda: 'dark')
    assert theme.PALETTE['navy'] == theme.DARK['navy']
    monkeypatch.setattr(theme, 'theme_name', lambda: 'light')
    assert theme.PALETTE['navy'] == theme.LIGHT['navy']


def test_body_text_contrast():
    def luminance(color):
        rgb = [int(color[i:i+2],16)/255 for i in (1,3,5)]
        v = [x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in rgb]
        return sum(a*b for a,b in zip(v,[.2126,.7152,.0722]))
    for palette in [theme.LIGHT,theme.DARK]:
        for fg in ['navy','slate','muted']:
            for bg in ['card_bg','light_bg']:
                a,b=sorted([luminance(palette[fg]),luminance(palette[bg])])
                assert (b+.05)/(a+.05)>=4.5, (fg,bg)
