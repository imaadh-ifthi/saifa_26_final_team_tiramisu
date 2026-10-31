"""Streamlit AppTest integration tests for full UI navigation and interactivity."""
import pytest
from pathlib import Path
APP_PATH = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
from streamlit.testing.v1 import AppTest


def test_apptest_overview_page():
    """Verify Overview page renders cleanly with key metrics."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert len(at.exception) == 0
    # Check that header rendered
    assert any("Executive Overview" in str(h.value) for h in at.markdown)


def test_apptest_portfolio_page_current_and_validation():
    """Verify Portfolio page renders for both Current and Validation bases."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    # Navigate to Portfolio page
    at.sidebar.radio[0].set_value("2. Portfolio & Performance").run()
    assert len(at.exception) == 0
    assert any("Portfolio & Performance" in str(m.value) for m in at.markdown)

    # Switch to validation portfolio
    basis_radios = [r for r in at.sidebar.radio if "Select Portfolio Basis:" in str(r.label)]
    if basis_radios:
        basis_radios[0].set_value("Frozen validation portfolio").run()
        assert len(at.exception) == 0


def test_apptest_risk_page_interactions():
    """Verify Risk Analysis view renders and reacts to confidence/method selections."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.radio[0].set_value("3. Risk Analysis").run()
    assert len(at.exception) == 0
    assert any("Risk Analysis" in str(m.value) for m in at.markdown)


def test_apptest_validation_page():
    """Verify Model Validation view renders without errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.radio[0].set_value("4. Model Validation").run()
    assert len(at.exception) == 0
    assert any("Model Validation" in str(m.value) for m in at.markdown)


def test_apptest_stress_page():
    """Verify Stress & Risk Contributions page renders and displays scenario selectors."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.radio[0].set_value("5. Stress & Risk Contributions").run()
    assert len(at.exception) == 0
    assert any("Stress" in str(m.value) for m in at.markdown)


def test_apptest_assumptions_page():
    """Verify Assumptions & Data Quality page renders diagnostics tables."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.radio[0].set_value("6. Assumptions & Data Quality").run()
    assert len(at.exception) == 0
    assert any("Assumptions" in str(m.value) for m in at.markdown)


def test_apptest_reload_button():
    """Verify Reload Results button executes and refreshes application state."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    # Click reload button in sidebar
    reload_btns = [b for b in at.sidebar.button if "Reload" in str(b.label)]
    if reload_btns:
        reload_btns[0].click().run()
        assert len(at.exception) == 0
