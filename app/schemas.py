"""Schemas, constants, and asset definitions for the Serendib Endowment Dashboard."""
from enum import Enum
from typing import NamedTuple


class PortfolioBasis(str, Enum):
    CURRENT = "current"
    VALIDATION = "validation"


class AssetInfo(NamedTuple):
    ticker: str
    name: str
    asset_class: str
    category: str  # Equity, Fixed Income, Real Assets
    color: str


ASSET_REGISTRY: dict[str, AssetInfo] = {
    "SPY": AssetInfo("SPY", "SPDR S&P 500 ETF Trust", "US Large-Cap Equity", "Equity", "#1e3a8a"),
    "EFA": AssetInfo("EFA", "iShares MSCI EAFE ETF", "Developed Markets Equity", "Equity", "#2563eb"),
    "EEM": AssetInfo("EEM", "iShares MSCI Emerging Markets ETF", "Emerging Markets Equity", "Equity", "#3b82f6"),
    "AGG": AssetInfo("AGG", "iShares Core U.S. Aggregate Bond ETF", "US Investment Grade Fixed Income", "Fixed Income", "#0d9488"),
    "GLD": AssetInfo("GLD", "SPDR Gold Shares", "Gold Bullion / Real Assets", "Real Assets", "#d97706"),
    "XLE": AssetInfo("XLE", "Energy Select Sector SPDR Fund", "US Energy Sector Equity", "Equity", "#dc2626"),
    "XLK": AssetInfo("XLK", "Technology Select Sector SPDR Fund", "US Technology Sector Equity", "Equity", "#7c3aed"),
}

METHOD_LABELS: dict[str, str] = {
    "historical": "Historical Simulation",
    "parametric_ewma": "EWMA Parametric (Normal)",
    "monte_carlo_bootstrap": "Bootstrap Monte Carlo",
    "evt_gpd": "Extreme Value Theory (EVT-GPD)",
    "historical_simple": "Historical Tail (Simple Returns)",
    "parametric_normal_simple": "Parametric Normal (Simple Returns)",
}

METHOD_DESCRIPTIONS: dict[str, str] = {
    "historical": (
        "Rolling empirical quantile from past 252 trading days. Preserves empirical tail shape and skewness "
        "without distributional assumptions, but reacts slowly to shifting volatility regimes."
    ),
    "parametric_ewma": (
        "Zero-mean normal distribution with pandas bias-corrected EWMA volatility (lambda=0.94). "
        "Adapts rapidly to changing market volatility but assumes standardized returns follow a thin-tailed normal distribution."
    ),
    "monte_carlo_bootstrap": (
        "Bootstrap resampling with replacement (10,000 simulations) from past 252 trading days. Preserves sample "
        "skewness and tail properties; resampling does not add new historical evidence beyond the sample."
    ),
    "evt_gpd": (
        "Generalized Pareto Distribution fitted to positive tail loss exceedances above an empirically selected threshold "
        "(1,000-day window, refit every 21 days). Models heavy fat tails beyond empirical sample bounds."
    ),
    "historical_simple": (
        "Euler component risk using trailing empirical asset returns at the portfolio loss quantile."
    ),
    "parametric_normal_simple": (
        "Euler component risk using analytical normal gradients with sample covariance."
    ),
}

CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
DEFAULT_NOTIONAL = 10_000_000.0

PALETTE = {
    "navy": "#0f172a",
    "slate": "#334155",
    "muted": "#64748b",
    "light_bg": "#f8fafc",
    "card_bg": "#ffffff",
    "border": "#e2e8f0",
    "teal": "#0d9488",
    "teal_light": "#ccfbf1",
    "blue": "#2563eb",
    "blue_light": "#dbeafe",
    "amber": "#d97706",
    "amber_light": "#fef3c7",
    "red": "#dc2626",
    "red_light": "#fee2e2",
    "green": "#16a34a",
    "green_light": "#dcfce7",
}
