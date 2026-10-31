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
    "SPY": AssetInfo("SPY", "SPDR S&P 500 ETF Trust", "US Large-Cap Equity", "Equity", "#2563eb"),
    "EFA": AssetInfo("EFA", "iShares MSCI EAFE ETF", "Developed Markets Equity", "Equity", "#2563eb"),
    "EEM": AssetInfo("EEM", "iShares MSCI Emerging Markets ETF", "Emerging Markets Equity", "Equity", "#2563eb"),
    "AGG": AssetInfo("AGG", "iShares Core U.S. Aggregate Bond ETF", "US Investment Grade Fixed Income", "Fixed Income", "#0d9488"),
    "GLD": AssetInfo("GLD", "SPDR Gold Shares", "Gold Bullion / Real Assets", "Real Assets", "#d97706"),
    "XLE": AssetInfo("XLE", "Energy Select Sector SPDR Fund", "US Energy Sector Equity", "Equity", "#2563eb"),
    "XLK": AssetInfo("XLK", "Technology Select Sector SPDR Fund", "US Technology Sector Equity", "Equity", "#2563eb"),
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

from .theme import PALETTE
