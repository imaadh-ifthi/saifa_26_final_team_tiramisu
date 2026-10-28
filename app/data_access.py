"""Data loading, validation, and adapter layer for Quant Edge 1.0."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st

from .schemas import ASSET_REGISTRY, PortfolioBasis, DEFAULT_NOTIONAL
from .formatting import parse_boolean

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return REPO_ROOT


def resolve_path(rel_path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    return (REPO_ROOT / rel_path).resolve()


def check_file_exists(rel_path: str | Path) -> bool:
    """Check if a file exists relative to the repository root."""
    return resolve_path(rel_path).is_file()


@st.cache_data(show_spinner=False)
def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = resolve_path("config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_selected_portfolio() -> dict[str, Any]:
    """Load current proposed portfolio artifact from outputs/models/selected_portfolio.json."""
    path = resolve_path("outputs/models/selected_portfolio.json")
    if not path.is_file():
        raise FileNotFoundError(f"Selected portfolio JSON not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Validate required keys
    for key in ("portfolio_value_usd", "target_return_annual", "weights", "expected_return_annual", "volatility_annual"):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in selected_portfolio.json")
    return data


@st.cache_data(show_spinner=False)
def load_analytical_run() -> dict[str, Any]:
    """Load analytical run metadata from outputs/models/analytical_run.json."""
    path = resolve_path("outputs/models/analytical_run.json")
    if not path.is_file():
        raise FileNotFoundError(f"Analytical run JSON not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_portfolio_weights(basis: PortfolioBasis = PortfolioBasis.CURRENT) -> pd.DataFrame:
    """
    Load portfolio weights with asset metadata and dollar allocations.
    Validates weight sum equals 1.0 within numerical tolerance.
    """
    if basis == PortfolioBasis.CURRENT:
        path = resolve_path("outputs/tables/portfolio_weights.csv")
    else:
        path = resolve_path("outputs/tables/oos_portfolio_weights.csv")

    if not path.is_file():
        raise FileNotFoundError(f"Portfolio weights file not found at {path}")

    df = pd.read_csv(path)
    if "asset" not in df.columns or "weight" not in df.columns:
        raise ValueError(f"Portfolio weights CSV must have 'asset' and 'weight' columns. Found {df.columns.tolist()}")

    df["weight"] = df["weight"].astype(float)
    total_weight = df["weight"].sum()
    if not np.isclose(total_weight, 1.0, atol=1e-4):
        raise ValueError(f"Weights do not sum to 1.0 (sum = {total_weight})")

    # Add descriptive metadata from schema registry
    df["name"] = df["asset"].map(lambda a: ASSET_REGISTRY[a].name if a in ASSET_REGISTRY else a)
    df["asset_class"] = df["asset"].map(lambda a: ASSET_REGISTRY[a].asset_class if a in ASSET_REGISTRY else "Other")
    df["category"] = df["asset"].map(lambda a: ASSET_REGISTRY[a].category if a in ASSET_REGISTRY else "Other")
    df["color"] = df["asset"].map(lambda a: ASSET_REGISTRY[a].color if a in ASSET_REGISTRY else "#64748b")

    # Clean display weights (treat microscopic weights < 1e-6 as 0 for display)
    df["display_weight"] = df["weight"].apply(lambda w: 0.0 if abs(w) < 1e-6 else w)

    notional = DEFAULT_NOTIONAL
    try:
        portfolio_meta = load_selected_portfolio()
        notional = float(portfolio_meta.get("portfolio_value_usd", DEFAULT_NOTIONAL))
    except Exception:
        pass

    df["dollar_allocation"] = df["weight"] * notional
    return df


@st.cache_data(show_spinner=False)
def load_expected_returns() -> pd.DataFrame:
    """Load blended expected returns from outputs/tables/expected_returns.csv."""
    path = resolve_path("outputs/tables/expected_returns.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Expected returns file not found at {path}")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_covariance_matrix() -> pd.DataFrame:
    """Load covariance matrix from outputs/tables/covariance_matrix.csv."""
    path = resolve_path("outputs/tables/covariance_matrix.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Covariance matrix not found at {path}")
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data(show_spinner=False)
def load_efficient_frontier() -> pd.DataFrame:
    """Load efficient frontier grid from outputs/tables/efficient_frontier.csv."""
    path = resolve_path("outputs/tables/efficient_frontier.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Efficient frontier not found at {path}")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_kpi_summary(basis: PortfolioBasis = PortfolioBasis.CURRENT) -> dict[str, Any]:
    """Load investment KPIs for either current retrospective or frozen OOS validation."""
    if basis == PortfolioBasis.CURRENT:
        path = resolve_path("outputs/tables/kpi_summary.csv")
    else:
        path = resolve_path("outputs/tables/oos_kpi_summary.csv")

    if not path.is_file():
        raise FileNotFoundError(f"KPI summary file not found at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"KPI summary at {path} is empty")
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_performance_series(basis: PortfolioBasis = PortfolioBasis.CURRENT) -> pd.DataFrame:
    """
    Load daily return, cumulative wealth, and drawdown series.
    - If validation: reads outputs/tables/oos_performance.csv directly.
    - If current: reconstructs daily return series from simple_returns.csv and current weights.
    """
    if basis == PortfolioBasis.VALIDATION:
        path = resolve_path("outputs/tables/oos_performance.csv")
        if not path.is_file():
            raise FileNotFoundError(f"OOS performance file not found at {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # Compute drawdown from portfolio_wealth
        peak = df["portfolio_wealth"].cummax().clip(lower=1.0)
        df["portfolio_drawdown"] = df["portfolio_wealth"] / peak - 1.0
        policy_peak = df["policy_wealth"].cummax().clip(lower=1.0)
        df["policy_drawdown"] = df["policy_wealth"] / policy_peak - 1.0
        df["basis"] = "Frozen validation portfolio (OOS)"
        return df

    # Reconstruct for current proposed allocation
    prices_path = resolve_path("outputs/tables/simple_returns.csv")
    if not prices_path.is_file():
        raise FileNotFoundError(f"Simple returns file not found at {prices_path}")
    simple_returns = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    weights_df = load_portfolio_weights(PortfolioBasis.CURRENT)
    weights = pd.Series(weights_df["weight"].values, index=weights_df["asset"])

    cfg = load_config()
    policy_weights = pd.Series(cfg["benchmarks"]["policy_benchmark_weights"]).reindex(weights.index).fillna(0.0)
    policy_weights /= policy_weights.sum()

    # Align complete cases
    complete = simple_returns.reindex(columns=weights.index).dropna()
    port_ret = complete @ weights
    policy_ret = complete @ policy_weights

    wealth = (1.0 + port_ret).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / peak - 1.0

    policy_wealth = (1.0 + policy_ret).cumprod()
    policy_peak = policy_wealth.cummax().clip(lower=1.0)
    policy_drawdown = policy_wealth / policy_peak - 1.0

    res = pd.DataFrame({
        "date": complete.index,
        "portfolio_return": port_ret.values,
        "policy_return": policy_ret.values,
        "portfolio_wealth": wealth.values,
        "policy_wealth": policy_wealth.values,
        "portfolio_drawdown": drawdown.values,
        "policy_drawdown": policy_drawdown.values,
        "basis": "Current proposed portfolio (Retrospective replay)",
    }).sort_values("date").reset_index(drop=True)
    return res


@st.cache_data(show_spinner=False)
def load_var_summary(basis: PortfolioBasis = PortfolioBasis.CURRENT) -> pd.DataFrame:
    """Load latest VaR summary table for current or validation allocation."""
    if basis == PortfolioBasis.CURRENT:
        path = resolve_path("outputs/tables/var_summary.csv")
    else:
        path = resolve_path("outputs/tables/risk_summary_oos.csv")

    if not path.is_file():
        raise FileNotFoundError(f"VaR summary file not found at {path}")

    df = pd.read_csv(path, parse_dates=["latest_date"])
    df["confidence"] = df["confidence"].astype(float)
    return df


@st.cache_data(show_spinner=False)
def load_var_forecasts(basis: PortfolioBasis = PortfolioBasis.CURRENT) -> pd.DataFrame:
    """Load full historical daily risk forecasts."""
    if basis == PortfolioBasis.CURRENT:
        path = resolve_path("outputs/tables/var_forecasts.csv")
    else:
        path = resolve_path("outputs/tables/risk_forecasts_oos.csv")

    if not path.is_file():
        raise FileNotFoundError(f"VaR forecasts file not found at {path}")

    # Use specific dtypes to be memory efficient and robust
    df = pd.read_csv(path, parse_dates=["date", "estimation_end"])
    df["confidence"] = df["confidence"].astype(float)
    df["exception"] = df["exception"].apply(parse_boolean)
    df["log_es_finite"] = df["log_es_finite"].apply(parse_boolean)
    return df


@st.cache_data(show_spinner=False)
def load_backtest_results() -> pd.DataFrame:
    """Load baseline OOS backtesting results."""
    path = resolve_path("outputs/tables/backtest_results.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Backtest results not found at {path}")
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    df["confidence"] = df["confidence"].astype(float)
    df["kupiec_reject"] = df["kupiec_reject"].apply(parse_boolean)
    df["independence_reject"] = df["independence_reject"].apply(parse_boolean)
    df["conditional_coverage_reject"] = df["conditional_coverage_reject"].apply(parse_boolean)
    return df


@st.cache_data(show_spinner=False)
def load_evt_diagnostics() -> pd.DataFrame:
    """Load EVT candidate threshold diagnostics."""
    path = resolve_path("outputs/tables/evt_threshold_diagnostics.csv")
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["fit_date", "estimation_end"])
    df["date"] = df["fit_date"]
    if "selected" in df.columns:
        df["selected"] = df["selected"].apply(parse_boolean)
    if "stable_with_next" in df.columns:
        df["stable_with_next"] = df["stable_with_next"].apply(parse_boolean)
    return df


@st.cache_data(show_spinner=False)
def load_stress_summary() -> pd.DataFrame:
    """Load historical crisis and hypothetical scenario summary."""
    path = resolve_path("outputs/tables/stress_summary.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Stress summary file not found at {path}")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_stress_paths() -> pd.DataFrame:
    """Load wealth and drawdown trajectories during historical crisis periods."""
    path = resolve_path("outputs/tables/stress_paths.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Stress paths file not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_stress_shock_contributions() -> pd.DataFrame:
    """Load hypothetical stagflation scenario position P&L contributions."""
    path = resolve_path("outputs/tables/stress_shock_contributions.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Stress shock contributions file not found at {path}")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_risk_contributions() -> pd.DataFrame:
    """Load Euler marginal and component risk contributions."""
    path = resolve_path("outputs/tables/risk_contributions.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Risk contributions file not found at {path}")
    df = pd.read_csv(path)
    df["confidence"] = df["confidence"].astype(float)
    return df


@st.cache_data(show_spinner=False)
def load_data_quality_report() -> pd.DataFrame:
    """Load data quality diagnostics."""
    path = resolve_path("outputs/tables/data_quality_report.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Data quality report not found at {path}")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_adf_report() -> pd.DataFrame:
    """Load Augmented Dickey-Fuller stationarity test report."""
    path = resolve_path("outputs/tables/adf_report.csv")
    if not path.is_file():
        raise FileNotFoundError(f"ADF report not found at {path}")
    df = pd.read_csv(path)
    df["stationary_at_1pct"] = df["stationary_at_1pct"].apply(parse_boolean)
    return df


@st.cache_data(show_spinner=False)
def load_data_cutoffs() -> dict[str, Any]:
    """
    Detect explicit timestamps and discrepancy dates across data layers:
    - Latest price date
    - Latest complete portfolio return date
    - Latest evaluated risk date
    - Missing price observations
    """
    prices_path = resolve_path("outputs/tables/prices.csv")
    returns_path = resolve_path("outputs/tables/simple_returns.csv")
    forecasts_path = resolve_path("outputs/tables/risk_forecasts_oos.csv")

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])

    latest_price_date = prices.index.max()
    complete_returns = returns.dropna()
    latest_return_date = complete_returns.index.max()
    latest_risk_date = forecasts["date"].max()

    # Detect missing price dates per asset
    missing_by_asset = {}
    missing_dates = {}
    for col in prices.columns:
        n_missing = int(prices[col].isna().sum())
        missing_by_asset[col] = n_missing
        if n_missing > 0:
            missing_dates[col] = [d.strftime("%Y-%m-%d") for d in prices.index[prices[col].isna()]]

    return {
        "latest_price_date": latest_price_date,
        "latest_return_date": latest_return_date,
        "latest_risk_date": latest_risk_date,
        "missing_by_asset": missing_by_asset,
        "missing_dates": missing_dates,
        "price_observations": len(prices),
        "complete_return_observations": len(complete_returns),
    }


def recompute_backtest_window(
    forecasts_df: pd.DataFrame,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    method: str | None = None,
    confidence: float | None = None,
    significance: float = 0.05,
) -> dict[str, Any]:
    """
    Recompute Kupiec POF, Christoffersen independence, conditional coverage,
    and Basel traffic light on filtered forecast rows using existing backend functions.
    """
    from src.quantedge.backtesting import kupiec_test, christoffersen_test, basel_traffic_light
    from scipy.stats import chi2

    filtered = forecasts_df.copy()
    if start_date is not None:
        filtered = filtered[filtered["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["date"] <= pd.Timestamp(end_date)]
    if method is not None:
        filtered = filtered[filtered["method"] == method]
    if confidence is not None:
        filtered = filtered[np.isclose(filtered["confidence"], confidence)]

    filtered = filtered.sort_values("date").reset_index(drop=True)
    n = len(filtered)
    if n == 0:
        return {"error": "No forecast observations in selected window"}

    hits = filtered["exception"].to_numpy().astype(int)
    c = confidence if confidence is not None else float(filtered["confidence"].iloc[0])

    kupiec = kupiec_test(hits, c, significance)
    independent = christoffersen_test(hits, significance)

    lr_cc = kupiec["kupiec_lr"] + (independent["independence_lr"] if np.isfinite(independent["independence_lr"]) else 0.0)
    p_cc = float(chi2.sf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan
    traffic = basel_traffic_light(hits, c)

    return {
        "observations": n,
        "start_date": filtered["date"].min(),
        "end_date": filtered["date"].max(),
        "confidence": c,
        "method": method or "all",
        **kupiec,
        **independent,
        "conditional_coverage_lr": lr_cc,
        "conditional_coverage_p_value": p_cc,
        "conditional_coverage_reject": p_cc < significance if np.isfinite(p_cc) else None,
        **traffic,
    }


def clear_app_cache() -> None:
    """Clear Streamlit cached data to reload all files from disk."""
    st.cache_data.clear()
