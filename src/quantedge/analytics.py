"""Required analytical layers; callable from the pipeline or independently on saved prices."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .backtesting import backtest_forecasts
from .cleaning import compute_returns
from .config import load_config
from .contributions import risk_contributions
from .data import download_fred_series, load_fred_csv
from .estimation import (annualized_historical_returns, market_proxy_returns, capm_expected_returns,
                         shrink_expected_returns, annualized_covariance, shrink_covariance_toward_diagonal)
from .exports import save_frame
from .kpis import compute_kpis
from .portfolio import optimize_configured
from .risk import build_risk_forecasts, common_evaluation_sample, summarize_risk
from .stress import historical_stress, hypothetical_stress


def fit_training_portfolio(simple_returns: pd.DataFrame, risk_free_percent: pd.Series,
                            config: dict) -> tuple[pd.Series, dict]:
    cfg = config["validation"]
    start, end, test_start = (pd.Timestamp(cfg[key]) for key in ("train_start", "train_end", "test_start"))
    if start > end or end >= test_start:
        raise ValueError("Training period must finish before the test period")
    training = simple_returns.loc[start:end, config["data"]["tickers"]].dropna()
    if len(training) < cfg.get("min_train_observations", 504):
        raise ValueError("Insufficient complete training observations")
    if not np.isfinite(training.to_numpy()).all() or (training <= -1).any().any():
        raise ValueError("Invalid training simple returns")
    # Quotes after the last training observation are never eligible, including holiday gaps.
    rf = risk_free_percent.sort_index().loc[:training.index.max()].dropna()
    if rf.empty or not np.isfinite(rf.iloc[-1]):
        raise ValueError("No risk-free observation available by the training cutoff")
    risk_free = float(rf.iloc[-1]/100)
    est = config["estimation"]
    periods = est["trading_days_per_year"]
    historical = annualized_historical_returns(training, periods)
    market = market_proxy_returns(training, config["benchmarks"]["market_proxy_weights"])
    capm = capm_expected_returns(training, market, risk_free, periods)
    mu = shrink_expected_returns(historical, capm, est["expected_return_shrinkage_to_prior"])
    sigma = shrink_covariance_toward_diagonal(annualized_covariance(np.log1p(training), periods),
                                             est["covariance_shrinkage"])
    target = config["portfolio"]["target_return_annual"]
    try:
        weights = optimize_configured(mu, sigma, target, list(training.columns), config)
    except ValueError as exc:
        raise ValueError("Training-only allocation is infeasible; revise the declared target/constraints. "
                         "Full-sample weights will not be substituted.") from exc
    return weights, {"training_start": str(training.index.min().date()),
                     "training_end": str(training.index.max().date()), "training_observations": len(training),
                     "risk_free_quote_date": str(rf.index[-1].date()), "risk_free_annual": risk_free,
                     "target_return_annual": target, "expected_return_annual": float(mu @ weights),
                     "weights": weights.to_dict(), "rebalancing": "daily_constant_weights_no_costs"}


def run_analytical_layers(simple_returns: pd.DataFrame, risk_free_percent: pd.Series,
                          current_weights: pd.Series, config: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    tickers = config["data"]["tickers"]
    complete = simple_returns.loc[:, tickers].dropna()
    if (complete.empty or complete.index.has_duplicates or not complete.index.is_monotonic_increasing
            or not np.isfinite(complete.to_numpy()).all() or (complete <= -1).any().any()):
        raise ValueError("Analytical layers require sorted unique, finite complete returns > -100%")
    if (set(current_weights.index) != set(tickers) or not np.isclose(current_weights.sum(), 1)
            or not np.isfinite(current_weights).all() or (current_weights < 0).any()):
        raise ValueError("Current portfolio weights must cover the universe and sum to one")
    current_weights = current_weights.reindex(tickers)
    weights, metadata = fit_training_portfolio(complete, risk_free_percent, config)
    validation = config["validation"]
    test_start = pd.Timestamp(validation["test_start"])
    test_end = pd.Timestamp(validation["test_end"]) if validation.get("test_end") else complete.index.max()
    if test_end < test_start:
        raise ValueError("Test end must not precede test start")
    # Warmup is restricted to the declared training period; pre-training returns are not used.
    history = complete.loc[pd.Timestamp(validation["train_start"]):test_end]
    log_returns = np.log1p(history @ weights)
    forecasts, diagnostics = build_risk_forecasts(log_returns, config, test_start, include_evt=True)
    methods = ["historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd"]
    common = common_evaluation_sample(forecasts, methods)
    common["evaluation_basis"] = "frozen_training_allocation_out_of_sample"
    common["allocation_training_end"] = pd.Timestamp(metadata["training_end"])
    if (common.estimation_end >= common.date).any():
        raise ValueError("Forecast estimation cutoff must precede its realized return")
    if common.date.min() != history.loc[test_start:].index.min():
        raise ValueError("Training history does not cover all model warmups at test start")
    backtests = backtest_forecasts(common, validation.get("significance", .05))
    oos_returns = (history @ weights).loc[test_start:]
    policy_weights = pd.Series(config["benchmarks"]["policy_benchmark_weights"]).reindex(tickers).fillna(0)
    policy_weights /= policy_weights.sum()
    policy = (history @ policy_weights).loc[test_start:]
    market = market_proxy_returns(history, config["benchmarks"]["market_proxy_weights"]).loc[test_start:]
    # OOS KPI excess returns use an as-of daily risk-free history, not today's quote.
    rf = risk_free_percent.sort_index().dropna().reindex(oos_returns.index, method="ffill")/100
    if rf.isna().any() or not np.isfinite(rf).all() or (rf <= -1).any():
        raise ValueError("Risk-free history must cover the test period")
    kpis = compute_kpis(oos_returns, policy, market, rf, config["estimation"]["trading_days_per_year"])
    oos_history = pd.DataFrame({"date": oos_returns.index, "portfolio_return": oos_returns.to_numpy(),
                                "policy_return": policy.to_numpy(), "market_return": market.to_numpy(),
                                "risk_free_annual": rf.to_numpy(),
                                "portfolio_wealth": (1+oos_returns).cumprod().to_numpy(),
                                "policy_wealth": (1+policy).cumprod().to_numpy()})
    stress_cfg = config["stress"]
    value = config["project"]["portfolio_value_usd"]
    historical, paths = historical_stress(simple_returns, current_weights, stress_cfg["historical"], value,
                                          stress_cfg.get("rebalance", "daily"))
    hypothetical, shock_components = hypothetical_stress(current_weights, stress_cfg["hypothetical"], value)
    window = config["var"]["historical_window"]
    contributions = risk_contributions(complete.iloc[-window:], current_weights,
                                       config["var"]["confidence_levels"], value)
    metadata.update(test_start=str(common.date.min().date()), test_end=str(common.date.max().date()),
                    evaluation_observations=int(common.date.nunique()),
                    data_cutoff=str(complete.index.max().date()),
                    current_weights=current_weights.to_dict(),
                    stress_allocation_basis="current_weights_retrospective_replay",
                    forecast_notional_usd=value,
                    risk_contribution_basis="current_weights_simple_return_tail_and_normal_approximation")
    tables = {"risk_forecasts_oos": common, "risk_summary_oos": summarize_risk(common),
              "backtest_results": backtests, "evt_threshold_diagnostics": diagnostics,
              "oos_portfolio_weights": weights.rename_axis("asset").rename("weight").reset_index(),
              "oos_performance": oos_history, "oos_kpi_summary": pd.DataFrame([kpis]),
              "stress_summary": pd.concat([historical, hypothetical], ignore_index=True),
              "stress_paths": paths, "stress_shock_contributions": shock_components,
              "risk_contributions": contributions}
    return tables, metadata


def export_analytical_layers(tables: dict, metadata: dict, config: dict, root: Path) -> None:
    for name, frame in tables.items():
        save_frame(frame, root/"outputs/tables"/f"{name}.csv",
                   config["outputs"]["save_csv"], config["outputs"]["save_parquet"])
    path = root/"outputs/models/analytical_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "configuration": config}
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prices", default="outputs/tables/prices.csv")
    parser.add_argument("--risk-free", help="FRED CSV history in percent; omit to download DGS3MO")
    parser.add_argument("--portfolio", default="outputs/models/selected_portfolio.json")
    args = parser.parse_args()
    config = load_config(args.config)
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    _, simple = compute_returns(prices)
    if args.risk_free:
        risk_free = load_fred_csv(args.risk_free, config["data"]["fred_risk_free_series"])
    else:
        risk_free = download_fred_series(config["data"]["fred_risk_free_series"], Path("data/raw/risk_free.csv"))
    weights = pd.Series(json.loads(Path(args.portfolio).read_text(encoding="utf-8"))["weights"], dtype=float)
    tables, metadata = run_analytical_layers(simple, risk_free, weights, config)
    export_analytical_layers(tables, metadata, config, Path.cwd())
    print(f"Analytical layers complete: {metadata['evaluation_observations']} OOS observations; "
          f"{len(tables['backtest_results'])} model/confidence backtests")


if __name__ == "__main__":
    main()
