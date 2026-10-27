from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config
from .data import load_or_download_prices, download_fred_series
from .cleaning import compute_returns, flag_missing, flag_extreme_returns, adf_report, build_data_quality_report
from .estimation import annualized_historical_returns, market_proxy_returns, capm_expected_returns, shrink_expected_returns, annualized_covariance, shrink_covariance_toward_diagonal
from .portfolio import efficient_frontier, optimize_configured
from .kpis import compute_kpis
from .var import build_var_suite
from .exports import save_frame, save_selected_portfolio
from .utils import ensure_output_dirs, get_logger


def run(config_path: str = "config.yaml") -> None:
    cfg, root = load_config(config_path), Path.cwd()
    ensure_output_dirs(root)
    log = get_logger()
    tickers = cfg["data"]["tickers"]
    prices = load_or_download_prices(tickers, cfg["data"]["start_date"], None, cfg["data"]["cache_raw_data"], root)
    missing_tickers = set(tickers).difference(prices.columns)
    if missing_tickers:
        raise ValueError(f"Price data missing configured tickers: {sorted(missing_tickers)}")
    rf_series = download_fred_series(cfg["data"]["fred_risk_free_series"], root/"data/raw/risk_free.csv" if cfg["data"]["cache_raw_data"] else None)
    log_returns, simple_returns = compute_returns(prices)
    estimation_start = pd.Timestamp(cfg["estimation"]["core_start_date"])
    est_log = log_returns.loc[log_returns.index >= estimation_start]
    est_simple = simple_returns.loc[simple_returns.index >= estimation_start]
    if est_simple.empty:
        raise ValueError("No observations in configured estimation window")
    periods = cfg["estimation"]["trading_days_per_year"]
    market_est = market_proxy_returns(est_simple, cfg["benchmarks"]["market_proxy_weights"])
    risk_free = float(rf_series.iloc[-1] / 100)
    historical = annualized_historical_returns(est_simple, periods)
    capm = capm_expected_returns(est_simple, market_est, risk_free, periods)
    mu = shrink_expected_returns(historical, capm, cfg["estimation"]["expected_return_shrinkage_to_prior"])
    sigma = shrink_covariance_toward_diagonal(annualized_covariance(est_log, periods), cfg["estimation"]["covariance_shrinkage"])
    frontier, _ = efficient_frontier(mu, sigma, cfg["portfolio"]["target_return_grid"], tickers, cfg)
    feasible = frontier.loc[frontier.status.eq("feasible")]
    if feasible.empty:
        raise ValueError("No configured efficient-frontier target is feasible")
    desired = cfg["portfolio"]["target_return_annual"]
    selected_target = float(feasible.iloc[(feasible.target_return - desired).abs().argsort().iloc[0]].target_return)
    weights = optimize_configured(mu, sigma, selected_target, tickers, cfg)

    complete = simple_returns.reindex(columns=tickers).dropna()
    portfolio_returns = complete @ weights
    policy_weights = pd.Series(cfg["benchmarks"]["policy_benchmark_weights"], dtype=float).reindex(tickers).fillna(0)
    policy_weights /= policy_weights.sum()
    policy = complete @ policy_weights
    market_all = market_proxy_returns(complete, cfg["benchmarks"]["market_proxy_weights"])
    kpis = compute_kpis(portfolio_returns, policy, market_all, risk_free, periods)
    portfolio_log = np.log1p(portfolio_returns)
    var_forecasts, var_summary = build_var_suite(portfolio_log, cfg)

    tables, processed = root/"outputs/tables", root/"data/processed"
    missing_flags, outlier_flags = flag_missing(prices), flag_extreme_returns(log_returns)
    data_quality = build_data_quality_report(prices, log_returns, missing_flags, outlier_flags)
    adf = adf_report(est_log)
    exports = [
        (prices, tables/"prices.csv"), (log_returns, tables/"log_returns.csv"), (simple_returns, tables/"simple_returns.csv"),
        (log_returns, processed/"log_returns.csv"), (simple_returns, processed/"simple_returns.csv"),
        (data_quality, tables/"data_quality_report.csv"), (adf, tables/"adf_report.csv"),
        (mu.rename("expected_return_annual").to_frame(), tables/"expected_returns.csv"), (sigma, tables/"covariance_matrix.csv"),
        (frontier, tables/"efficient_frontier.csv"), (weights.rename("weight").to_frame(), tables/"portfolio_weights.csv"),
        (pd.DataFrame([kpis]), tables/"kpi_summary.csv"), (var_summary, tables/"var_summary.csv"),
        (var_forecasts, tables/"var_forecasts.csv"),
    ]
    for frame, path in exports:
        save_frame(frame, path, cfg["outputs"]["save_csv"], cfg["outputs"]["save_parquet"])
    save_frame(pd.DataFrame({"asset": weights.index, "weight": weights.values, "expected_return_annual": mu.reindex(weights.index).values}), tables/"streamlit_portfolio_summary.csv")
    save_frame(var_summary, tables/"streamlit_var_dashboard.csv")
    expected = float(mu.reindex(tickers) @ weights)
    covariance = sigma.reindex(index=tickers, columns=tickers).to_numpy()
    volatility = float(np.sqrt(weights.to_numpy() @ covariance @ weights.to_numpy()))
    save_selected_portfolio(root/"outputs/models/selected_portfolio.json", cfg["project"]["portfolio_value_usd"], selected_target, weights, expected, volatility, cfg["portfolio"], {"start": str(est_simple.index.min().date()), "end": str(est_simple.index.max().date()), "observations": len(est_simple)}, "Expected returns blend historical estimates and CAPM; covariance uses diagonal shrinkage.")

    (tables/"var_method_comparison.md").write_text(
        "# VaR Method Comparison\n\n"
        "Forecasts use information available before each forecast date. Historical simulation preserves the observed empirical tail but adapts slowly. "
        "EWMA parametric VaR reacts to volatility changes but assumes normal standardized returns and zero mean. Bootstrap Monte Carlo resamples recent observed returns; "
        "it cannot generate losses beyond its sample. The method with the highest VaR is most conservative for the observed sample, while EWMA generally reacts fastest to changing volatility. "
        "For a conservative client, compare methods and supplement them with Expected Shortfall and stress analysis. VaR is not a guarantee of maximum loss.\n",
        encoding="utf-8",
    )
    update_docs(prices, weights, expected, volatility, var_summary, data_quality, adf, selected_target)
    log.info("Pipeline complete: target %.3f, expected return %.3f, volatility %.3f", selected_target, expected, volatility)


def update_docs(prices: pd.DataFrame, weights: pd.Series, expected: float, volatility: float, var_summary: pd.DataFrame, quality: pd.DataFrame, adf: pd.DataFrame, target: float) -> None:
    from datetime import date
    today = date.today().isoformat()
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    notes = docs/"LIVING_PROJECT_NOTES.md"
    weights_text = ", ".join(f"{ticker} {weight:.1%}" for ticker, weight in weights.items())
    summary_text = ", ".join(f"{row.method} {row.confidence:.0%}: {row.latest_var_pct:.2%}" for row in var_summary.itertuples())
    change = f"| {today} | Codex | Ran configured pipeline; target {target:.2%}, expected return {expected:.2%}, volatility {volatility:.2%} | outputs/, docs/ | Record reproducible results |\n"
    if notes.exists():
        with notes.open("a", encoding="utf-8") as handle:
            handle.write("\n" + change)
    else:
        notes.write_text(f"""# Living Project Notes

## Last Updated
- Date: {today}
- Agent: Codex
- Commit hash / branch: recorded in Git history

## Current Status
- What is working: config-driven Layers 1 and 2 pipeline.
- What is incomplete: later Expected Shortfall, EVT, formal backtesting, and dashboard.
- What is blocked: live results require public Yahoo Finance and FRED access.

## Repository Overview
- Purpose: conservative endowment portfolio risk analysis.
- Main entry points: `python -m src.quantedge.main`.
- Main outputs: CSV tables and selected portfolio JSON under outputs/.

## Architecture
- data layer: Yahoo Finance adjusted close and public FRED series.
- estimation layer: historical/CAPM blended means and diagonal covariance shrinkage.
- portfolio layer: constrained target-return minimum variance.
- VaR layer: historical simulation, EWMA normal, and bootstrap Monte Carlo.
- exports layer: tidy CSV/Parquet tables and portfolio JSON.

## Key Decisions
| Decision | Choice | Why | Alternatives Considered |
|---|---|---|---|
| Objective | Target-return minimum variance | Matches capital preservation mandate | Maximum Sharpe |
| Mean estimate | Blend historical and CAPM equally | Reduces reliance on noisy sample means | Historical only |

## Data Governance
- Source: Yahoo Finance via yfinance and FRED DGS3MO.
- Start date: 2007-01-01; estimation begins 2010-01-01.
- Adjusted close used: yes.
- Calendar alignment: no forward fill; complete rows used for portfolio time series.
- Missing data treatment: reported; incomplete price rows retained in exports.
- Outlier flagging: robust MAD flags; values are not deleted.
- ADF results: outputs/tables/adf_report.csv; {int(adf.stationary_at_1pct.sum())} of {len(adf)} stationary at 1%.

## Layer 1 Summary
- Expected return method: historical/CAPM blend.
- Covariance method: annualized log-return covariance with diagonal shrinkage.
- Optimizer objective: minimum variance subject to target return.
- Constraints used: configured exposure, asset, and single-position bounds.
- Selected target return: {target:.2%}.
- Final weights: {weights_text}.
- Annual expected return / volatility: {expected:.2%} / {volatility:.2%}.

## Layer 2 Summary
- Historical VaR: rolling empirical quantile.
- Parametric VaR: EWMA volatility with normal quantile.
- Monte Carlo VaR: rolling bootstrap from prior observations.
- Forecast alignment: one step ahead.
- Exception logic: actual log return below negative forecast log VaR.
- Latest VaR estimates: {summary_text}.
- Output files: outputs/tables/var_summary.csv and var_forecasts.csv.

## Known Issues
- Issue: estimates use the latest short-rate observation as a constant. Impact: KPI risk-free adjustment is not time-varying. Proposed fix: align a daily FRED history.
- Issue: VaR is a threshold measure. Impact: it does not describe loss severity beyond the threshold. Proposed fix: add Expected Shortfall and EVT.

## Next Steps for Teammates
- Bhaveenthan: extend Expected Shortfall and EVT.
- Rashad: implement formal VaR backtests.
- Daniel / Daniru: integrate dashboard outputs.

## Change Log
| Date | Agent | Change | Files Affected | Reason |
|---|---|---|---|---|
{change}""", encoding="utf-8")
    (docs/"DATA_GOVERNANCE.md").write_text(f"""# Data Governance

Yahoo Finance adjusted close is the sole price input. The explicit `Adj Close` field is selected with automatic adjustment disabled. FRED DGS3MO supplies the public short-rate series. Raw source downloads are cached under `data/raw/` when enabled. Calendars are not forward-filled. Missing values are retained in source exports and reported; complete rows are used for portfolio returns. Potential outliers are flagged using an eight-MAD threshold and are not silently removed. ADF diagnostics are written to `outputs/tables/adf_report.csv`.

Latest downloaded price date: {prices.index.max().date()}. Assets with missing prices: {int(quality.missing_price_days.sum())} asset-days. Potential extreme-return flags: {int(quality.extreme_return_flags.sum())}. Treat downloads as third-party source data and verify them before publication.
""", encoding="utf-8")
    (docs/"METHODOLOGY.md").write_text("""# Methodology and Limitations

## Portfolio construction
The engine minimizes annualized portfolio variance subject to a configured target return and long-only client constraints. This is preferred to a maximum-Sharpe objective because the mandate prioritizes downside control over return maximization. Expected returns blend annualized historical simple returns with CAPM estimates. The blend reduces dependence on noisy sample means. Covariance is estimated from log returns and shrunk toward its diagonal to reduce unstable correlations.

## Returns and risk
Log returns are used for statistical estimation and VaR. Simple returns are used for portfolio wealth and performance KPIs. VaR forecasts are shifted one observation so date t uses information only through t-1. Bootstrap Monte Carlo resamples the trailing empirical return window and preserves its observed skew and tail behavior, but cannot extrapolate beyond the sample. Historical simulation has the same limitation; EWMA normal VaR depends on its distributional assumption.

> VaR is not a guarantee of maximum loss. It is a probabilistic threshold estimate. Tail events can exceed VaR.

Expected Shortfall is needed to summarize losses beyond the VaR threshold. EVT may improve tail extrapolation, while formal backtesting is required to assess forecast calibration. The current model does not capture fees, taxes, liquidity, trading costs, changing market regimes, or forward-looking shocks. CAPM and covariance estimates are sample-dependent.
""", encoding="utf-8")
    (docs/"AGENT_RUN_REPORT.md").write_text(f"""# Agent Run Report

## Request
Implement the provided Quant Edge Layers 1 and 2 specification.

## Work completed
- Implemented data access, cleaning, ADF diagnostics, return estimation, constrained optimization, investment KPIs, three VaR methods, exports, and documentation updates.
- Created: config, source package, methodology, data governance, living notes, report, tests, app placeholder, and notebook placeholders.
- Dependencies added: only the packages listed in requirements.txt from the requested specification.

## Run evidence
- Price period: {prices.index.min().date()} to {prices.index.max().date()}.
- Data quality: {int(quality.missing_price_days.sum())} missing price cells and {int(quality.extreme_return_flags.sum())} flagged extreme returns.
- ADF: {int(adf.stationary_at_1pct.sum())} of {len(adf)} series stationary at 1%.
- Weights: {weights.to_dict()}.
- Expected return: {expected:.4%}; annualized volatility: {volatility:.4%}.
- VaR summary: {var_summary.to_dict(orient='records')}.
- Tests: run `pytest`.

## Assumptions made
- FRED latest DGS3MO rate is treated as a constant for all KPI calculations in the run.
- All configured tickers are required; absent downloads fail the pipeline.
- Observations with any missing asset return are omitted for portfolio-level calculations.
- Historical, CAPM, and diagonal-shrinkage approaches follow the provided specification.

## Known issues and recommendations
Refresh the public data before relying on outputs; inspect flagged outliers, infeasible target returns, and exception counts. Add formal backtests, Expected Shortfall, EVT, and stress scenarios before using this as a production risk system. Exclude fees, taxes, liquidity, and transaction costs explicitly when interpreting results.
""", encoding="utf-8")


if __name__ == "__main__":
    run()
