# Quant Edge 1.0 — Serendib Endowment Portfolio

This repository contains the quantitative analysis engine for the Quant Edge 1.0 Final Challenge. It constructs and evaluates a conservative USD 10 million portfolio for the Serendib University Endowment Fund. The mandate prioritizes capital preservation, downside measurement, and resilience over maximizing expected return.

## Team Scope

- **Layer 1:** portfolio construction, CAPM diagnostics, and investment KPIs.
- **Layer 2:** one-step-ahead Historical, EWMA Parametric, and Bootstrap Monte Carlo VaR.
- **Layer 3:** empirical and EWMA-normal Expected Shortfall with correct dollar-loss conversion.
- **Layer 4:** POT/GPD tail estimates, threshold stability diagnostics, and sensitivity tables.
- **Layer 5:** training-only allocation, out-of-sample forecasts, Kupiec and Christoffersen tests, and a legacy Basel traffic light.
- **Investor layer:** historical/hypothetical stress tests and marginal/component VaR and ES contributions.
- Dashboard integration remains separate; the UI draft is preserved in a Git stash.

## Current Status

The five analytical layers and the full interactive Streamlit dashboard application are implemented, verified, and tested. The application runs locally from pre-computed analytical outputs with `python -m streamlit run app/streamlit_app.py`. The test suite includes 39 passing unit and integration tests covering analytical formulas, data loading, dynamic backtesting, and visualization. With the current 2007-start training allocation, the four unchanged baseline methods record 31 to 48 exceptions in 2,212 evaluation days at 99% VaR; none avoids rejection by the joint conditional-coverage test. Review the current backtest exports and EVT warnings before presenting an investment recommendation.

The Streamlit application (`app/streamlit_app.py`) provides an institutional decision dashboard with six dedicated views: Executive Overview, Portfolio & Performance, Risk Analysis, Model Validation, Stress Testing & Risk Contributions, and Assumptions & Data Quality. See [docs/STREAMLIT_APP.md](docs/STREAMLIT_APP.md) for full architecture and data contract details.

## Repository Structure

```text
config.yaml                 Model assumptions and parameters
src/quantedge/              Data, estimation, portfolio, KPI, VaR, export modules
data/raw/                   Download cache (ignored by Git)
data/processed/             Return series (ignored by Git)
outputs/tables/             CSV and Markdown analysis outputs
outputs/models/             Selected portfolio JSON
docs/                       Methodology, governance, living notes, run report
tests/                      Unit and integration tests
app/                        Streamlit dashboard (schemas, data_access, charts, views)
notebooks/                  Analysis notebook placeholders
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run Analysis

Run from the repository root. The pipeline downloads Yahoo Finance adjusted close data and the public FRED DGS3MO series; network access is required on the first run.

```bash
python -m src.quantedge.main
```

Raw prices and the risk-free series are cached in `data/raw/` when enabled in `config.yaml`. Review all generated diagnostics before relying on portfolio estimates.

To run the new analytical layers against already saved prices and portfolio weights:

```bash
python -m src.quantedge.analytics
```

This downloads the public FRED history. For a fully offline rerun after that first download:

```bash
python -m src.quantedge.analytics --prices outputs/tables/prices.csv --risk-free data/raw/risk_free.csv
```

To also regenerate the current allocation and original VaR/ES exports entirely from those snapshots:

```bash
python -m src.quantedge.main --prices outputs/tables/prices.csv --risk-free data/raw/risk_free.csv
```

The default validation split fits portfolio weights on 2007–2017, including the financial crisis, and freezes them for 2018 onward. Current-allocation estimation also begins in 2007. Risk models update using prior observations only. The 2008 stress replay now overlaps weight-estimation data and is a retrospective diagnostic, not an independent crisis test. Including a crisis in training does not guarantee smaller future losses. If the training-only return target is infeasible, the run fails explicitly rather than using full-sample weights. Configure the split, EVT thresholds, and stress assumptions in `config.yaml`. See [the analytical layer guide](docs/ANALYTICAL_LAYERS.md) for definitions and output schemas.

## Run Tests

```bash
pytest
```

Separate candidate risk models and their comparisons live in [experiments/improved_risk](experiments/improved_risk/README.md). They have their own configuration, tests and output directory and do not replace the baseline models. Run the offline comparison with `python -m experiments.improved_risk.run`. These are development comparisons on an already inspected historical period, not a new untouched holdout.

See the [experimental results review](experiments/improved_risk/RESULTS_REVIEW.md) for the current training period, coverage and independence results. The separate [alert-response experiments](experiments/improved_risk/results/alert_breach_comparison/README.md) compare filtered historical, filtered bootstrap Monte Carlo and filtered EVT with the unchanged references. At 99%, the alert variants have 22, 22 and 25 exceptions respectively, and none is rejected by Kupiec, independence or joint coverage at 5% on this already-inspected development period. Filtered historical and bootstrap alert variants were also checked at 90% and 95%; the EVT alert variant has not been checked at those levels. These three models share the same alert rule, their favourable results are not independent validation, and they remain outside the main pipeline. No candidate is automatically promoted.

## Method Summary

Layer 1 estimates expected returns from a 50/50 blend of historical estimates and CAPM priors, shrinks annualized log-return covariance toward its diagonal, and solves a long-only minimum-variance problem for each feasible target return. Client constraints cap equity and single-asset exposures, set minimum defensive exposure, and constrain EEM, XLK, XLE, and GLD. KPIs use simple portfolio returns and include return, volatility, Sharpe, Sortino, Treynor, Jensen alpha, information ratio, drawdown, and Calmar ratio.

Layer 2 calculates 90%, 95%, and 99% one-day VaR forecasts. Every forecast uses information available through the previous trading day. VaR is reported as a positive loss threshold, with exceptions recorded for backtesting. It is not a guarantee of maximum loss; tail events can exceed VaR.

VaR exports also include ES. Formal validation uses the separate `risk_forecasts_oos.csv`, with a common evaluation period across all four models. The original `var_forecasts.csv` is a retrospective replay of the current allocation, not an out-of-sample portfolio backtest.

## Data Sources

- Yahoo Finance through `yfinance`, using adjusted close prices only.
- FRED public CSV endpoint, series DGS3MO for the short-rate input.

The engine does not use proprietary data. Data are not forward-filled; missing observations and robust-MAD outlier flags are reported. Flagged values are not silently deleted.

## Main Outputs

- `outputs/tables/portfolio_weights.csv`
- `outputs/tables/efficient_frontier.csv`
- `outputs/tables/kpi_summary.csv`
- `outputs/tables/data_quality_report.csv` and `adf_report.csv`
- `outputs/tables/var_summary.csv`, `var_forecasts.csv`, and `var_method_comparison.md`
- `outputs/tables/streamlit_portfolio_summary.csv` and `streamlit_var_dashboard.csv`
- `outputs/models/selected_portfolio.json`
- `outputs/tables/risk_forecasts_oos.csv`, `risk_summary_oos.csv`, and `backtest_results.csv`
- `outputs/tables/evt_threshold_diagnostics.csv` (candidate-threshold sensitivity and fitted parameters)
- `outputs/tables/oos_portfolio_weights.csv`, `oos_performance.csv`, and `oos_kpi_summary.csv`
- `outputs/tables/stress_summary.csv`, `stress_paths.csv`, and `stress_shock_contributions.csv`
- `outputs/tables/risk_contributions.csv` and `outputs/models/analytical_run.json`
- Living run records in `docs/`

## Assumptions and Limitations

Expected returns are noisy and sample-dependent. The original retrospective KPIs use the latest DGS3MO quote; OOS KPIs use an as-of daily risk-free history. Complete observations across the eligible universe are required. The engine excludes fees, taxes, liquidity constraints and transaction costs. Historical and bootstrap VaR cannot generate losses outside observed data; parametric VaR assumes normal log returns. EVT estimates are sensitive to thresholds and clustered extremes. The hypothetical stress is an assumed scenario, not a prediction. Stress replay and current-position risk contributions describe today's allocation; formal OOS validation uses its separately frozen training allocation.

## AI Usage Disclosure

Known assistance in this development session: OpenAI Codex supported code review, analytical implementation, automated tests, documentation, and pipeline execution. Automated checks included formula reference values, forecast timing, future-data isolation, risk-contribution sums, and end-to-end exports.

Team review and independent human validation remain pending. Before submission, the team must complete this disclosure with all other AI tools used, identify material assisted components, and describe the actual human checks and revisions. Automated tests are not a substitute for the team's ability to explain and defend its methods and conclusions.

## Remaining Work

| Priority | Work | Completion evidence |
|---|---|---|
| 1 | Review calibration failures and EVT warnings; justify model assumptions and parameter choices | Explain model disagreements, exception clustering and threshold sensitivity; freeze any chosen rule and evaluate it on future, uninspected observations |
| 2 | Complete portfolio decision analysis | Justify investment horizon, the 6.5% return objective, exposure limits and the recommended allocation; compare alternatives using downside risk, stress losses and position risk contributions |
| 3 | Improve the efficient-frontier demonstration | Choose feasible targets that actually change allocations; the current saved target grid produces the same minimum-variance portfolio at every point |
| 4 | Restore and integrate the Streamlit draft | Confidence/method/date controls; allocation, wealth and drawdown; VaR/ES/EVT comparison; backtests and traffic light; stress results; assumptions and limitations |
| 5 | Complete data-quality checks | Add explicit stale/flat-price and structural-break diagnostics alongside the existing missing-data and extreme-return flags, with documented human review |
| 6 | Prepare competition deliverables | A 7–15-page technical report/investment memo, a one-page executive tear sheet and actionable recommendation, and the pitch deck if selected for the final |
| 7 | Verify submission reproducibility and disclosure | Record dependency versions, verify a clean environment and the documented app launch command, retain source-data provenance, and complete the human-reviewed AI disclosure |

Possible extensions after these priorities include uncertainty intervals, volatility models beyond EWMA, and transaction-cost/rebalancing sensitivity. The separate filtered EVT and alert-response experiments need review and future evaluation before any integration. Further model changes should not be selected merely to make existing backtest p-values pass.

## Living Documentation

See [docs/LIVING_PROJECT_NOTES.md](docs/LIVING_PROJECT_NOTES.md) for current decisions and change history, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for method limitations, and [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) for source and cleaning policy.
