# Quant Edge 1.0 — Serendib Endowment Portfolio

This repository contains the quantitative analysis engine for the Quant Edge 1.0 Final Challenge. It constructs and evaluates a conservative USD 10 million portfolio for the Serendib University Endowment Fund. The mandate prioritizes capital preservation, downside measurement, and resilience over maximizing expected return.

## Team Scope

- **Layer 1:** portfolio construction, CAPM diagnostics, and investment KPIs.
- **Layer 2:** one-step-ahead Historical, EWMA Parametric, and Bootstrap Monte Carlo VaR.
- Later extensions: Expected Shortfall, EVT, formal backtesting, and a Streamlit dashboard.

The eligible universe is SPY, EFA, EEM, AGG, GLD, XLE, and XLK. Weights are determined by the configured constrained target-return minimum-variance optimization.

## Repository Structure

```text
config.yaml                 Model assumptions and parameters
src/quantedge/              Data, estimation, portfolio, KPI, VaR, export modules
data/raw/                   Download cache (ignored by Git)
data/processed/             Return series (ignored by Git)
outputs/tables/             CSV and Markdown analysis outputs
outputs/models/             Selected portfolio JSON
docs/                       Methodology, governance, living notes, run report
tests/                      Unit tests
app/                         Streamlit integration placeholder
notebooks/                   Analysis notebook placeholders
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

## Run Tests

```bash
pytest
```

## Method Summary

Layer 1 estimates expected returns from a 50/50 blend of historical estimates and CAPM priors, shrinks annualized log-return covariance toward its diagonal, and solves a long-only minimum-variance problem for each feasible target return. Client constraints cap equity and single-asset exposures, set minimum defensive exposure, and constrain EEM, XLK, XLE, and GLD. KPIs use simple portfolio returns and include return, volatility, Sharpe, Sortino, Treynor, Jensen alpha, information ratio, drawdown, and Calmar ratio.

Layer 2 calculates 90%, 95%, and 99% one-day VaR forecasts. Every forecast uses information available through the previous trading day. VaR is reported as a positive loss threshold, with exceptions recorded for backtesting. It is not a guarantee of maximum loss; tail events can exceed VaR.

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
- Living run records in `docs/`

## Assumptions and Limitations

Expected returns are noisy and sample-dependent. The latest DGS3MO quote is used as a constant risk-free rate for the current run. The portfolio return series uses dates with complete observations across the eligible universe. The current engine excludes fees, taxes, liquidity constraints, transaction costs, and forward-looking shocks. Historical and bootstrap VaR cannot generate losses outside observed data; parametric VaR depends on a normality assumption. VaR should be supplemented with Expected Shortfall, stress analysis, and formal backtesting before operational use.

## AI Usage Disclosure

Placeholder: document AI tools used, their contribution, and human review according to competition rules.

## Living Documentation

See [docs/LIVING_PROJECT_NOTES.md](docs/LIVING_PROJECT_NOTES.md) for current decisions and change history, [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for method limitations, and [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) for source and cleaning policy.
