# Streamlit Application Architecture & Data Contracts

## 1. Overview and Launch

The Streamlit dashboard presents the Serendib University Endowment Fund investment proposal (reference value: USD 10,000,000). It serves as an institutional decision dashboard for the investment committee, answering:
1. Where would the money be invested?
2. Why was this allocation selected?
3. How did the portfolio behave historically?
4. How much could it lose?
5. How reliable were its risk forecasts?
6. What happens during crises?
7. Which positions contribute most to risk?
8. What assumptions and limitations affect the conclusions?

### Launch Command

Run from the repository root:
```bash
python -m streamlit run app/streamlit_app.py
```

The application runs entirely offline from pre-computed analytical outputs under `outputs/tables/` and `outputs/models/` without triggering market data downloads or pipeline reruns.

---

## 2. Page Structure

The application is structured into 6 modular views under `app/views/`:

1. **Overview (`app/views/overview.py`)**:
   - Reference notional (USD 10,000,000).
   - Core distinction: Model Expected Return (7.66%) vs Realized Historical Compound Return (7.58%).
   - Model Annual Volatility (8.57%) vs Realized Historical Volatility (9.19%).
   - Latest complete market observation date (2026-10-20).
   - Proposed asset allocation summary.
   - What these results mean for the endowment.
   - Mandatory investment committee status notice: *"Team investment recommendation pending"*.

2. **Portfolio & Performance (`app/views/portfolio.py`)**:
   - Asset allocation donut chart and holdings table (ticker, description, target weight, dollar allocation).
   - Broad asset class exposures vs mandate bounds (Equity: 40.0% vs max 60.0%; Defensive AGG: 45.0% vs min 35.0%; Gold GLD: 15.0% vs 5–15% bounds).
   - Cumulative wealth trajectory ($10M growth) vs Policy Benchmark (35% SPY, 10% EFA, 5% EEM, 40% AGG, 10% GLD).
   - Drawdown (underwater) comparison chart.
   - Complete KPI card suite: Sharpe, Sortino, Treynor, Jensen's Alpha, Information Ratio, Calmar, Beta, and risk-free conventions.
   - Efficient frontier diagnosis: Explains why targets from 5.0% to 7.5% collapse to the identical minimum-variance solution.

3. **Risk Analysis (`app/views/risk.py`)**:
   - VaR (loss threshold) vs Expected Shortfall (tail conditional expectation) definition.
   - Latest 1-day dollar and percentage loss metrics ($96,938 / 0.97% VaR; $133,231 / 1.33% ES at 95% historical).
   - Methodology comparison table: Historical Simulation, Bootstrap Monte Carlo, EWMA Parametric Normal, and EVT-GPD.
   - Daily VaR/ES evolution time series chart.
   - Cross-methodology loss comparison bar chart at common latest date.
   - EVT threshold diagnostics and fit stability heuristics.

4. **Model Validation (`app/views/validation.py`)**:
   - Locked to **Frozen validation portfolio** (OOS evaluation 2018–2026, 2,212 days).
   - Core visual: Realized daily losses vs forecast VaR with positive loss convention and breach markers (gains visible as negative losses below zero).
   - Formal likelihood-ratio hypothesis tests: Kupiec POF test (coverage) and Christoffersen independence test.
   - Dynamic recomputation: Selecting a custom evaluation date window re-evaluates statistical tests via backend formulas.
   - Basel Regulatory Traffic Light: Evaluated strictly at 99% confidence over trailing 250 days.
   - Governance Insight: Explains why recent 250-day Green traffic lights coexist with multi-year full-history test rejections.

5. **Stress Testing & Risk Contributions (`app/views/stress.py`)**:
   - Historical crisis replays: GFC 2008 (-21.03%, -$2,102,601), COVID-19 crash (-15.28%, -$1,528,265), and 2022 rate shock (-12.18%, -$1,218,238).
   - Interactive crisis trajectory visualizer (wealth index and drawdown curves).
   - Retrospective replay warning disclosure.
   - Hypothetical Stagflation Shock: Team-assumed simultaneous shocks (SPY -25%, EFA -28%, EEM -35%, AGG -12%, GLD +8%, XLE -15%, XLK -35%), producing -$1,499,218 net impact (-14.99%) and demonstrating gold hedging.
   - Euler Risk Budgeting: Marginal VaR/ES, Component VaR/ES, and Risk Shares across assets.
   - Diverging horizontal bar chart preserving negative hedging risk contributions.

6. **Assumptions & Data Quality (`app/views/assumptions.py`)**:
   - Three distinct date cutoffs: Latest Price Date (2026-10-22), Latest Complete Return Date (2026-10-20), Latest Evaluated Risk Date (2026-10-20).
   - Discrepancy detector: Highlights missing EFA and AGG prices on 2026-10-21 affecting October 22 returns.
   - Data quality report: Missing price days and 8-MAD robust outlier flags.
   - Augmented Dickey-Fuller (ADF) stationarity report: 7 of 7 assets stationary at 1%.
   - Full modeling assumptions and explicit limitations (fees, costs, slippage excluded; static notional; tail model constraints; AI assistance disclosure).

---

## 3. Strict Separation of Two Portfolios

The application enforces strict separation between:

| Dimension | Current Proposed Portfolio | Frozen Validation Portfolio |
|---|---|---|
| **Weights Source** | `outputs/models/selected_portfolio.json`, `portfolio_weights.csv` | `outputs/tables/oos_portfolio_weights.csv`, `analytical_run.json` |
| **Optimization Sample** | 2007-01-04 to 2026-10-22 | 2007-01-04 to 2017-12-29 (training only) |
| **Holding Horizon** | 2007-01-04 to 2026-10-20 (4,980 days) | 2018-01-02 to 2026-10-20 (2,212 days) |
| **Analysis Type** | Retrospective historical replay | Out-of-sample forward evaluation |
| **Risk-Free Rate** | Fixed quote (4.22% DGS3MO) | As-of daily historical series (avg 2.74%) |
| **Available VaR Models** | Historical, EWMA Normal, Bootstrap | Historical, EWMA Normal, Bootstrap, EVT-GPD |
| **Pages Where Used** | Overview, Portfolio, Risk, Stress, Assumptions | Portfolio, Risk, Validation, Assumptions |

---

## 4. Sidebar Controls and Dynamics

- **Portfolio Basis**: Selects between Current proposed portfolio and Frozen validation portfolio on pages supporting both. Locked automatically on Validation and Stress pages.
- **Confidence Level**: Controls 1-day VaR/ES confidence (90%, 95%, 99%). Filters corresponding forecast rows without affecting portfolio weights.
- **Risk Methodology**: Selects model (Historical, EWMA Normal, Bootstrap, EVT). Filtered dynamically based on available models in the active dataset.
- **OOS Evaluation Window**: On the Model Validation page, allows setting custom start and end dates. Submitting the form recomputes Kupiec POF, Christoffersen independence, and Basel traffic lights via backend functions.
- **Reload Results**: Clears Streamlit `@st.cache_data` and reloads CSV/JSON artifacts from disk.

---

## 5. Software Architecture

```
app/
├── streamlit_app.py        # Streamlit entry point, layout, and routing
├── schemas.py              # Constants, asset registry, method labels, palette
├── formatting.py           # Currency, percentage, ratio, date formatting, boolean parsing
├── data_access.py          # Cached data loaders, validation, and dynamic backtesting
├── charts.py               # Interactive Plotly figures with institutional styling
├── components.py           # Reusable cards, badges, callouts, and download buttons
└── views/
    ├── __init__.py
    ├── overview.py         # Executive overview view
    ├── portfolio.py        # Portfolio holdings, growth, and frontier view
    ├── risk.py             # VaR, Expected Shortfall, and EVT view
    ├── validation.py       # Formal backtesting, Kupiec, Christoffersen, traffic light view
    ├── stress.py           # Crisis replay, stagflation shock, risk contributions view
    └── assumptions.py      # Data governance, cutoffs, ADF, and limitations view
```
