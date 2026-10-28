# Agent Run Report

## Request
Implement the Quant Edge analytical layers, including ES, EVT, out-of-sample validation, stress testing and risk contributions.

## Work completed
- Implemented data access, cleaning, ADF diagnostics, return estimation, constrained optimization, investment KPIs, VaR/ES, EVT threshold diagnostics, formal OOS backtests, stress tests, risk contributions and exports.
- Created: config, source package, methodology, data governance, living notes, report, tests, app placeholder, and notebook placeholders.
- Dependencies added: only the packages listed in requirements.txt from the requested specification.

## Run evidence
- Price period: 2007-01-03 to 2026-10-22.
- Data quality: 2 missing price cells and 95 flagged extreme returns.
- ADF: 7 of 7 series stationary at 1%.
- Weights: {'SPY': 0.22807378821751478, 'EFA': 0.0961127229084155, 'EEM': 8.239936510889834e-18, 'AGG': 0.44999999999999996, 'GLD': 0.14999999999999997, 'XLE': 0.012714857253651284, 'XLK': 0.0630986316204186}.
- Expected return: 7.6650%; annualized volatility: 8.5689%.
- VaR summary: [{'method': 'historical', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.007198552488964965, 'latest_var_usd': 71985.52488964965, 'latest_es_pct': 0.01096432712838719, 'latest_es_usd': 109643.27128387189, 'average_var_pct': 0.005740677916626281, 'number_of_exceptions': 499, 'observations': 4728}, {'method': 'historical', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.009693793704106488, 'latest_var_usd': 96937.93704106487, 'latest_es_pct': 0.013323088223895534, 'latest_es_usd': 133230.88223895535, 'average_var_pct': 0.008184662803778433, 'number_of_exceptions': 270, 'observations': 4728}, {'method': 'historical', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.01598776573606651, 'latest_var_usd': 159877.65736066512, 'latest_es_pct': 0.018673552431991848, 'latest_es_usd': 186735.52431991848, 'average_var_pct': 0.014125323875819388, 'number_of_exceptions': 73, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.007290072747069535, 'latest_var_usd': 72900.72747069535, 'latest_es_pct': 0.011065200661292282, 'latest_es_usd': 110652.00661292282, 'average_var_pct': 0.005792402634756307, 'number_of_exceptions': 491, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.009901198414648336, 'latest_var_usd': 99011.98414648336, 'latest_es_pct': 0.013360789197372597, 'latest_es_usd': 133607.89197372596, 'average_var_pct': 0.008288974989560825, 'number_of_exceptions': 262, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.0160095059731388, 'latest_var_usd': 160095.059731388, 'latest_es_pct': 0.01880810334338235, 'latest_es_usd': 188081.0334338235, 'average_var_pct': 0.014817891369242537, 'number_of_exceptions': 65, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.006096374206815566, 'latest_var_usd': 60963.74206815566, 'latest_es_pct': 0.008337180730802696, 'latest_es_usd': 83371.80730802695, 'average_var_pct': 0.006378252298820177, 'number_of_exceptions': 420, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.007817840320223826, 'latest_var_usd': 78178.40320223826, 'latest_es_pct': 0.009792577560465698, 'latest_es_usd': 97925.77560465698, 'average_var_pct': 0.008176250470079157, 'number_of_exceptions': 257, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.011038986212363985, 'latest_var_usd': 110389.86212363985, 'latest_es_pct': 0.012635686876236618, 'latest_es_usd': 126356.86876236618, 'average_var_pct': 0.011537002848085474, 'number_of_exceptions': 96, 'observations': 4728}].
- Tests: run `pytest`.

## Assumptions made
- Retrospective KPIs use the latest DGS3MO quote; OOS KPIs use as-of historical quotes.
- All configured tickers are required; absent downloads fail the pipeline.
- Observations with any missing asset return are omitted for portfolio-level calculations.
- Historical, CAPM, and diagonal-shrinkage approaches follow the provided specification.

## Known issues and recommendations
Refresh public data before relying on outputs; inspect flagged outliers, infeasible targets, formal backtests and EVT stability warnings. Analytical layers include ES, EVT, OOS validation, stress tests and risk contributions. Review ANALYTICAL_LAYERS.md and acknowledge that fees, taxes, liquidity and transaction costs are excluded.

## Subsequent risk-model development (2026-10-28)

This run report describes the original analytical pipeline. Later experiments remain separate under [`experiments/improved_risk/`](../experiments/improved_risk/README.md). With 2007-start training and 2018–2026 evaluation, the unchanged baseline methods still fail joint 99% conditional coverage. Filtered historical + alert and filtered bootstrap + alert each record 22 exceptions; filtered EVT + alert records 25. All three are not rejected by the recorded 99% Kupiec, independence and joint tests on this already-inspected development period. They share one prior-close CUSUM alert and have not replaced the four baseline exports. The [experiment results review](../experiments/improved_risk/RESULTS_REVIEW.md) and [breach plots](../experiments/improved_risk/results/alert_breach_comparison/README.md) give exact scores, dates, forecast-size impact and limitations. The current repository test suite passes 42 tests; this does not independently validate risk forecasts.

## Streamlit Application Implementation (2026-10-28)

- **Agent:** Rashad / Antigravity IDE
- **Branch:** `rashad/streamlit-app`
- **Request:** Build the complete interactive Streamlit application for the Serendib Endowment investment proposal.
- **Work Completed:**
  - Designed and implemented complete application across 6 views: `app/views/overview.py`, `portfolio.py`, `risk.py`, `validation.py`, `stress.py`, and `assumptions.py`.
  - Implemented robust data access and validation adapter (`app/data_access.py`), institutional formatting utilities (`app/formatting.py`), domain schemas (`app/schemas.py`), interactive Plotly charts (`app/charts.py`), and reusable UI components (`app/components.py`).
  - Added dynamic out-of-sample backtest window recomputation calling backend hypothesis test functions (`kupiec_test`, `christoffersen_test`, `basel_traffic_light`).
  - Implemented 14 new test cases in `tests/test_app_data_access.py` and `tests/test_app_integration.py`. All 39 tests pass.
  - Successfully launched local server and visually inspected all 6 views with browser automation (`verify_streamlit_app`). Confirmed zero tracebacks, zero errors, clean layout, correct dates, and active filters.
  - Documented application architecture in `docs/STREAMLIT_APP.md` and updated `README.md` and `docs/LIVING_PROJECT_NOTES.md`.
