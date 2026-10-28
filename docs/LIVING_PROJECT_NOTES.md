# Living Project Notes

## Last Updated
- Date: 2026-10-28
- Agent: Rashad / Antigravity IDE
- Branch: `rash`

## Current Status
- What is working: all five analytical layers, stress tests, position risk contributions, full interactive Streamlit decision dashboard across 6 dedicated views (`app/streamlit_app.py`, `app/views/`), and comprehensive 20-scenario test suite; 87 software, scenario, and integration tests pass.
- What is incomplete: human team-authored investment memo, final presentation tear sheet, and human fiduciary sign-off.
- What is blocked: none known.

## Repository Overview
- Purpose: conservative USD 10 million endowment portfolio analysis.
- Main entry points: `python -m src.quantedge.main`.
- Main outputs: CSV tables in `outputs/tables/` and selected portfolio JSON in `outputs/models/`.

## Architecture
- data layer: Yahoo Finance adjusted close and public FRED data.
- estimation layer: historical/CAPM blended means; diagonal shrinkage covariance.
- portfolio layer: constrained target-return minimum variance and frontier.
- VaR layer: Historical, EWMA Parametric, and bootstrap Monte Carlo.
- exports layer: CSV/Parquet tables and JSON portfolio artifact.

## Key Decisions
| Decision | Choice | Why | Alternatives Considered |
|---|---|---|---|
| Optimizer objective | Target-return minimum variance | Prioritizes capital preservation | Maximum Sharpe |
| Expected returns | 50/50 historical and CAPM blend | Reduces dependence on noisy sample means | Historical-only estimate |
| Covariance | Shrink toward diagonal | Stabilizes sample correlations | Raw sample covariance |

## Data Governance
- Source: Yahoo Finance via yfinance; FRED DGS3MO.
- Start date: 2007-01-01; current estimation and validation training now begin in 2007 to include the financial crisis. Validation training ends in 2017; evaluation begins in 2018.
- Adjusted close used: yes.
- Calendar alignment: no forward fill; complete cases for portfolio calculations.
- Missing data treatment: retained in source outputs and reported.
- Outlier flagging: robust MAD threshold; no outlier deletion.
- ADF results: see the regenerated `outputs/tables/adf_report.csv` for the current 2007-onward estimation sample. The earlier 2010-onward run rejected the unit-root null for all seven assets at 1%.
- Data quality: 2 missing price cells; 95 potential extreme-return flags across assets. None were deleted.

## Layer 1 Summary
- Expected return method: historical/CAPM blend.
- Covariance method: annualized log-return covariance with diagonal shrinkage.
- Optimizer objective: constrained minimum variance at target return.
- Constraints used: equity and defensive exposure, EEM/XLK/XLE limits, GLD bounds, and single-position cap.
- Selected target return: 6.50% configured target.
- Final weights (current 2007-start estimation): SPY 22.81%, EFA 9.61%, EEM 0.00%, AGG 45.00%, GLD 15.00%, XLE 1.27%, XLK 6.31%.
- Estimated annual return / volatility: 7.66% / 8.57%; see regenerated KPI exports for realized performance.

## Layer 2 Summary
- Historical VaR: rolling empirical quantile.
- Parametric VaR: zero-mean normal with EWMA volatility.
- Monte Carlo VaR: bootstrap from trailing return window.
- Forecast alignment: one step ahead; previous observations only.
- Exception logic: actual log return below negative VaR log loss.
- Latest VaR and ES: see regenerated `outputs/tables/var_summary.csv`; prior numerical summaries describe the older 2010-start allocation.
- Output files: `outputs/tables/var_summary.csv`, `var_forecasts.csv`, and `var_method_comparison.md`.

## Risk-model development findings
- The unchanged four baseline methods remain in the main pipeline. At 99% VaR on the frozen 2007–2017 allocation and 2,212 evaluation days, raw EVT has 31 exceptions, historical 38, bootstrap Monte Carlo 32 and normal EWMA 48. None avoids rejection by the joint conditional-coverage test.
- Separate experiments show 25 exceptions for filtered EVT + next-day alert and 22 each for filtered historical + alert and filtered bootstrap + alert. All three are not rejected by the recorded 99% Kupiec, independence and joint tests at 5% **on the previously inspected development period**. The filtered empirical alert variants are also assessed at 90% and 95%; the EVT alert variant is only assessed at 99%.
- These three candidates share the same training-calibrated prior-close CUSUM rule. Filtered historical and bootstrap avoid the same seven breaches, so their matching outcomes are not independent evidence. Alerts raise VaR on 124 evaluation days. ES is calculated but has no separate ES backtest. Do not present these results as prospective validation or as main-pipeline replacements.
- See [experimental results](../experiments/improved_risk/RESULTS_REVIEW.md), the [EVT response review](../experiments/improved_risk/results/evt_response/REVIEW.md), the [filtered response review](../experiments/improved_risk/results/filtered_response/REVIEW.md), and the [breach comparison chart](../experiments/improved_risk/results/alert_breach_comparison/README.md).

## Known Issues
- Retrospective KPIs retain the latest risk-free quote; OOS KPIs now align a historical as-of rate series.
- Several unchanged baseline OOS calibration tests reject, especially at 99%; green recent traffic lights do not override failures over the full evaluation period. Alert-candidate non-rejection on the inspected period is not independent validation.
- Current raw EVT: 1 of 106 refits use the warning fallback; inspect threshold diagnostics. Tail dependence and threshold uncertainty remain limitations.

## Next Steps for Teammates
- Review the tail fits, baseline rejections and alert-candidate limitations before defending the methodology. Assess the EVT alert at 90% and 95%, consider an ES-specific check, and log frozen forecasts prospectively for independent evaluation.
- Streamlit UI fully verified across all 6 views with 87 passing tests; ready for team review and competition rehearsal.
- Write the investment memo, tear sheet and final AI-use disclosure after human validation.

## Change Log
| Date | Agent | Change | Files Affected | Reason |
|---|---|---|---|---|
| 2026-10-28 | Rashad | Implemented full Streamlit decision dashboard across 6 views, robust data access layer, dynamic backtesting filter, Plotly charts, and test suite; 87 software tests pass | `app/`, `tests/`, `requirements.txt`, `docs/STREAMLIT_APP.md`, `README.md` | Deliver complete, interactive institutional investment dashboard for Serendib Endowment |
| 2026-10-28 | Codex | Added breach diagnosis, fixed volatility probe, separate EVT and filtered empirical next-day alert experiments, side-by-side breach plots and documentation; 42 software tests pass | `experiments/improved_risk/`, README, docs | Investigate clustered exceptions and compare causal responses without replacing baseline outputs |
| 2026-10-27 | Codex | Added isolated filtered historical/bootstrap, EWMA-Student-t and filtered EVT candidates plus fixed sensitivities; compared 11 variants on identical dates; 31 software tests pass | `experiments/improved_risk/`, README | Investigate calibration without replacing or tuning the baselines to pass |
| 2026-10-26 | Codex | Stashed the UI draft; implemented ES, EVT, frozen-allocation OOS validation, stress tests and Euler contributions; ran snapshot pipeline and 18 tests | `src/`, tests, `config.yaml`, outputs, docs | Complete required analytical layers with explicit timing and diagnostic evidence |
| 2026-10-23 | Codex | Added configurable Layers 1 and 2 foundation and team documentation | `src/`, `config.yaml`, `docs/`, tests | Establish reproducible analysis engine |
| 2026-10-23 | Codex | Ran pipeline on public data and recorded portfolio, data quality, ADF, VaR, and KPI results | `outputs/`, `docs/` | Establish an auditable initial run |

| 2026-10-23 | Codex | Ran configured pipeline; target 6.50%, expected return 8.02%, volatility 7.57% | outputs/, docs/ | Record reproducible results |

| 2026-10-26 | Codex | Ran configured pipeline; target 6.50%, expected return 8.01%, volatility 7.57% | outputs/, docs/ | Record reproducible results |

| 2026-10-27 | Codex | Ran configured pipeline; target 6.50%, expected return 7.66%, volatility 8.57% | outputs/, docs/ | Record reproducible results |

## 2007-start update
- Both current estimation and validation training now include the financial crisis; validation still ends in 2017 and testing starts in 2018.
- All 42 current software tests pass, including a regression demonstrating that 2008 observations affect the fitted allocation and tests of next-day alert timing.
- The 2008 replay now overlaps estimation data; it is a retrospective diagnostic.
- See [TRAINING_START_COMPARISON.md](TRAINING_START_COMPARISON.md) for before/after weights and scenario losses.

| 2026-10-27 | Codex | Ran configured pipeline; target 6.50%, expected return 7.66%, volatility 8.57% | outputs/, docs/ | Record reproducible results |
