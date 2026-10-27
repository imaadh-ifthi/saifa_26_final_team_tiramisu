# Living Project Notes

## Last Updated
- Date: 2026-10-27
- Agent: Codex
- Commit hash / branch: see Git history on `main`

## Current Status
- What is working: all five analytical layers, historical/hypothetical stress tests, position risk contributions, and 31 passing tests including separate experimental candidates.
- What is incomplete: integration with the preserved UI draft and team-authored investment interpretation.
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
- Start date: 2007-01-01; estimation from 2010-01-01.
- Adjusted close used: yes.
- Calendar alignment: no forward fill; complete cases for portfolio calculations.
- Missing data treatment: retained in source outputs and reported.
- Outlier flagging: robust MAD threshold; no outlier deletion.
- ADF results: all 7 return series rejected the unit-root null at 1% in the 2010-2026 estimation sample.
- Data quality: 2 missing price cells; 95 potential extreme-return flags across assets. None were deleted.

## Layer 1 Summary
- Expected return method: historical/CAPM blend.
- Covariance method: annualized log-return covariance with diagonal shrinkage.
- Optimizer objective: constrained minimum variance at target return.
- Constraints used: equity and defensive exposure, EEM/XLK/XLE limits, GLD bounds, and single-position cap.
- Selected target return: 6.50% configured target.
- Final weights: SPY 23.17%, EFA 11.84%, EEM 1.33%, AGG 45.00%, GLD 15.00%, XLE 2.48%, XLK 1.17%.
- Estimated annual return / volatility: 8.02% / 7.57%; realized KPI annual return / volatility: 7.07% / 9.26%.

## Layer 2 Summary
- Historical VaR: rolling empirical quantile.
- Parametric VaR: zero-mean normal with EWMA volatility.
- Monte Carlo VaR: bootstrap from trailing return window.
- Forecast alignment: one step ahead; previous observations only.
- Exception logic: actual log return below negative VaR log loss.
- Latest VaR: Historical 95% $95,463 and 99% $155,901; EWMA 95% $74,778 and 99% $105,596; bootstrap 95% $102,672 and 99% $159,980 (as of 2026-10-20).
- Output files: `outputs/tables/var_summary.csv`, `var_forecasts.csv`, and `var_method_comparison.md`.

## Known Issues
- Retrospective KPIs retain the latest risk-free quote; OOS KPIs now align a historical as-of rate series.
- Several OOS calibration tests reject, especially at 99%; green recent traffic lights do not override failures over the full evaluation period.
- One EVT refit lacks adjacent-threshold stability and is explicitly flagged for review. Tail dependence and threshold uncertainty remain limitations.

## Next Steps for Teammates
- Review the tail fits and model rejections before defending the methodology.
- Integrate the documented analytical CSV contracts into the preserved UI draft.
- Write the investment memo, tear sheet and final AI-use disclosure after human validation.

## Change Log
| Date | Agent | Change | Files Affected | Reason |
|---|---|---|---|---|
| 2026-10-27 | Codex | Added isolated filtered historical/bootstrap, EWMA-Student-t and filtered EVT candidates plus fixed sensitivities; compared 11 variants on identical dates; 31 software tests pass | `experiments/improved_risk/`, README | Investigate calibration without replacing or tuning the baselines to pass |
| 2026-10-26 | Codex | Stashed the UI draft; implemented ES, EVT, frozen-allocation OOS validation, stress tests and Euler contributions; ran snapshot pipeline and 18 tests | `src/`, tests, `config.yaml`, outputs, docs | Complete required analytical layers with explicit timing and diagnostic evidence |
| 2026-10-23 | Codex | Added configurable Layers 1 and 2 foundation and team documentation | `src/`, `config.yaml`, `docs/`, tests | Establish reproducible analysis engine |
| 2026-10-23 | Codex | Ran pipeline on public data and recorded portfolio, data quality, ADF, VaR, and KPI results | `outputs/`, `docs/` | Establish an auditable initial run |

| 2026-10-23 | Codex | Ran configured pipeline; target 6.50%, expected return 8.02%, volatility 7.57% | outputs/, docs/ | Record reproducible results |

| 2026-10-26 | Codex | Ran configured pipeline; target 6.50%, expected return 8.01%, volatility 7.57% | outputs/, docs/ | Record reproducible results |
