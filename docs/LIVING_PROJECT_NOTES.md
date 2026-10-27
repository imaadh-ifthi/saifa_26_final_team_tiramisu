# Living Project Notes

## Last Updated
- Date: 2026-10-23
- Agent: Codex
- Commit hash / branch: see Git history on `main`

## Current Status
- What is working: config-driven Layers 1 and 2 pipeline, public-data run, and unit tests (3 passed).
- What is incomplete: Expected Shortfall, EVT, formal backtesting, and interactive dashboard remain future layers.
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
- Issue: risk-free rate is the latest DGS3MO observation, treated as constant. Impact: KPI risk-free adjustment is not time varying. Proposed fix: align full history to portfolio dates.
- Issue: VaR does not quantify losses beyond the threshold. Impact: tail severity is omitted. Proposed fix: Expected Shortfall, EVT, and stress tests.

## Next Steps for Teammates
- Bhaveenthan: extend Expected Shortfall and EVT.
- Rashad: implement formal VaR backtests.
- Daniel / Daniru: integrate the tidy dashboard artifacts.

## Change Log
| Date | Agent | Change | Files Affected | Reason |
|---|---|---|---|---|
| 2026-10-23 | Codex | Added configurable Layers 1 and 2 foundation and team documentation | `src/`, `config.yaml`, `docs/`, tests | Establish reproducible analysis engine |
| 2026-10-23 | Codex | Ran pipeline on public data and recorded portfolio, data quality, ADF, VaR, and KPI results | `outputs/`, `docs/` | Establish an auditable initial run |

| 2026-10-23 | Codex | Ran configured pipeline; target 6.50%, expected return 8.02%, volatility 7.57% | outputs/, docs/ | Record reproducible results |
