# Layer 3-5 upgrade (Expected Shortfall, EVT, backtesting)

This builds on the existing Layer 3-5 work. The earlier notes listed three open items:
GARCH-based filtering, ES-specific backtests, and a formal model comparison. This folder covers
all three. Nothing in `outputs/` or `experiments/improved_risk/` is changed.

## How to run

From the repository root:

```bash
python -m pytest tests/test_layers345_upgrade.py -q      # 11 checks, a few seconds
python -m experiments.layers345_upgrade.run             # full run, about 8-10 minutes
python -m experiments.layers345_upgrade.run --reuse     # re-score saved forecasts, about 1 minute
```

The run needs `outputs/tables/prices.csv`, `outputs/models/analytical_run.json`,
`outputs/tables/risk_forecasts_oos.csv`, and the saved forecasts in
`experiments/improved_risk/results/`. All of these are in the repo, so no internet is needed.

## What was added

| Layer | Addition | Where |
|---|---|---|
| 3, 4 | GJR-GARCH(1,1) volatility with Student-t likelihood, refit monthly on 1,000 days | `src/quantedge/garch.py` |
| 3, 4 | Conditional EVT (GARCH filter + GPD on shocks), FHS, GARCH-t, forecast combination | `experiments/layers345_upgrade/models.py` |
| 3 | One quantile-function engine giving VaR, ES and PIT for every model | `src/quantedge/tail_grid.py` |
| 4 | EQD automatic threshold choice, extremal index, bootstrap intervals for VaR/ES | `src/quantedge/evt_extra.py` |
| 3, 5 | ES backtests: Du-Escanciano (unconditional + clustering), multinomial, McNeil-Frey | `src/quantedge/es_backtests.py` |
| 5 | Dynamic Quantile test, duration test, Monte Carlo p-values | `src/quantedge/backtests_extra.py` |
| 5 | FZ0 joint VaR-ES score, Diebold-Mariano test, Model Confidence Set | `src/quantedge/scoring.py` |

## Protocol

* Same allocation, dates and data as the baseline OOS run. The baseline forecasts are rebuilt
  and checked against the saved file before anything is compared.
* Every setting is fixed in `config.yaml` and taken from the cited papers. Nothing is tuned on
  2018-2026, and no model is chosen by its p-value.
* Each forecast for day t uses returns up to t-1 only. GARCH and EVT refit every 21 days on
  the previous 1,000 days. Volatility updates daily between refits.
* Daniel's saved candidates (`filtered_historical_alert`, `filtered_bootstrap_alert`,
  `filtered_historical_reference`, `filtered_evt_cusum`) are scored on the same table.

## Results

See [REVIEW.md](REVIEW.md) for the interpretation and [results/RESULTS.md](results/RESULTS.md)
for every number. Output files:

* `results/backtests_all.csv`: every test for every model and confidence
* `results/multinomial_es_backtest.csv`, `results/model_confidence_set.csv`, `results/diebold_mariano.csv`
* `results/extremal_index.csv`, `results/eqd_threshold_choices.csv`, `results/garch_fits.csv`
* `results/current_allocation_next_day.csv`: tomorrow's VaR/ES with 90% intervals
* `results/yearly_breaches_99.csv`, `results/garch_family_forecasts.csv.gz`

## References

* Glosten, Jagannathan and Runkle (1993), *Journal of Finance* 48(5): GJR-GARCH.
* McNeil and Frey (2000), *Journal of Empirical Finance* 7: conditional EVT.
* Barone-Adesi, Giannopoulos and Vosper (1999), *Journal of Futures Markets*: filtered historical simulation.
* Murphy, Tawn and Varty (2025), automated threshold selection (arXiv:2310.17999).
* Ferro and Segers (2003), *JRSS B* 65: extremal index (intervals estimator).
* Du and Escanciano (2017), *Management Science* 63(4): ES backtests via cumulative violations.
* Kratz, Lok and McNeil (2018), *Journal of Banking and Finance* 88: multinomial VaR backtests.
* Engle and Manganelli (2004), *JBES* 22(4): CAViaR and the Dynamic Quantile test.
* Christoffersen and Pelletier (2004), *Journal of Financial Econometrics* 2(1): duration test.
* Fissler and Ziegel (2016), *Annals of Statistics*; Patton, Ziegel and Chen (2019), *Journal of Econometrics*: FZ0 loss.
* Hansen, Lunde and Nason (2011), *Econometrica* 79(2): Model Confidence Set.
* Taylor (2020), *International Journal of Forecasting* 36(2): VaR/ES forecast combination.
