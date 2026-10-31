# Improved risk-model development experiment

This folder contains separate candidate models, configuration, tests and results. It does not replace `src/quantedge/` or write into the baseline `outputs/` folder. It reuses the baseline allocation, tail mathematics and backtesting utilities so comparisons use the same definitions.

Run from the repository root:

```powershell
python -m pytest experiments/improved_risk/tests -q
python -m experiments.improved_risk.run
python -m experiments.improved_risk.diagnose_breaches
python -m experiments.improved_risk.volatility_probe
python -m experiments.improved_risk.evt_response
python -m experiments.improved_risk.filtered_response
python -m experiments.improved_risk.plot_alert_breaches
```

The comparison is offline and requires `outputs/tables/prices.csv` and `data/raw/risk_free.csv`. If the ignored FRED snapshot is absent, obtain it through the existing analytical pipeline first; see the root README. Paths can be supplied with `--prices`, `--risk-free`, `--base-config` and `--experiment-config`. The default results directory is `experiments/improved_risk/results/`.

## Fixed development protocol

The reviewed 2018–2026 historical period has already influenced the motivation for these experiments. It is **not an untouched holdout**. `config.yaml` declares all candidates before the comparison; there is no grid search, fitted VaR multiplier, discarded crisis period, altered test significance, or automatic promotion of a winner. The manifest records the protocol, input and implementation SHA-256 hashes, configurations, allocation provenance and dependency versions.

Portfolio weights are estimated with the unchanged training-only allocation function, using the base configuration's 2007–2017 period and contemporaneously available risk-free quote. Training now includes the financial crisis; the comparison is rerun with the resulting allocation. Each forecast is computed with information strictly before its realized return. All models/confidences must cover the complete identical evaluation period. Insufficient warmup or unavailable fits are errors rather than a reason to remove inconvenient dates.

## Models

| Candidate | Definition | Default estimation settings |
|---|---|---|
| `filtered_historical` | Standardize old returns using each date's own preceding volatility estimate, then rescale those shocks using today's forecast volatility | 500 residual observations |
| `filtered_bootstrap` | Resample those standardized/rescaled shocks before computing VaR and ES | 500 observations; 10,000 simulations; seeded draws shared across confidence levels |
| `ewma_student_t` | Zero-mean Student-t shocks, explicitly scaled to unit variance, multiplied by forecast volatility | Degrees of freedom estimated by maximum likelihood on 1,000 residuals every 21 observations; bounds 2.1–100 |
| `filtered_evt` | Fit a GPD to losses in standardized residuals; rescale threshold and scale by forecast volatility every day | 1,000 residuals; fit every 21 observations; same threshold diagnostics and selection heuristic as the baseline |
| `gjr_filtered_historical` | Standardize returns using causal GJR-GARCH(1,1) conditional volatility, rescale historical residuals by the one-step forecast, and calculate empirical VaR/ES | 1,000 observations; refit every 21 observations with previous-fit warm start; Student-t innovations |
| `gjr_filtered_bootstrap` | Resample standardized GJR-GARCH residuals with replacement, rescale by the one-step forecast, and calculate empirical VaR/ES | 1,000 observations; refit every 21 observations; 10,000 simulations with seeded draws |

The EWMA candidates use the baseline's bias-corrected pandas EWMA standard-deviation estimator with decay 0.94, shifted one observation. They deliberately isolate filtering/tail choices while holding the volatility estimator fixed. A 60-observation warmup is required. A numerical volatility floor of 1e-8 prevents division by zero; it is not a calibrated risk buffer. Standardized residuals are formed as `z_t = r_t / sigma_t`, where `sigma_t` uses only returns before t. Using today's volatility to standardize yesterday's shocks would be incorrect. The two GJR candidates use `src/quantedge/garch.py` and a 1,000-observation rolling sample, refitted every 21 evaluation observations with the previous fit as a warm start. Their parameters and convergence state are recorded at each refit; forecasts use only returns through the prior date. GJR forecasts are calculated only when their names are declared under `candidates` in `config.yaml`.

For Student-t df `nu`, the distribution's scale is `sqrt((nu-2)/nu)`, not 1: an ordinary Student-t has variance `nu/(nu-2)`. Holding variance fixed prevents accidentally increasing both volatility and tail heaviness. Degrees of freedom fit a zero-location, unit-variance model and are estimated from past data, never from future exception counts. A fit at a declared bound is flagged. Log VaR/ES have analytical formulas; simple-loss ES integrates the conditional tail quantile using deterministic 512-node Gauss–Legendre quadrature. Tests compare it with independent adaptive integration, including heavy tails near the lower df bound. It is not obtained by transforming mean log ES.

Filtered EVT retains monthly tail-shape refits but updates risk with forecast volatility **daily**. `estimation_end` records the volatility information cutoff; `tail_fit_end` records the earlier residual-shape fitting cutoff. Threshold diagnostics are explicitly in standardized-residual units. Adjacent-threshold stability and fallback warnings remain visible. The candidate uses EWMA filtering, not GARCH, and does not claim to reproduce a particular published GARCH-EVT implementation.

## Baselines and sensitivity checks

The runner recomputes the unchanged four baseline methods on the same allocation and dates. It also exports separately labeled raw historical window sensitivities (500 and 1,000 observations) and a 252-observation empirical inverse-CDF quantile variant. Empirical ES retains the exact fractional tail-mass convention for all quantile variants. These sensitivity checks are not automatically selected based on their p-values.

## Results

Read [RESULTS_REVIEW.md](RESULTS_REVIEW.md) for interpretation of the current fixed comparison. Results are regenerated when the base training window changes; consult the accompanying manifest for the exact period and configuration.

* `results/comparison.md`: readable 99% comparison, including coverage, independence, conditional coverage and mean quantile loss.
* `results/backtests.csv`: all confidences and legacy traffic lights.
* `results/scores.csv`: mean quantile forecasting loss (lower is better), average VaR/ES and breach severity.
* `results/yearly_exceptions.csv`: yearly exceptions and expected counts, so calm periods cannot hide crisis behavior.
* `results/forecasts.csv.gz`: all evaluated predictions and realized outcomes, compressed to reduce artifact size. Read with `pandas.read_csv` directly.
* `results/fit_diagnostics.csv`: Student-t df fits, standardized EVT threshold evidence, and GJR-GARCH parameters and convergence status at each refit.
* `results/baseline_evt_diagnostics.csv`: matching raw-EVT evidence.
* `results/risk_summary.csv`: latest and average risk across models.
* `results/manifest.json`: frozen settings and provenance for that run.
* In Streamlit, the GJR historical and bootstrap forecasts are available only for the frozen validation portfolio; the method selector defaults to GJR filtered historical at 99% confidence when that basis is active.
* `results/diagnostics/diagnosis.md`: descriptive 99% breach analysis by year, short episodes and forecast-volatility regime. Its companion CSVs list each breach and realized asset loss contributions; `var_breach_timeline.png` plots the daily forecasts and losses. Run `diagnose_breaches` after regenerating forecasts.
* `results/MECHANISM_REVIEW.md`: dated raw-EVT refit chronology, calm-period filtered breaches and interpretation of the separately declared 0.90 versus 0.94 EWMA response probe. `results/volatility_response/` contains its daily forecasts, backtests and yearly counts. Run `volatility_probe` after regenerating the base experiment.
* `results/evt_response/REVIEW.md`: five-arm EVT daily-scale and next-day CUSUM experiment, with the crisis chart, complete forecast paths, backtests, alerts and provenance. `evt_response.yaml` declares its training-only calibration and response rule. Run `evt_response` after regenerating the base experiment.
* `results/filtered_response/REVIEW.md`: side-by-side filtered historical and bootstrap forecasts with and without the same prior-close alert, at 90%, 95%, and 99%. The companion CSVs contain all daily forecasts, backtests, yearly counts, alert impact, and provenance. Run `filtered_response` after regenerating the base experiment.
* `results/alert_breach_comparison/README.md`: an eight-panel chart of three distinct 99% alert candidates and their references, plus a repeated Monte Carlo candidate beside the original raw bootstrap; a March 2020 zoom and a counts/p-values summary are included. Run `plot_alert_breaches` after regenerating the EVT and filtered-response experiments.

Passing software tests checks arithmetic, causality and export consistency. Non-rejection in a statistical test is different: it is not a guarantee of model adequacy. VaR tests do not validate ES. Tail asymmetry, jumps, filter misspecification, threshold/parameter uncertainty, and small tail samples remain limitations. A heavier-tailed distribution can worsen coverage at other confidences, and filtering can still miss abrupt volatility transitions.

No model is automatically moved into the baseline pipeline. Human review should consider all confidences, independence, quantile scores, crisis behavior, warnings and economic interpretation. Any model selected using this period needs a separately reserved evaluation, ideally future observations. GARCH-based filtering, ES-specific backtests and a formal rolling model-selection experiment remain extensions.

## References

* [Bank of England: filtered historical simulation and its competitors](https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors)
* [McNeil and Frey: conditional tail-risk estimation](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf)
* [SciPy Student-t distribution and scale convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
