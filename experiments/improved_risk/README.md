# Improved risk-model development experiment

This folder contains separate candidate models, configuration, tests and results. It does not replace `src/quantedge/` or write into the baseline `outputs/` folder. It reuses the baseline allocation, tail mathematics and backtesting utilities so comparisons use the same definitions.

Run from the repository root:

```powershell
python -m pytest experiments/improved_risk/tests -q
python -m experiments.improved_risk.run
```

The comparison is offline and requires `outputs/tables/prices.csv` and `data/raw/risk_free.csv`. If the ignored FRED snapshot is absent, obtain it through the existing analytical pipeline first; see the root README. Paths can be supplied with `--prices`, `--risk-free`, `--base-config` and `--experiment-config`. The default results directory is `experiments/improved_risk/results/`.

## Fixed development protocol

The reviewed 2018–2026 historical period has already influenced the motivation for these experiments. It is **not an untouched holdout**. `config.yaml` declares all candidates before the comparison; there is no grid search, fitted VaR multiplier, discarded crisis period, altered test significance, or automatic promotion of a winner. The manifest records the protocol, input and implementation SHA-256 hashes, configurations, allocation provenance and dependency versions.

Portfolio weights are estimated with the unchanged training-only allocation function, using the base configuration's 2010–2017 period and contemporaneously available risk-free quote. Each forecast is computed with information strictly before its realized return. All models/confidences must cover the complete identical evaluation period. Insufficient warmup or unavailable fits are errors rather than a reason to remove inconvenient dates.

## Models

| Candidate | Definition | Default estimation settings |
|---|---|---|
| `filtered_historical` | Standardize old returns using each date's own preceding volatility estimate, then rescale those shocks using today's forecast volatility | 500 residual observations |
| `filtered_bootstrap` | Resample those standardized/rescaled shocks before computing VaR and ES | 500 observations; 10,000 simulations; seeded draws shared across confidence levels |
| `ewma_student_t` | Zero-mean Student-t shocks, explicitly scaled to unit variance, multiplied by forecast volatility | Degrees of freedom estimated by maximum likelihood on 1,000 residuals every 21 observations; bounds 2.1–100 |
| `filtered_evt` | Fit a GPD to losses in standardized residuals; rescale threshold and scale by forecast volatility every day | 1,000 residuals; fit every 21 observations; same threshold diagnostics and selection heuristic as the baseline |

All candidates use the baseline's bias-corrected pandas EWMA standard-deviation estimator with decay 0.94, shifted one observation. This deliberately isolates the effect of filtering/tail choices from changing the volatility estimator. A 60-observation warmup is required. A numerical volatility floor of 1e-8 prevents division by zero; it is not a calibrated risk buffer. Standardized residuals are formed as `z_t = r_t / sigma_t`, where `sigma_t` uses only returns before t. Using today's volatility to standardize yesterday's shocks would be incorrect.

For Student-t df `nu`, the distribution's scale is `sqrt((nu-2)/nu)`, not 1: an ordinary Student-t has variance `nu/(nu-2)`. Holding variance fixed prevents accidentally increasing both volatility and tail heaviness. Degrees of freedom fit a zero-location, unit-variance model and are estimated from past data, never from future exception counts. A fit at a declared bound is flagged. Log VaR/ES have analytical formulas; simple-loss ES integrates the conditional tail quantile using deterministic 512-node Gauss–Legendre quadrature. Tests compare it with independent adaptive integration, including heavy tails near the lower df bound. It is not obtained by transforming mean log ES.

Filtered EVT retains monthly tail-shape refits but updates risk with forecast volatility **daily**. `estimation_end` records the volatility information cutoff; `tail_fit_end` records the earlier residual-shape fitting cutoff. Threshold diagnostics are explicitly in standardized-residual units. Adjacent-threshold stability and fallback warnings remain visible. The candidate uses EWMA filtering, not GARCH, and does not claim to reproduce a particular published GARCH-EVT implementation.

## Baselines and sensitivity checks

The runner recomputes the unchanged four baseline methods on the same allocation and dates. It also exports separately labeled raw historical window sensitivities (500 and 1,000 observations) and a 252-observation empirical inverse-CDF quantile variant. Empirical ES retains the exact fractional tail-mass convention for all quantile variants. These sensitivity checks are not automatically selected based on their p-values.

## Results

Read [RESULTS_REVIEW.md](RESULTS_REVIEW.md) for interpretation of the first fixed comparison, including improved coverage and remaining independence failures.

* `results/comparison.md`: readable 99% comparison, including coverage, independence, conditional coverage and mean quantile loss.
* `results/backtests.csv`: all confidences and legacy traffic lights.
* `results/scores.csv`: mean quantile forecasting loss (lower is better), average VaR/ES and breach severity.
* `results/yearly_exceptions.csv`: yearly exceptions and expected counts, so calm periods cannot hide crisis behavior.
* `results/forecasts.csv.gz`: all evaluated predictions and realized outcomes, compressed to reduce artifact size. Read with `pandas.read_csv` directly.
* `results/fit_diagnostics.csv`: Student-t df fits and standardized EVT threshold evidence.
* `results/baseline_evt_diagnostics.csv`: matching raw-EVT evidence.
* `results/risk_summary.csv`: latest and average risk across models.
* `results/manifest.json`: frozen settings and provenance for that run.

Passing software tests checks arithmetic, causality and export consistency. Non-rejection in a statistical test is different: it is not a guarantee of model adequacy. VaR tests do not validate ES. Tail asymmetry, jumps, filter misspecification, threshold/parameter uncertainty, and small tail samples remain limitations. A heavier-tailed distribution can worsen coverage at other confidences, and filtering can still miss abrupt volatility transitions.

No model is automatically moved into the baseline pipeline. Human review should consider all confidences, independence, quantile scores, crisis behavior, warnings and economic interpretation. Any model selected using this period needs a separately reserved evaluation, ideally future observations. GARCH-based filtering, ES-specific backtests and a formal rolling model-selection experiment remain extensions.

## References

* [Bank of England: filtered historical simulation and its competitors](https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors)
* [McNeil and Frey: conditional tail-risk estimation](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf)
* [SciPy Student-t distribution and scale convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
