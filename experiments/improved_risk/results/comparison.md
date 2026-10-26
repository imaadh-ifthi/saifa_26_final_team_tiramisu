# Fixed-candidate development comparison

This historical period was already inspected. Results are development evidence, not an untouched holdout.
No candidate is automatically promoted and no parameter search is performed.

## 99% results

| Model | Category | Exceptions / observations | Kupiec p | Independence p | Conditional coverage p | Mean quantile loss |
|---|---|---:|---:|---:|---:|---:|
| evt_gpd | baseline | 34 / 2212 | 0.018626 | 0.00011403 | 3.6703e-05 | 0.000282958 |
| ewma_student_t | candidate | 37 / 2212 | 0.0037325 | 0.024948 | 0.0012083 | 0.000228784 |
| filtered_bootstrap | candidate | 26 / 2212 | 0.41982 | 0.0028479 | 0.0084243 | 0.000230844 |
| filtered_evt | candidate | 26 / 2212 | 0.41982 | 0.0028479 | 0.0084243 | 0.000225644 |
| filtered_historical | candidate | 27 / 2212 | 0.31346 | 0.0036203 | 0.0087338 | 0.00022866 |
| historical | baseline | 36 / 2212 | 0.0065412 | 0.000207 | 2.5403e-05 | 0.000283921 |
| historical_quantile_inverted_cdf | sensitivity | 32 / 2212 | 0.047803 | 0.00090864 | 0.0005751 | 0.00028354 |
| historical_window_1000 | sensitivity | 33 / 2212 | 0.030236 | 0.001173 | 0.00049346 | 0.000277128 |
| historical_window_500 | sensitivity | 33 / 2212 | 0.030236 | 0.001173 | 0.00049346 | 0.000294037 |
| monte_carlo_bootstrap | baseline | 32 / 2212 | 0.047803 | 0.00090864 | 0.0005751 | 0.000285029 |
| parametric_ewma | baseline | 49 / 2212 | 7.3755e-07 | 0.11936 | 1.4127e-06 | 0.000236055 |

P-values below the configured significance level reject the relevant null. Non-rejection is not proof of correctness.
Quantile loss is a proper VaR forecasting score (lower is better); compare it only at the same confidence and dates.
ES is calculated but not independently validated by the VaR coverage tests.

All confidences, yearly exception counts, fit warnings and severity metrics are available in the adjacent CSVs.
See ../README.md for model definitions, fixed settings, remaining limitations and rerun commands.
