# Fixed-candidate development comparison

This historical period was already inspected. Results are development evidence, not an untouched holdout.
No candidate is automatically promoted and no parameter search is performed.

## 99% results

| Model | Category | Exceptions / observations | Kupiec p | Independence p | Conditional coverage p | Mean quantile loss |
|---|---|---:|---:|---:|---:|---:|
| evt_gpd | baseline | 31 / 2212 | 0.073577 | 4.2802e-05 | 4.6683e-05 | 0.000318606 |
| ewma_student_t | candidate | 39 / 2212 | 0.0011293 | 0.033908 | 0.00052559 | 0.000254805 |
| filtered_bootstrap | candidate | 29 / 2212 | 0.16057 | 0.0056767 | 0.008152 | 0.000253043 |
| filtered_evt | candidate | 31 / 2212 | 0.073577 | 0.0085879 | 0.0063839 | 0.000249895 |
| filtered_historical | candidate | 29 / 2212 | 0.16057 | 0.0056767 | 0.008152 | 0.000251399 |
| historical | baseline | 38 / 2212 | 0.002078 | 0.0036935 | 0.00012922 | 0.000315761 |
| historical_quantile_inverted_cdf | sensitivity | 32 / 2212 | 0.047803 | 0.00090864 | 0.0005751 | 0.000311162 |
| historical_window_1000 | sensitivity | 33 / 2212 | 0.030236 | 0.001173 | 0.00049346 | 0.000311093 |
| historical_window_500 | sensitivity | 36 / 2212 | 0.0065412 | 0.000207 | 2.5403e-05 | 0.000332482 |
| monte_carlo_bootstrap | baseline | 32 / 2212 | 0.047803 | 0.00090864 | 0.0005751 | 0.000309595 |
| parametric_ewma | baseline | 48 / 2212 | 1.6888e-06 | 0.10713 | 2.8791e-06 | 0.00026338 |

P-values below the configured significance level reject the relevant null. Non-rejection is not proof of correctness.
Quantile loss is a proper VaR forecasting score (lower is better); compare it only at the same confidence and dates.
ES is calculated but not independently validated by the VaR coverage tests.

All confidences, yearly exception counts, fit warnings and severity metrics are available in the adjacent CSVs.
See ../README.md for model definitions, fixed settings, remaining limitations and rerun commands.
