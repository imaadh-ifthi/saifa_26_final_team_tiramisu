# Backward-window robustness comparison

**The 2021–2026 arm looks ahead to future events when scoring 2013–2020. These are descriptive, retrospective results, not valid out-of-sample evidence. Both windows and the test years have already been inspected during model development.**

The portfolio weights were estimated once from 2007–2012 and held identical in both arms. Each arm fits its historical shocks and EVT tail once on its named training block. Only volatility and the alert state update from preceding test-day returns. This differs from the existing rolling-refit models; compare patterns, not their headline p-values.

| Training window | Model | 99% breaches / days | Kupiec p | Independence p | Joint p | Mean quantile loss | Mean VaR USD |
|---|---|---:|---:|---:|---:|---:|---:|
| early_2007_2012 | evt_gpd | 13/2015 | 0.0869 | 0.0711 | 0.0453 | 0.000327129 | 213,463 |
| early_2007_2012 | filtered_bootstrap | 39/2015 | 0.0002 | 0.0065 | 0.0000 | 0.000251019 | 140,486 |
| early_2007_2012 | filtered_bootstrap_alert | 34/2015 | 0.0048 | 0.1323 | 0.0060 | 0.00023214 | 144,581 |
| early_2007_2012 | filtered_evt | 33/2015 | 0.0084 | 0.0017 | 0.0002 | 0.00024605 | 149,032 |
| early_2007_2012 | filtered_evt_cusum | 30/2015 | 0.0398 | 0.0805 | 0.0262 | 0.000227768 | 153,367 |
| early_2007_2012 | filtered_historical | 34/2015 | 0.0048 | 0.0022 | 0.0002 | 0.000248441 | 144,554 |
| early_2007_2012 | filtered_historical_alert | 31/2015 | 0.0244 | 0.0918 | 0.0192 | 0.000229959 | 148,763 |
| early_2007_2012 | historical | 59/2015 | 0.0000 | 0.0015 | 0.0000 | 0.000373046 | 111,916 |
| early_2007_2012 | monte_carlo_bootstrap | 57/2015 | 0.0000 | 0.0010 | 0.0000 | 0.000369716 | 113,685 |
| early_2007_2012 | parametric_ewma | 49/2015 | 0.0000 | 0.0357 | 0.0000 | 0.000262306 | 129,294 |
| future_2021_2026 | evt_gpd | 24/2015 | 0.4027 | 0.0023 | 0.0067 | 0.000320725 | 175,550 |
| future_2021_2026 | filtered_bootstrap | 8/2015 | 0.0020 | 0.8006 | 0.0080 | 0.00025712 | 236,323 |
| future_2021_2026 | filtered_bootstrap_alert | 6/2015 | 0.0002 | 0.8498 | 0.0010 | 0.000260749 | 243,281 |
| future_2021_2026 | filtered_evt | 28/2015 | 0.0970 | 0.0004 | 0.0005 | 0.000240438 | 162,866 |
| future_2021_2026 | filtered_evt_cusum | 23/2015 | 0.5326 | 0.0266 | 0.0703 | 0.000225222 | 167,734 |
| future_2021_2026 | filtered_historical | 10/2015 | 0.0118 | 0.7521 | 0.0400 | 0.000251445 | 226,342 |
| future_2021_2026 | filtered_historical_alert | 7/2015 | 0.0007 | 0.8251 | 0.0030 | 0.000253439 | 233,020 |
| future_2021_2026 | historical | 25/2015 | 0.2953 | 0.0029 | 0.0069 | 0.000321782 | 171,058 |
| future_2021_2026 | monte_carlo_bootstrap | 25/2015 | 0.2953 | 0.0029 | 0.0069 | 0.000321167 | 173,571 |
| future_2021_2026 | parametric_ewma | 49/2015 | 0.0000 | 0.0357 | 0.0000 | 0.000262243 | 129,440 |

All 90%, 95%, and 99% results are in `backtests.csv` and `scores.csv`. P-values below 0.05 reject their null hypotheses. Non-rejection does not validate a model. The Basel traffic light is included in the CSV. VaR backtests do not validate ES.
