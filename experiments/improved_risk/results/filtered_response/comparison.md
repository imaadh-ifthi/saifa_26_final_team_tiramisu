# Filtered empirical next-day response

The detector rule and fast/slow EWMA settings are exactly those declared in the separate EVT response protocol. Its threshold is calibrated using 2007-2017 returns only. Both filtered models and all three confidences use the same 2,212 evaluation dates. This 2018-2026 period was already inspected before the idea was proposed; these are development results.
The training-calibrated threshold is 2 (26 training triggers in 2708 valid volatility days). A close-of-day alert can first change the next day's forecast. The candidate uses max(slow, fast) EWMA only while the alert is active; historical standardized shocks and bootstrap draws are otherwise unchanged.

## 90% VaR

| Model | Breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Mean VaR USD |
|---|---:|---:|---:|---:|---:|---:|
| filtered_bootstrap_alert | 231 | 0.49014 | 0.84471 | 0.77314 | 0.00123204 | 76,588 |
| filtered_bootstrap_reference | 237 | 0.26773 | 0.89501 | 0.5364 | 0.001247555 | 74,817 |
| filtered_historical_alert | 232 | 0.44725 | 0.88227 | 0.74098 | 0.001233802 | 76,277 |
| filtered_historical_reference | 236 | 0.29889 | 0.85728 | 0.57364 | 0.001248868 | 74,520 |

## 95% VaR

| Model | Breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Mean VaR USD |
|---|---:|---:|---:|---:|---:|---:|
| filtered_bootstrap_alert | 109 | 0.87568 | 0.86396 | 0.97344 | 0.0007785483 | 106,821 |
| filtered_bootstrap_reference | 113 | 0.8155 | 0.92192 | 0.96848 | 0.0007903748 | 104,339 |
| filtered_historical_alert | 111 | 0.96889 | 0.79526 | 0.96616 | 0.0007781114 | 106,133 |
| filtered_historical_reference | 115 | 0.66967 | 0.99362 | 0.91299 | 0.0007907728 | 103,671 |

## 99% VaR

| Model | Breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Mean VaR USD |
|---|---:|---:|---:|---:|---:|---:|
| filtered_bootstrap_alert | 22 | 0.97952 | 0.50605 | 0.80137 | 0.0002380065 | 194,465 |
| filtered_bootstrap_reference | 29 | 0.16057 | 0.0056767 | 0.008152 | 0.0002530431 | 189,795 |
| filtered_historical_alert | 22 | 0.97952 | 0.50605 | 0.80137 | 0.0002362867 | 190,533 |
| filtered_historical_reference | 29 | 0.16057 | 0.0056767 | 0.008152 | 0.0002513986 | 185,947 |

## Alert impact

| Model | Confidence | Active days | Days with higher VaR | Mean uplift on active days USD | Breaches avoided | New breaches |
|---|---:|---:|---:|---:|---:|---:|
| filtered_historical | 90% | 124 | 124 | 31,341 | 4 | 0 |
| filtered_historical | 95% | 124 | 124 | 43,917 | 4 | 0 |
| filtered_historical | 99% | 124 | 124 | 81,815 | 7 | 0 |
| filtered_bootstrap | 90% | 124 | 124 | 31,595 | 6 | 0 |
| filtered_bootstrap | 95% | 124 | 124 | 44,269 | 4 | 0 |
| filtered_bootstrap | 99% | 124 | 124 | 83,314 | 7 | 0 |

P-values below 0.05 reject the corresponding null. Non-rejection is not proof of correctness. A higher p-value after examining this same history does not independently validate the rule. VaR backtests do not validate ES. The alert raises forecasts on some days and may reduce sensitivity to future losses; use the quantile loss and average VaR columns alongside breach counts.
