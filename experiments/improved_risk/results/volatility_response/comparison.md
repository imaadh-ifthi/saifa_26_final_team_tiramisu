# Fixed EWMA response probe

The 2018-2026 period is already inspected. This is a descriptive development comparison, not an independent selection or an automatically promoted model.
The declared comparison changes only EWMA lambda from 0.94 to 0.90 for filtered historical and filtered bootstrap. It recomputes each historical standardized shock using its own prior-information volatility estimate. The original 0.94 forecasts are reproduced from the saved data before comparing.

| Model | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Average VaR USD |
|---|---:|---:|---:|---:|---:|---:|
| filtered_bootstrap_faster_0.90 | 31 | 0.073577 | 0.0085879 | 0.0063839 | 0.0002467474 | 188,848 |
| filtered_bootstrap_reference_0.94 | 29 | 0.16057 | 0.0056767 | 0.008152 | 0.0002530431 | 189,795 |
| filtered_historical_faster_0.90 | 31 | 0.073577 | 0.0085879 | 0.0063839 | 0.0002451552 | 184,363 |
| filtered_historical_reference_0.94 | 29 | 0.16057 | 0.0056767 | 0.008152 | 0.0002513986 | 185,947 |

P-values below 0.05 reject the corresponding null. Non-rejection is not proof of adequacy. Quantile loss is lower-is-better; average VaR shows forecast conservatism. Review dated breaches and yearly counts as well as aggregate p-values.
