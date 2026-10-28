# EVT daily response experiment

Five predeclared EVT paths use the same frozen allocation and evaluation dates. Detector threshold was calibrated only on training returns. The 2018-2026 period is already inspected development evidence.
Training-only CUSUM threshold: **2**; 26 triggers in 2708 valid training days. The signal computed after close first changes the next day's forecast, and stays active for five forecast days unless another trigger extends it.

| Arm | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Average VaR USD |
|---|---:|---:|---:|---:|---:|---:|
| filtered_evt_cusum | 25 | 0.54658 | 0.44953 | 0.62647 | 0.0002331115 | 189,490 |
| filtered_evt_reference | 31 | 0.073577 | 0.0085879 | 0.0063839 | 0.0002498951 | 185,185 |
| raw_evt_daily_scale | 26 | 0.41982 | 0.00015542 | 0.00056531 | 0.0002863893 | 182,136 |
| raw_evt_daily_scale_cusum | 22 | 0.97952 | 0.018085 | 0.061159 | 0.0002630108 | 186,130 |
| raw_evt_reference | 31 | 0.073577 | 4.2802e-05 | 4.6683e-05 | 0.0003186057 | 178,778 |

## Signal and forecast impact

The detector triggered after 29 evaluation-period closes. It was active for 124 forecast days. Among 29 triggers with a full five-day follow-up, 9 were followed by a filtered-EVT reference breach. The others are not automatically false alarms: an alert can respond to an unusually large return even when later losses stay below VaR.

| Candidate overlay | Days with higher VaR | Mean uplift on active days | Reference breaches avoided | New breaches |
|---|---:|---:|---:|---:|
| raw_evt_daily_scale_cusum | 124 | $71,249 | 4 | 0 |
| filtered_evt_cusum | 124 | $76,791 | 6 | 0 |

The overlay uses max(slow EWMA, fast EWMA) only during an active alert. Raw daily scaling keeps the 21-day GPD fit but multiplies its threshold and scale by today's slow volatility divided by the volatility at fit time. Filtered EVT retains its standardized GPD fit. VaR/ES are recomputed in the original GPD units; there is no fitted VaR multiplier.

A higher p-value on this inspected period is not validation. Check signal timing, false alerts, risk forecast size, quantile loss, and crisis paths; ES is not validated by VaR tests.
