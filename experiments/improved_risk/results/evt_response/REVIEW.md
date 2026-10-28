# EVT next-day response: interpretation

This separate experiment reuses the frozen 2007-2017 allocation and the 2,212 evaluation dates from 2018-01-02 through 2026-10-20. The evaluation period had already been inspected when these ideas were proposed. The five paths were fixed in `../../evt_response.yaml` before this side-by-side run. They are development comparisons, not an independent validation or a production rule.

## What each path changes

| Path | Tail fit | Daily volatility scale | Response to prior-close CUSUM alert |
|---|---|---|---|
| `raw_evt_reference` | Raw portfolio losses, every 21 trading days | No | No |
| `raw_evt_daily_scale` | Same raw fit | Yes, slow EWMA relative to its fit date | No |
| `raw_evt_daily_scale_cusum` | Same raw fit | Yes | Use max(slow EWMA, fast EWMA) while alert is active |
| `filtered_evt_reference` | Standardized losses, every 21 trading days | Yes, slow EWMA | No |
| `filtered_evt_cusum` | Same standardized fit | Yes | Use max(slow EWMA, fast EWMA) while alert is active |

The CUSUM accumulates `-return / prior_EWMA_volatility - 1`, floored at zero and reset on a trigger. The threshold grid and target maximum of three training triggers per year produced a threshold of **2.0** from 2007-2017 data alone. Slow EWMA has lambda 0.94; fast EWMA has lambda 0.80. A close-of-day trigger can affect only the **next five forecast days**; another trigger can extend that period. This active rule is evaluated sequentially. All variants retain the original GPD refit dates and selected thresholds.

Daily scaling of a GPD fitted to *raw* returns is a diagnostic heuristic: the fitted sample mixes different volatility conditions. Filtered EVT, whose GPD is fitted to standardized shocks, has a more direct interpretation. Neither approach makes an alert an independent prediction of a future loss.

## What happened

| Path | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Mean VaR |
|---|---:|---:|---:|---:|---:|---:|
| Raw EVT | 31 | 0.0736 | 0.000043 | 0.000047 | 0.0003186 | $178,778 |
| Raw EVT with daily scale | 26 | 0.4198 | 0.000155 | 0.000565 | 0.0002864 | $182,136 |
| Raw EVT with daily scale and alert | 22 | 0.9795 | 0.0181 | 0.0612 | 0.0002630 | $186,130 |
| Filtered EVT | 31 | 0.0736 | 0.00859 | 0.00638 | 0.0002499 | $185,185 |
| Filtered EVT with alert | 25 | 0.5466 | 0.4495 | 0.6265 | 0.0002331 | $189,490 |

All 99% figures are on the same dates. Filtered EVT with the alert is the only path whose three reported tests do not reject at 5% **on this reviewed development period**. The raw daily-scale-plus-alert path still fails independence, even though its combined-test p-value is just above 0.05. These p-values do not establish a robust model, especially after inspecting this history and considering several candidate mechanisms. ES is calculated but has no independent ES backtest here; 90% and 95% coverage are not part of this focused experiment.

The detector triggered after **29 evaluation-period closes** and was active on **124 of 2,212 forecast days**. Nine of 29 full five-day post-trigger windows contained a breach of the original filtered EVT forecast. The other 20 alerts are not automatically mistakes; a detector flags unusual conditions rather than promising a subsequent VaR breach. On active days the overlay increased VaR by an average of about **$76,791** for filtered EVT, and it removed six of its 31 breaches without adding new breaches. It reduced mean quantile loss despite a slightly higher mean VaR. The raw daily-scale overlay removed four breaches relative to raw daily scaling alone.

The alert first fired after the **February 24, 2020** close, so February 25 was its first affected forecast. It fired again after the **March 9** close, so March 10 was affected. The overlay did **not** prevent the March 9 loss, and filtered EVT still breached on March 12: the $665k realized loss exceeded even the raised $648k forecast. This is the timing limitation we wanted to test. See [`feb_mar_2020.png`](feb_mar_2020.png) and the dated [`detector_timeline.csv`](detector_timeline.csv).

The saved [`comparison.md`](comparison.md), [`backtests.csv`](backtests.csv), [`daily_forecasts.csv.gz`](daily_forecasts.csv.gz), [`alert_review.csv`](alert_review.csv), and [`alert_impact.csv`](alert_impact.csv) give the complete numerical evidence. The alert review compares the *original* filtered EVT breaches in the five days after each trigger; it is a descriptive measure, not a calibrated false-alarm probability. The run manifest records settings and input hashes.

## Decision for now

Keep this experiment separate. It demonstrates a causal next-day response and a plausible reduction in clustered 99% breaches, with an explicit increase in VaR on alert days. Do not promote it on the strength of this already-inspected sample. Next evaluation should include 90% and 95% coverage, an ES-specific check, alert costs or operational burden, and genuinely uninspected future observations before any allocation or production decision.
