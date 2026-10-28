# Filtered historical and bootstrap: next-day alert experiment

This separate experiment reuses the **same CUSUM detector, threshold calibration, fast/slow EWMA settings, and five-day alert duration** as the EVT response experiment. It changes only how the two existing filtered empirical models use volatility on alert days. The forecast for a day never uses that day's return; a close-of-day trigger can first affect the following trading day. All three confidence levels are evaluated on the same 2,212 dates from 2018-01-02 through 2026-10-20, with the frozen 2007-2017 allocation.

The rule is specified in [`filtered_response.yaml`](../../filtered_response.yaml) and references the existing [`evt_response.yaml`](../../evt_response.yaml). The CUSUM threshold of **2.0** was calibrated on the 2007-2017 training period, where it triggered 26 times over 2,708 valid volatility days. While an alert is active, the response uses the larger of the usual EWMA volatility (lambda 0.94) and a faster EWMA estimate (lambda 0.80). It applies that scale to the **same standardized historical shocks** used by the original models; bootstrap draws use the same date-seeded random stream. On alert dates the script independently recomputes the reference tails and checks that they match the saved forecasts before calculating the changed VaR and ES.

## Results

| Model | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Average VaR |
|---|---:|---:|---:|---:|---:|---:|
| Filtered historical, original | 29 | 0.161 | 0.0057 | 0.0082 | 0.00025140 | $185,947 |
| Filtered historical, alert | **22** | **0.980** | **0.506** | **0.801** | **0.00023629** | **$190,533** |
| Filtered bootstrap, original | 29 | 0.161 | 0.0057 | 0.0082 | 0.00025304 | $189,795 |
| Filtered bootstrap, alert | **22** | **0.980** | **0.506** | **0.801** | **0.00023801** | **$194,465** |

At 5% significance, neither alert version is rejected by the three **99% VaR** tests on this development period. Both original and alert versions are also not rejected by those tests at **90% and 95%**; the detailed p-values and scores are in [`comparison.md`](comparison.md). At 99%, quantile loss improves by about 6% for each alert variant, while mean VaR rises by roughly $4,600 across all days.

The detector was active on **124 of 2,212 forecast days**. It increased the historical model's 99% VaR by an average of **$81,815 on active days** and bootstrap VaR by **$83,314**. Each alert variant avoided seven reference breaches, introduced no new breaches, and avoided the **same seven dates**: February 5 and 8, 2018; February 25 and 27 and March 12, 2020; July 24, 2024; and April 4, 2025. Because the two empirical models produce very similar forecasts, these matching outcomes are **not two independent confirmations**.

The detector fired after the March 9, 2020 close. On **March 10**, filtered historical VaR rose from about **$448k to $652k**. On **March 12**, the alert forecast was about **$726k**, above the roughly **$665k** realized loss; the original $506k forecast breached. It could not prevent the **March 9** breach because that day's return was not available when its forecast was made. The [dated forecasts](daily_forecasts.csv.gz), [detector timeline](detector_timeline.csv), [yearly counts](yearly.csv), and [alert impact](alert_impact.csv) show the full paths and tradeoff.

## Interpretation

This is promising **development evidence**, not a final validation. The 2018-2026 history was already inspected while designing the alert idea, even though its threshold itself used training data only. A high p-value on these dates cannot establish that the rule will perform as well on new crises. The alert also raises risk forecasts on some days that are not followed by a breach; average VaR and quantile loss must be assessed alongside exception counts. The reported Kupiec and Christoffersen tests assess **VaR**, not ES, although ES was recomputed consistently with the changed empirical tail.

Keep the original models and this experiment side by side. A confirmation claim needs a frozen rule and later observations that were not used to design it.
