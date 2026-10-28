# Development results review: 2007-start training

This report supersedes the earlier 2010-start comparison after the requested change to training from 2007. The earlier results remain available in commit `303ba13`. Candidate settings, confidence levels and test significance are unchanged.

The frozen validation allocation is fitted on 2007-01-04 through 2017-12-29. All 11 models/variants are compared at three confidence levels on the same 2,212 evaluation dates, 2018-01-02 through 2026-10-20. This already inspected period is development evidence, not an untouched holdout.

## Main 99% comparison

| Model | Category | Exceptions | Kupiec p | Independence p | Conditional coverage p |
|---|---|---:|---:|---:|---:|
| evt_gpd | baseline | 31 | 0.073577 | 4.2802e-05 | 4.6683e-05 |
| ewma_student_t | candidate | 39 | 0.0011293 | 0.033908 | 0.00052559 |
| filtered_bootstrap | candidate | 29 | 0.16057 | 0.0056767 | 0.008152 |
| filtered_evt | candidate | 31 | 0.073577 | 0.0085879 | 0.0063839 |
| filtered_historical | candidate | 29 | 0.16057 | 0.0056767 | 0.008152 |
| historical | baseline | 38 | 0.002078 | 0.0036935 | 0.00012922 |
| monte_carlo_bootstrap | baseline | 32 | 0.047803 | 0.00090864 | 0.0005751 |
| parametric_ewma | baseline | 48 | 1.6888e-06 | 0.10713 | 2.8791e-06 |

Approximately 22 exceptions are expected at 99%. P-values below 0.05 reject the relevant null hypothesis. Non-rejection does not establish model adequacy. Overall exception frequency, clustering and joint coverage must be interpreted separately.

At 90%, candidates not rejecting Kupiec coverage: ewma_student_t, filtered_bootstrap, filtered_evt, filtered_historical. Candidates not rejecting joint conditional coverage: ewma_student_t, filtered_bootstrap, filtered_evt, filtered_historical.

At 95%, candidates not rejecting Kupiec coverage: ewma_student_t, filtered_bootstrap, filtered_evt, filtered_historical. Candidates not rejecting joint conditional coverage: ewma_student_t, filtered_bootstrap, filtered_evt, filtered_historical.

At 99%, among the initial four candidates above, filtered_bootstrap, filtered_evt and filtered_historical do not reject Kupiec coverage; none avoids rejection by joint conditional coverage. The later alert-response experiments are reported separately below.

## Breach timing diagnosis

The [99% breach diagnosis](results/diagnostics/diagnosis.md) uses the saved forecasts and frozen allocation; it changes no fitted model. It plots forecast VaR against each realized loss and exports dated breach events, close-together episodes, yearly counts, and volatility-regime counts.

The clearest concentrated episode is March 2020. Raw EVT records nine breaches from March 9 to April 1 under the stated five-trading-day-gap episode definition; historical and raw bootstrap each record five from March 9 to March 18. On March 12 the portfolio loss was about $665k, while raw EVT VaR was about $125k, historical VaR $194k, and bootstrap VaR $224k. Raw EVT VaR remained roughly $125k on March 9, 12, 16 and 18 because its tail fit updates periodically; that makes its immediate-crisis response a concrete issue to investigate. These four dates alone do not establish that fit cadence is the sole cause.

Filtered bootstrap records 11 of its 29 breaches in the lowest prior-forecast-volatility quartile and five in the highest; filtered historical has the same split, while filtered EVT has 13 and five. Several candidate breaches therefore also occur after relatively calm forecasts. A faster volatility response might help around a sudden jump, but it cannot by itself be assumed to cure all candidate failures. The asset columns in `breach_events.csv` attribute realized daily losses to the frozen weights; they do not attribute forecast-model error causally.

This period remains previously inspected development data. The episode and quartile definitions are descriptive and were not used to re-fit candidates or select a winner.

The [mechanism review and fixed volatility probe](results/MECHANISM_REVIEW.md) trace the raw EVT refits and test one predeclared faster EWMA filter on the two filtered empirical models. Lambda 0.90 marginally lowers mean quantile loss but raises their 99% breach counts from 29 to 31; both still reject 99% independence and combined coverage. It is not promoted as a fix.

The separate [EVT next-day response experiment](results/evt_response/REVIEW.md) compares existing raw/filtered EVT, raw EVT with a daily volatility scale, and both scale-aware paths with a training-calibrated prior-close CUSUM alert. Filtered EVT with the alert has 25 breaches and does not reject the three 99% tests on the already-inspected period; average VaR rises on alert days. This is development evidence, not independent validation or an automatic model replacement.

The separate [filtered empirical alert experiment](results/filtered_response/REVIEW.md) applies that exact detector rule to filtered historical and filtered bootstrap forecasts. Each has 22 rather than 29 exceptions at 99% and does not reject Kupiec, independence, or combined coverage on the already-inspected period; 90% and 95% outcomes are also reported. The two methods avoid the same seven dated breaches and are highly correlated, so their agreement is not independent corroboration.

## Fit warnings and interpretation

Filtered EVT uses the declared no-stability warning fallback in 3 of 106 refits. Student-t degrees of freedom range from 5.81 to 14.89. Inspect the detailed diagnostics rather than treating favorable aggregate coverage as proof of a reliable tail fit.

Yearly exceptions, quantile forecasting scores, breach severity and raw-window/quantile sensitivities are available in `results/`. VaR coverage tests do not independently validate ES. No model is automatically promoted.

Including 2008 in weight estimation does not guarantee lower losses. The current-allocation GFC replay improves from -21.34% to -21.03%, while the 2022 replay worsens from -10.85% to -12.18%. The GFC is now in training, so that replay is retrospective rather than an independent crisis test. See [the training-start comparison](../../docs/TRAINING_START_COMPARISON.md) for both allocations and the complete scenario comparison.

All 42 current software tests pass, including crisis-period inclusion, numerical checks of the optimized GPD survival integral and next-day alert timing. Statistical non-rejection is a separate result. The current evaluation period has already been inspected; an independent model-selection claim requires future, uninspected observations.
