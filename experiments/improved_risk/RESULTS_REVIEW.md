# Development results review — 27 October 2026

The fixed experiment evaluates 11 models/variants at three confidence levels on the same 2,212 dates, 2018-01-02 through 2026-10-20. Four models are unchanged baselines, four are new candidates and three are explicitly labeled sensitivity checks. Approximately 22 exceptions are expected at 99% confidence. The inspected historical period is development evidence; this is not an independent final holdout.

## Main 99% comparison

| Model | Exceptions | Kupiec p-value | Independence p-value | Interpretation at 5% significance |
|---|---:|---:|---:|---|
| Baseline historical | 36 | 0.00654 | 0.000207 | Coverage and independence reject |
| Filtered historical | 27 | 0.313 | 0.00362 | Coverage does not reject; independence rejects |
| Baseline bootstrap | 32 | 0.0478 | 0.000909 | Coverage and independence reject |
| Filtered bootstrap | 26 | 0.420 | 0.00285 | Coverage does not reject; independence rejects |
| Baseline EWMA-normal | 49 | 0.000000738 | 0.119 | Coverage rejects; independence does not reject |
| EWMA-Student-t | 37 | 0.00373 | 0.0249 | Coverage and independence reject |
| Baseline raw EVT | 34 | 0.0186 | 0.000114 | Coverage and independence reject |
| Filtered EVT | 26 | 0.420 | 0.00285 | Coverage does not reject; independence rejects |

The volatility-filtered empirical/EVT candidates improve the overall exception frequency and do not reject Kupiec coverage at 99%. Their remaining exceptions cluster, so independence and joint conditional coverage still reject at 99%. It would be misleading to describe them as passing all backtests. At 90% and 95%, none of the four candidates rejects the coverage, independence or joint conditional coverage tests in this run.

Student-t reduces 99% exceptions from 49 to 37 relative to EWMA-normal but is not sufficient to obtain acceptable 99% coverage. Fitted degrees of freedom range approximately 6.15–12.92, with no boundary warnings. This supports investigating whether symmetric Student-t innovations plus the existing EWMA filter adequately describe downside shocks; it does not justify searching df values against the known exception count.

All four candidates improve the 99% mean quantile forecasting score relative to their corresponding baseline in this development comparison. The score is separate from coverage: it penalizes both excessive conservatism and underestimated losses. It does not validate ES or remove the independence failures.

The raw historical 500- and 1,000-observation variants each have 33 exceptions at 99%, still rejecting Kupiec coverage. The inverse-CDF quantile variant has 32 and also rejects. These checks do not support claiming that a simple window/convention change resolves the baseline problem.

## Fit warnings and remaining uncertainty

There are 106 Student-t refits and 106 filtered-EVT refits. Six filtered-EVT selections use the declared warning fallback because adjacent thresholds do not meet the stability rule. Keep these warnings visible; favorable aggregate coverage does not establish tail-fit reliability.

The next scientific questions are whether a different volatility filter (such as a predeclared GARCH candidate), asymmetric innovations, or more explicit treatment of abrupt volatility transitions improves conditional behavior. They require a separate experiment, not retroactive tuning of this run. ES-specific validation and model/threshold uncertainty intervals are also outstanding extensions.

No candidate is automatically promoted into the baseline pipeline. Preserve the baseline comparison, freeze any subsequent model-selection rules, and reserve new observations for final evaluation. Machine-readable results and the input/configuration/implementation hashes are in `results/`.
