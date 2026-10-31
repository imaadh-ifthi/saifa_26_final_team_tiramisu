# Risk-model comparison and proposed competition choice

## Decision and scope

**Proposed lead model: `gjr_evt` (GJR-GARCH volatility filter with an EVT tail).** Use it for the selected VaR/ES forecast in the competition analysis, while showing historical simulation, EWMA-normal parametric VaR, and bootstrap Monte Carlo as the three required method comparisons. Show raw EVT as a tail-risk reference and `gjrx_vix_evt_d1000` as a challenger. This is a recommendation for the team's review, **not** a change to the main pipeline or the earlier frozen VIX-model choice.

This decision is based on the common 2019-01-02 to 2026-10-20 retrospective comparison: 1,961 forecast days, with portfolio weights fitted using 2007-01-04 to 2018-12-31 data and held fixed during evaluation. All forecasts use information available before their forecast day. The team had already inspected much of 2019-2026 while developing these models. Chronological forecasting prevents same-day look-ahead in the calculations, but **does not remove model-selection bias**. These results cannot be described as a fresh independent holdout.

The competition brief requires comparison of historical, parametric, and Monte Carlo VaR at 95% and 99%; ES and justified EVT; out-of-sample Kupiec, Christoffersen independence, and a Basel traffic light; and a portfolio decision supported by stress and position-risk analysis. It does **not** require every candidate to pass every statistical test. The chosen model should be explained alongside failures and limitations.

## Comparison on the same portfolio and dates

All counts below are breaches out of 1,961 forecasts. `Joint` is Christoffersen conditional coverage; `MF ES` is the McNeil-Frey ES test. A p-value below 0.05 rejects the corresponding null. The FZ0 score compares joint VaR/ES forecast quality on the **same dates**; lower is better. The final column reports the mean 99% VaR, not a current next-day forecast.

| Model | 95% breaches | 95% joint p | 95% MF ES p | 99% breaches | 99% Kupiec p | 99% independence p | 99% joint p | 99% MF ES p | 99% FZ0 | Mean 99% VaR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GJR + EVT (`gjr_evt`)** | 108 | 0.223 | 0.376 | 22 | 0.595 | 0.246 | 0.443 | 0.152 | -3.667 | $189,983 |
| VIX + GJR + EVT | 110 | 0.158 | 0.297 | 20 | 0.930 | 0.199 | 0.436 | **0.0446** | **-3.766** | $184,326 |
| Range + GJR + EVT | 110 | 0.476 | 0.373 | 22 | 0.595 | 0.246 | 0.443 | **0.0390** | -3.740 | $185,855 |
| Filtered historical + alert | 99 | 0.597 | 0.884 | 21 | 0.755 | 0.500 | 0.759 | 0.688 | -3.701 | $219,916 |
| Filtered EVT + CUSUM alert | — | — | — | 18 | 0.711 | 0.563 | 0.790 | 0.768 | -3.700 | $216,314 |
| Historical simulation | 108 | 0.541 | 0.306 | 34 | **0.00310** | **0.0213** | **0.000891** | **0.00760** | -3.334 | $206,066 |
| EWMA-normal parametric | 100 | 0.979 | **0.000200** | 41 | **0.0000229** | 0.280 | **0.0000711** | **0.000200** | -3.477 | $162,501 |
| Bootstrap Monte Carlo | 101 | 0.698 | 0.113 | 29 | **0.0466** | **0.00812** | **0.00416** | **0.000400** | -3.353 | $214,845 |
| Raw EVT (`evt_gpd`) | 109 | 0.390 | 0.841 | 22 | 0.595 | **0.00140** | **0.00527** | 0.163 | -3.101 | $210,246 |

The other evaluated candidates remain in [COMPARISON.md](COMPARISON.md) and [backtests.csv](backtests.csv). Plain GJR-Student-t and Markov-switching GJR-Student-t both fail 99% coverage in this run; filtered historical/bootstrap/EVT without an alert show clustering or Dynamic Quantile failures at 99%. Their presence is useful for diagnosing the mechanism, not a reason to promote them.

## Why `gjr_evt` leads

1. **It holds up at both confidence levels the brief requires.** At 95% and 99%, Kupiec, independence, joint coverage, Dynamic Quantile, duration, McNeil-Frey ES, and both available Du-Escanciano ES tests do not reject at 5%. This is a broader result than matching the expected 99% breach count. The 99% count of 22 is close to the approximately 19.6 breaches expected over 1,961 days.
2. **Its mechanism addresses an observed weakness.** A GJR volatility filter responds to recent shocks and can react differently after negative returns; EVT fits the tail of the filtered shocks. Raw EVT has a reasonable 99% total breach count here, but its independence and joint tests reject, consistent with a tail estimate that does not respond sufficiently to clustered volatility.
3. **Its ES evidence is stronger than the top-scoring challengers'.** The VIX model has a better FZ0 score and two fewer 99% breaches, but its 99% McNeil-Frey ES test rejects. The range model has the same ES rejection and a 95% Dynamic Quantile rejection (p = 0.032). We should not claim the GJR model has the best forecast score; its advantage is the balance of validation evidence and a simpler input set.
4. **It needs fewer external inputs and is easier to reproduce.** Unlike the VIX and range variants, it does not depend on an extra VIX or OHLC series and their alignment/revision rules. That is a practical advantage for a judge-run repository, though it does not establish statistical superiority.

This is a **judgment across several criteria**, not a rule that picks whichever model has the largest p-value or fewest breaches. Non-rejection does not prove a forecast is correct, and multiple tests on the same days are correlated. The 95% Dynamic Quantile p-value for `gjr_evt` is 0.057, close to the 0.05 cutoff. At the optional 90% level its duration test rejects (p = 0.00952); report this if presenting 90% results. Its average 99% VaR is about $190,000, below the alert models' roughly $216,000-$224,000; this does not mean it is automatically safer or better for every day.

## Roles for the other models

- **VIX + GJR + EVT:** retain as the strongest forecast-score challenger and explain its 99% ES failure. The earlier crisis-response experiment selected this model under a different, predeclared development rule. This new retrospective comparison does not erase that earlier selection. The team should explicitly decide which protocol governs the final submission and keep results from the different portfolio weights and test periods separate.
- **Historical, parametric, Monte Carlo:** include all three at 95% and 99%, as required. The plain 99% versions fail at least one core backtest here. Their disagreement is evidence about empirical tails, the normality assumption, and changing volatility; hiding failures would make the report less defensible. The Monte Carlo implementation is an empirical bootstrap, so its similarity to historical simulation is expected.
- **Raw EVT:** retain for threshold diagnostics and tail-risk explanation. Its 99% breach *total* alone looks acceptable, but the clustering tests reject.
- **CUSUM alerts:** use only as an exploratory next-day risk indicator. Their 99% results are attractive, but the alert design was influenced by earlier inspection of overlapping dates. The filtered EVT alert was evaluated only at 99% in this comparison; it cannot be claimed to pass the required 95% checks.
- **Traffic lights:** every model's most recent 250-day 99% light is green. Over all overlapping rolling windows, `gjr_evt` has 1,405 green and 307 yellow windows, versus 1,585 green and 127 yellow for the VIX model. The windows overlap heavily, and one green label does not override full-period test failures.

## Before presenting this as the competition recommendation

1. Freeze the team's final portfolio, risk-model specification, input snapshots, confidence levels, and selection rule. The allocation in this comparison (about 39.35% SPY, 30.65% AGG, 15% GLD, 15% XLK) differs from the current allocation used in the earlier investment memo and decision exports. **Do not combine forecast-test figures from this run with stress losses or risk contributions for that other allocation.**
2. If `gjr_evt` is adopted, regenerate the current-position VaR/ES, stress results, risk contributions, and investment memo for one consistent portfolio and date. Preserve the earlier VIX decision as a documented prior experiment rather than silently replacing it.
3. Disclose the already-inspected test period and the 90% duration failure. Freeze forecasts before new outcomes for prospective confirmation; no historical reshuffling of 2007-2026 can turn these inspected outcomes into a fresh final exam.

Evidence and reproducibility: [COMPARISON.md](COMPARISON.md), [backtests.csv](backtests.csv), [rolling_traffic_lights.csv](rolling_traffic_lights.csv), [manifest.json](manifest.json), and `python -m experiments.compare_2007_2018` from the repository root. The source snapshots and hashes are recorded in the manifest; the extra VIX and OHLC raw inputs must be restored to reproduce those challenger forecasts.
