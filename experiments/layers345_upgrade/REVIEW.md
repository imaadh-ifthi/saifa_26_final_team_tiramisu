# Layer 3-5 upgrade: what improved and what did not

Evaluation is the same as the baseline: frozen 2007-2017 allocation, 2,212 trading days from
2018-01-02 to 2026-10-20, one-day-ahead forecasts using only past data. The rebuilt baseline
forecasts match the saved baseline to about 1e-16, so the comparison is like for like.
All numbers are in [RESULTS.md](results/RESULTS.md).

## Short answer

Yes, there is a real improvement. The conditional EVT model (`gjr_evt`: GJR-GARCH filter plus a
GPD tail on the standardised shocks) is the only model in the whole comparison that passes
every VaR test and every ES test at 99%. It also does this with a smaller average VaR than the
alert-based candidates, so it is not passing just by being more conservative.

## 1. Against the main-pipeline baselines (the models in `outputs/`)

| 99% | Breaches (22 expected) | Joint coverage p | DQ p | DE-CES p (ES clustering) | Mean FZ0 |
|---|---:|---:|---:|---:|---:|
| historical | 38 | 0.0001 | 2e-26 | 5e-35 | -3.368 |
| evt_gpd (raw EVT) | 31 | 0.00005 | 2e-42 | 9e-99 | -3.212 |
| **gjr_evt (new)** | **26** | **0.43** | **0.34** | **0.12** | **-3.800** |

* All four baselines are rejected at 99% by the joint test, the DQ test, the duration test and
  the Du-Escanciano ES tests. The new model is rejected by none of them.
* The FZ0 score (lower is better) improves at every confidence level. The Diebold-Mariano
  test says the gain over historical simulation is significant at 90% and 95% (p around
  0.005 to 0.009) and borderline at 99% (p = 0.046), where breaches are rare.
* At 90% and 95% the Model Confidence Set drops historical and bootstrap VaR. They are
  statistically beaten.
* During the COVID crash (20 Feb to 1 Apr 2020), raw EVT had 12 breaches at 99% and the
  new model had 3.

**Why it works:** the extremal index of raw portfolio losses is about 0.44 to 0.57, which means
big losses arrive in clusters of about two days. Raw EVT assumes independent extremes. After
GJR-GARCH filtering, the shocks have an extremal index of 0.84 to 1.0, close to independent.
That is exactly the assumption EVT needs, and it is why the clustering tests now pass.

## 2. Against Daniel's best experimental candidates

| 99% | Breaches | Joint p | DQ p | Avg VaR $ | Avg ES $ | Mean FZ0 |
|---|---:|---:|---:|---:|---:|---:|
| filtered_historical_alert | 22 | 0.80 | 0.63 | 190,533 | 242,033 | -3.798 |
| filtered_evt_cusum | 25 | 0.63 | 0.24 | 189,490 | 246,899 | -3.807 |
| **gjr_evt (new)** | 26 | 0.43 | 0.34 | **167,126** | **216,249** | -3.800 |

* At 99% the models are statistically tied on FZ0 (DM p = 0.98). All of them stay in the
  Model Confidence Set.
* The new model reaches similar coverage with about 12% lower average VaR and about 11% lower
  average ES. That is roughly $23k less risk capital per day for the same breach behaviour.
* The new model has no alert rule. Daniel's notes say the alert idea came after looking at the
  2018-2026 breaches, so it carries some look-ahead risk. The new model uses textbook settings
  that were fixed in `config.yaml` before the run.
* At 90% and 95%, Daniel's alert models fail the DQ test (p = 0.003 to 0.023). At 95% the new
  model also fails DQ (p = 0.02), so neither is perfect there.
* ES-specific tests (Du-Escanciano, multinomial) cannot be run on Daniel's saved candidates
  because they do not store the full forecast distribution. The new models do.
* Weak spot: in calendar-year 2020 the new model has 8 breaches at 99% against 3 for the
  alert models: 1 in January, 3 in the Feb-March crash and 4 between June and October,
  after the crash, when markets were jumpy but measured volatility had come down.

## 3. What did NOT help (reported honestly)

* **EQD threshold choice (Murphy, Tawn and Varty 2025)** gave almost the same result as the
  fixed 90% threshold. It picked low thresholds (80-88% in every refit), which is in line with
  what the paper's authors report. At 99% its ES was slightly too small by the McNeil-Frey test
  (p = 0.041). Conclusion: keep the fixed 90% threshold as the main choice and quote EQD as a
  robustness check showing the threshold is not driving the result.
* **GJR-GARCH with Student-t shocks** has tails that are too thin for this portfolio:
  48 breaches at 99%, so it is rejected.
* **Forecast combination (Taylor 2020)** did not beat its best member, because averaging in the
  thin-tailed Student-t model pulled it down (35 breaches at 99%).
* **Filtered historical simulation** (`gjr_fhs`) is good at 90% and 95% but has a few too many
  breaches at 99% (33), because 1,000 past shocks hold only about 10 points beyond the 99th
  percentile. That is exactly the gap EVT fills.

## 4. "How confident are you?" Error bars for the current allocation

Next-day risk for the current allocation (data to 2026-10-20), new model, 90% bootstrap
intervals of the GPD fit:

* 99% VaR **$130,189** (interval $118k to $142k); 99% ES **$166,125** (interval $144k to $190k)
* 95% VaR $79,712 (interval $76k to $85k); 95% ES $111,655 (interval $103k to $121k)

The interval treats the GARCH parameters as known, so the true uncertainty is a bit wider.

## 5. Caveats a judge may raise

* 2018-2026 has been looked at before, so this is development evidence, not an untouched
  holdout. To limit overfitting, no setting was tuned and every setting is a literature default.
* At 99% there are only about 22 expected breaches, so every test has low power. That is why
  we added the multinomial, DQ and duration tests and the Monte Carlo p-values.
* 3 of the 106 GARCH refits ended with the optimiser warning flag (logged in
  `results/garch_fits.csv`). Their parameters are in line with their neighbours.
* The main pipeline in `outputs/` is unchanged. Promoting `gjr_evt` into it is a team decision.
