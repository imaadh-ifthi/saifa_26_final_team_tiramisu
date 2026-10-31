# 2007–2018 fit, 2019–2026 retrospective comparison

**These dates have already been inspected by the team. This is a chronological robustness run, not a fresh independent holdout. No candidate is selected automatically by a passing p-value.**

Portfolio weights fitted on 2007-01-04–2018-12-31; 1961 common forecast days. CUSUM threshold 2, calibrated on pre-2019 returns only. All model forecasts use prior-day information.

The three simple references, filtered candidates, GJR models, VIX/range candidates, Markov model, and earlier CUSUM candidates are shown below. CUSUM was originally designed after inspecting much of this test period; its result is especially susceptible to selection bias.

## 99% VaR and ES

| Model | Breaches / days | Kupiec p | Independence p | Joint p | DQ p | MF ES p | Mean FZ0 | Mean VaR $ | Mean ES $ | Latest 250-day light |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gjrx_vix_evt_d1000 | 20/1961 | 0.93 | 0.199 | 0.436 | 0.475 | 0.0446 | -3.7665 | 184,326 | 228,370 | green |
| gjrx_range_evt_m1000 | 22/1961 | 0.595 | 0.246 | 0.443 | 0.425 | 0.039 | -3.7400 | 185,855 | 231,758 | green |
| filtered_historical_alert | 21/1961 | 0.755 | 0.5 | 0.759 | 0.585 | 0.688 | -3.7010 | 219,916 | 273,314 | green |
| filtered_evt_cusum | 18/1961 | 0.711 | 0.563 | 0.79 | 0.784 | 0.768 | -3.6996 | 216,314 | 283,173 | green |
| filtered_bootstrap_alert | 21/1961 | 0.755 | 0.5 | 0.759 | 0.57 | 0.568 | -3.6860 | 223,977 | 273,055 | green |
| filtered_historical | 25/1961 | 0.241 | 0.04 | 0.061 | 6.22e-05 | 0.775 | -3.6701 | 214,441 | 266,481 | green |
| gjr_evt | 22/1961 | 0.595 | 0.246 | 0.443 | 0.339 | 0.152 | -3.6674 | 189,983 | 249,006 | green |
| filtered_evt | 24/1961 | 0.336 | 0.0337 | 0.0659 | 3.93e-05 | 0.872 | -3.6652 | 211,222 | 276,502 | green |
| filtered_bootstrap | 25/1961 | 0.241 | 0.04 | 0.061 | 5.42e-05 | 0.654 | -3.6542 | 218,414 | 266,232 | green |
| gjr_t | 36/1961 | 0.000865 | 0.693 | 0.0036 | 0.00039 | 0.0624 | -3.6220 | 166,392 | 210,754 | green |
| ms_gjr_t_m1000 | 39/1961 | 0.000105 | 0.804 | 0.000525 | 7.27e-05 | 0.0448 | -3.6073 | 162,972 | 202,761 | green |
| parametric_ewma | 41/1961 | 2.29e-05 | 0.28 | 7.11e-05 | 8.96e-07 | 0.0002 | -3.4772 | 162,501 | 185,861 | green |
| monte_carlo_bootstrap | 29/1961 | 0.0466 | 0.00812 | 0.00416 | 1.83e-26 | 0.0004 | -3.3530 | 214,845 | 267,708 | green |
| historical | 34/1961 | 0.0031 | 0.0213 | 0.000891 | 1.49e-29 | 0.0076 | -3.3341 | 206,066 | 267,843 | green |
| evt_gpd | 22/1961 | 0.595 | 0.0014 | 0.00527 | 6.46e-36 | 0.163 | -3.1010 | 210,246 | 299,680 | green |

## Evidence for a model choice

* **Best balanced candidate in this run: `gjr_evt`.** At 95%, 97.5%, and 99%, Kupiec, independence, joint coverage, DQ, duration, McNeil–Frey ES, and both Du–Escanciano ES tests do not reject at 5%. Its 90% duration test does reject, so this is not a clean pass at every confidence.
* **Best 99% FZ0 score: `gjrx_vix_evt_d1000`.** It has 20 breaches and passes the 99% VaR tests, but McNeil–Frey ES rejects at 99% (p = 0.0446), and 90% joint coverage rejects (p = 0.00676). The range model also has a 99% McNeil–Frey rejection (p = 0.039).
* **CUSUM alert variants pass the listed 99% VaR tests**, but were originally developed after inspecting overlapping outcomes. Their 95% DQ results are weaker, and `filtered_evt_cusum` was only generated at 99% here. They should not be promoted on this retrospective result alone.
* Every model's latest 250-day traffic light is green. This single-window label cannot distinguish the candidates; see the full rolling file and tests.

### Leading candidates across confidence levels

| Model | Confidence | Breaches | Joint p | DQ p | Duration p | MF ES p | Mean FZ0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gjr_evt | 90.0% | 211 | 0.139 | 0.244 | 0.00952 | 0.275 | -4.4004 |
| gjr_evt | 95.0% | 108 | 0.223 | 0.0571 | 0.311 | 0.376 | -4.1686 |
| gjr_evt | 97.5% | 52 | 0.802 | 0.161 | 0.733 | 0.17 | -3.9511 |
| gjr_evt | 99.0% | 22 | 0.443 | 0.339 | 0.861 | 0.152 | -3.6674 |
| gjrx_vix_evt_d1000 | 90.0% | 201 | 0.00676 | 0.0857 | 0.00393 | 0.1 | -4.4157 |
| gjrx_vix_evt_d1000 | 95.0% | 110 | 0.158 | 0.209 | 0.203 | 0.297 | -4.1854 |
| gjrx_vix_evt_d1000 | 97.5% | 56 | 0.536 | 0.236 | 0.454 | 0.4 | -4.0078 |
| gjrx_vix_evt_d1000 | 99.0% | 20 | 0.436 | 0.475 | 0.525 | 0.0446 | -3.7665 |
| gjrx_range_evt_m1000 | 90.0% | 201 | 0.219 | 0.226 | 0.00559 | 0.0924 | -4.4143 |
| gjrx_range_evt_m1000 | 95.0% | 110 | 0.476 | 0.032 | 0.819 | 0.373 | -4.1898 |
| gjrx_range_evt_m1000 | 97.5% | 51 | 0.821 | 0.178 | 0.449 | 0.108 | -3.9943 |
| gjrx_range_evt_m1000 | 99.0% | 22 | 0.443 | 0.425 | 0.594 | 0.039 | -3.7400 |
| filtered_historical_alert | 90.0% | 190 | 0.597 | 0.0111 | 0.931 | 0.783 | -4.3782 |
| filtered_historical_alert | 95.0% | 99 | 0.597 | 0.028 | 0.291 | 0.884 | -4.1345 |
| filtered_historical_alert | 97.5% | 46 | 0.651 | 0.293 | 0.246 | 0.616 | -3.9130 |
| filtered_historical_alert | 99.0% | 21 | 0.759 | 0.585 | 0.274 | 0.688 | -3.7010 |
| filtered_evt_cusum | 99.0% | 18 | 0.79 | 0.784 | 0.166 | 0.768 | -3.6996 |

### Rolling 250-day traffic lights at 99%

These are overlapping windows, so counts are descriptive and are not independent tests.

| Model | Green windows | Yellow windows | Red windows |
|---|---:|---:|---:|
| evt_gpd | 1453 | 18 | 241 |
| filtered_bootstrap | 1093 | 619 | 0 |
| filtered_bootstrap_alert | 1447 | 265 | 0 |
| filtered_evt | 1291 | 421 | 0 |
| filtered_evt_cusum | 1520 | 192 | 0 |
| filtered_historical | 1197 | 515 | 0 |
| filtered_historical_alert | 1424 | 288 | 0 |
| gjr_evt | 1405 | 307 | 0 |
| gjr_t | 858 | 821 | 33 |
| gjrx_range_evt_m1000 | 1427 | 285 | 0 |
| gjrx_vix_evt_d1000 | 1585 | 127 | 0 |
| historical | 852 | 607 | 253 |
| monte_carlo_bootstrap | 876 | 820 | 16 |
| ms_gjr_t_m1000 | 724 | 898 | 90 |
| parametric_ewma | 670 | 873 | 169 |

## Reading the comparison

P-values below 0.05 reject their corresponding null. Kupiec tests the breach rate; Christoffersen independence tests whether breaches cluster; joint coverage combines them. The 250-day traffic light applies at 99% only. Quantile loss and FZ0 measure forecast quality beyond breach counts; lower scores are better on the same dates. MF tests ES on breach days. PIT-based Du–Escanciano ES tests are available only for models that produce a full quantile distribution.

The Monte Carlo backtest p-values in the CSV use 1,000 null simulations. They are a check on rare-event asymptotics, not a second independent data sample. The historical 2008 stress replay uses weights fitted through 2018, so it is a retrospective scenario rather than an out-of-sample portfolio outcome.

Reproduce this run from the repo root with `python -m experiments.compare_2007_2018` after restoring `data/raw/vixcls_fred.csv`, `vix_yahoo.csv`, and `ohlc.csv` using `python -m experiments.crisis_response.download_data`. Source file hashes and the fitted portfolio weights are in `manifest.json`. Yahoo may revise historic OHLC data.

`backtests.csv` contains 90%, 95%, 97.5%, and 99% results plus DQ, duration, Monte Carlo p-values, ES tests and scores. `rolling_traffic_lights.csv`, `yearly.csv`, `stress_summary.csv`, `risk_contributions.csv`, `diebold_mariano.csv`, and the daily forecasts allow a fuller review.
