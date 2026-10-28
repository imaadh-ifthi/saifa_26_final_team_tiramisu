# 99% VaR breach diagnosis

This describes saved, already-inspected 2018-2026 forecasts. It fits no models and does not select parameters.
All amounts use the frozen training allocation. A breach is actual loss > same-day forecast VaR.
Adjacent means consecutive trading-day breaches. An episode joins breaches up to five trading days apart; this is descriptive, not a statistical test.

## Overview

| Model | Breaches | Adjacent pairs | Largest 5-day-gap episode | Mean VaR, USD |
|---|---:|---:|---:|---:|
| evt_gpd | 31/2212 | 5 | 9 | 178,778 |
| ewma_student_t | 39/2212 | 3 | 3 | 156,147 |
| filtered_bootstrap | 29/2212 | 3 | 3 | 189,795 |
| filtered_evt | 31/2212 | 3 | 3 | 185,185 |
| filtered_historical | 29/2212 | 3 | 3 | 185,947 |
| historical | 38/2212 | 4 | 5 | 180,747 |
| monte_carlo_bootstrap | 32/2212 | 4 | 5 | 189,635 |
| parametric_ewma | 48/2212 | 3 | 3 | 144,182 |

## Years with most breaches

- **evt_gpd:** 2020: 15/253 (expected 2.5), 2018: 7/251 (expected 2.5), 2022: 3/251 (expected 2.5)
- **ewma_student_t:** 2020: 10/253 (expected 2.5), 2018: 7/251 (expected 2.5), 2024: 7/252 (expected 2.5)
- **filtered_bootstrap:** 2024: 7/252 (expected 2.5), 2020: 6/253 (expected 2.5), 2018: 5/251 (expected 2.5)
- **filtered_evt:** 2020: 10/253 (expected 2.5), 2024: 6/252 (expected 2.5), 2018: 5/251 (expected 2.5)
- **filtered_historical:** 2024: 7/252 (expected 2.5), 2020: 6/253 (expected 2.5), 2018: 5/251 (expected 2.5)
- **historical:** 2022: 11/251 (expected 2.5), 2020: 8/253 (expected 2.5), 2024: 7/252 (expected 2.5)
- **monte_carlo_bootstrap:** 2020: 8/253 (expected 2.5), 2022: 8/251 (expected 2.5), 2024: 7/252 (expected 2.5)
- **parametric_ewma:** 2020: 11/253 (expected 2.5), 2018: 7/251 (expected 2.5), 2024: 7/252 (expected 2.5)

## Close-together breach episodes

Largest episodes across models (overlapping dates are the same market event):

| Model | Start | End | Breaches | Largest excess over VaR, USD |
|---|---|---|---:|---:|
| evt_gpd | 2020-03-09 | 2020-04-01 | 9 | 539,778 |
| historical | 2020-03-09 | 2020-03-18 | 5 | 470,711 |
| monte_carlo_bootstrap | 2020-03-09 | 2020-03-18 | 5 | 440,866 |
| monte_carlo_bootstrap | 2018-01-30 | 2018-02-08 | 4 | 116,919 |
| historical | 2022-04-22 | 2022-05-09 | 4 | 85,641 |
| monte_carlo_bootstrap | 2022-04-22 | 2022-05-09 | 4 | 81,153 |
| historical | 2025-04-03 | 2025-04-10 | 3 | 159,870 |
| monte_carlo_bootstrap | 2025-04-03 | 2025-04-10 | 3 | 152,823 |
| evt_gpd | 2025-04-03 | 2025-04-10 | 3 | 147,705 |
| historical | 2018-02-02 | 2018-02-08 | 3 | 118,677 |
| evt_gpd | 2020-02-24 | 2020-02-27 | 3 | 108,178 |
| historical | 2020-02-24 | 2020-02-27 | 3 | 103,838 |
| monte_carlo_bootstrap | 2020-02-24 | 2020-02-27 | 3 | 102,514 |
| evt_gpd | 2018-02-02 | 2018-02-08 | 3 | 85,336 |
| parametric_ewma | 2020-02-24 | 2020-02-27 | 3 | 85,331 |

## Forecast-volatility regimes

Quartiles are defined once from filtered-bootstrap's prior-information EWMA volatility over evaluation dates. They describe conditions, not an independent validation.

| Model | Calm Q1 | Q2 | Q3 | Turbulent Q4 |
|---|---:|---:|---:|---:|
| evt_gpd | 4/553 | 6/553 | 6/553 | 15/553 |
| ewma_student_t | 16/553 | 12/553 | 5/553 | 6/553 |
| filtered_bootstrap | 11/553 | 9/553 | 4/553 | 5/553 |
| filtered_evt | 13/553 | 8/553 | 5/553 | 5/553 |
| filtered_historical | 11/553 | 9/553 | 4/553 | 5/553 |
| historical | 7/553 | 7/553 | 9/553 | 15/553 |
| monte_carlo_bootstrap | 7/553 | 5/553 | 6/553 | 14/553 |
| parametric_ewma | 17/553 | 14/553 | 7/553 | 10/553 |

## How to read the files

`var_breach_timeline.png` plots daily loss, forecast VaR, and breach dates. `breach_events.csv` gives each breach's forecast, realized loss, excess, volatility regime, and realized asset loss contributions. `episodes.csv`, `yearly.csv`, and `volatility_regimes.csv` provide aggregations.
Realized asset contributions are weight times realized asset return; they do not identify the cause of model error. Large losses after a volatility jump suggest a hypothesis to investigate, not a proved mechanism.
These VaR diagnostics do not validate ES. The 2018-2026 sample is already inspected; no p-value here is an untouched holdout result.
