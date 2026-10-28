# Breach mechanism review and fixed volatility probe

This is a follow-up on the already inspected 2018-2026 development period. It does not fit a change detector, search over EWMA settings, or promote a model. The companion experiment declares exactly one alternative, lambda 0.90, before comparison with the existing 0.94 filter. Both use the same frozen 2007-2017 portfolio and evaluation dates.

## 1. Raw EVT during March 2020

The selected raw EVT fit on **March 5** used its trailing 1,000 returns through **March 4**. It chose the 85th-percentile threshold with 150 exceedances, shape 0.0427, and 99% VaR of approximately **$125,145**. Its scheduled next refit was **April 3**, using data through April 2; by then the selected shape had increased to 0.3549 and VaR to about **$180,670**. The March 9, 12, 16 and 18 losses therefore had no opportunity to enter the March 5 fit before those forecasts were made. This is a direct consequence of caching the entire raw EVT VaR between 21-trading-day refits.

| Date | Actual loss | Raw EVT VaR | Filtered EVT VaR | Filtered historical VaR |
|---|---:|---:|---:|---:|
| Mar 9 | $366,800 | $125,145 | $309,483 | $301,038 |
| Mar 12 | $664,924 | $125,145 | $451,635 | $506,153 |
| Mar 16 | $512,092 | $125,145 | $723,229 | $831,453 |
| Mar 18 | $393,275 | $125,145 | $779,519 | $895,752 |

Filtered EVT's **tail shape was also last fitted using data through March 4** on these dates, but it multiplied that shape by a fresh, prior-information volatility forecast each day. It breached on March 9 and 12, then covered March 16 and 18. This strongly supports stale *scale* as an explanation for raw EVT's extended crisis episode. It does not establish that daily raw EVT refitting would be stable or preferable; GPD parameters estimated from only 150 tail observations can move sharply.

## 2. Filtered models after calm forecasts

Using the filtered bootstrap's prior-information volatility quartiles, the filtered historical and filtered bootstrap models each had **11 of 29** 99% breaches in the calmest quartile and **5 of 29** in the most turbulent. The calm threshold was daily EWMA volatility at or below about **0.443%**. The 11 filtered historical dates were Feb 2 and Oct 10 2018; May 13 and Aug 5 2019; Feb 24 2020; Sep 28 2021; Apr 10, Jul 17, Oct 31 and Dec 18 2024; and Oct 10 2025. They are spread across years, not solely one crisis.

On every one of those 11 dates, the **next day's** EWMA volatility estimate rose by more than 20% (median rise about 32%). That rise is largely mechanical after a return big enough to breach VaR; it must not be treated as advance warning. At the forecast time, those losses represented about 2.69 to 6.18 times the prior volatility, while the model's 99% standardized tail threshold ranged about 2.43 to 3.66. This shows the immediate issue is an unexpectedly large standardized loss following a low prior estimate. A volatility filter can respond *after* that move, but cannot foresee it from the same day's return. The asset-loss columns in `diagnostics/breach_events.csv` give realized portfolio attribution, not a causal attribution of the forecast error.

## 3. One declared faster-response comparison

[`volatility_response.yaml`](../volatility_response.yaml) declares lambda **0.90** versus **0.94**, only for filtered historical and filtered bootstrap at 99%. The script rebuilds every standardized shock with the corresponding causal volatility history, checks that lambda 0.94 reproduces saved forecasts, and uses identical 2,212 evaluation days. [The full comparison](volatility_response/comparison.md) contains p-values, quantile loss, and average VaR; `daily_forecasts.csv.gz` and `yearly.csv` contain dated outcomes.

| Model | Lambda | 99% breaches | Independence p | Combined p | Mean quantile loss | Average VaR |
|---|---:|---:|---:|---:|---:|---:|
| Filtered historical | 0.94 | 29 | 0.00568 | 0.00815 | 0.00025140 | $185,947 |
| Filtered historical | 0.90 | 31 | 0.00859 | 0.00638 | 0.00024516 | $184,363 |
| Filtered bootstrap | 0.94 | 29 | 0.00568 | 0.00815 | 0.00025304 | $189,795 |
| Filtered bootstrap | 0.90 | 31 | 0.00859 | 0.00638 | 0.00024675 | $188,848 |

The faster filter improves mean quantile loss slightly, but increases 99% breaches from 29 to 31 for both models; independence and combined coverage still reject at 5%. For filtered historical, it removes only **one of the 11 original calm-quartile breaches** and introduces five new breaches elsewhere. It raises March 12 VaR from about $506k to $545k, still below the $665k actual loss. It is therefore **not a demonstrated fix**. The changed volatility filter also changes standardized historical shocks, so individual VaR forecasts need not always rise even though lambda is smaller.

## Implication

Raw EVT has a specific update-cadence weakness; a separated daily scale update is already demonstrated by filtered EVT, though that model also fails 99% independence. The filtered models' calm-period breaches often begin with an unanticipated shock. A single globally faster EWMA response does not solve that pattern on this reviewed period. The next research question should distinguish a tail-distribution problem from a predictable volatility precursor, and any model-selection claim needs genuinely uninspected observations.
