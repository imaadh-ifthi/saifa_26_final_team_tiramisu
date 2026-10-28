# Crisis response: what improved and what didn't

Short version: the frozen winner, **`gjrx_vix_evt_d1000`** (GJR-GARCH with yesterday's VIX in the
variance equation, refit every day, GPD tail on the shocks), is a real but modest improvement
on `gjr_evt`. It cuts the 2020 breaches at 99% from 8 to 5. It scores better on FZ0 at every
level and needs less risk capital on average. But it does **not** reduce the breaches in the
COVID crash itself (3 and 3). Its edge over `gjr_evt` is borderline, not significant at 5%.
And it has too many breaches at 97.5%. It's not a clean win on every test, and we say so below.

All numbers: [results/test/RESULTS.md](results/test/RESULTS.md) (test) and
[results/dev/DEV_RESULTS.md](results/dev/DEV_RESULTS.md) (how it was picked).

## How the choice was made (and why you can trust it wasn't tuned on COVID)

1. 12 candidates and the selection rule were written into `config.yaml` and committed
   (commit `6fe6ff9`) before any result existed.
2. All 12 were run on 2011-01-03 to 2017-12-29 only (1,761 forecast days, inside training).
   The rule: must pass joint coverage and DQ at 95% and 99%, and the Du-Escanciano and
   McNeil-Frey ES tests at 97.5%. Then pick the lowest average FZ0 over 95/97.5/99%.
3. The winner was written to `frozen_choice.yaml` with hashes of the config and code, and
   committed (commit `4d35769`) **before** the 2018-2026 run. The test stage checks the
   hash and reported "code changed after freezing: none".
4. 2018-01-02 to 2026-10-20 was then run once. Same frozen 2007-2017 allocation, same 2,212 days.

Dev ranking (top): `gjrx_vix_evt_d1000` -4.5363, `ms_gjr_t_m1000` -4.5249, `gjrx_range_evt_m1000` -4.5071.
`gjr_evt_m1000` (your current model) scored -4.5036 and failed one gate (joint coverage at 95%,
p = 0.014). On dev, the winner beat it on FZ0 at 95% with DM p = 0.045.

**One slip to disclose.** While timing the code, before the config was written, I ran four
candidates (asset_dcc, ms_gjr, har_range, gjrx_vix daily) over Feb-Apr 2020 and saw their
breach counts. Nothing was changed because of it. The candidate list was already in
`models.py`, and the rule picks on 2011-2017 only. But a judge should know it happened.

## 1. Full test period vs gjr_evt and Daniel's best

| | Conf. | Breaches (expected) | Joint p | DQ p | ES tests (DE-UES / MF) | Mean FZ0 | Avg VaR $ | Avg ES $ |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **gjrx_vix_evt_d1000** | 99% | 26 (22) | 0.43 | 0.38 | 0.14 / **0.046** | **-3.881** | **161,508** | **198,274** |
| gjr_evt (yours) | 99% | 26 (22) | 0.43 | 0.34 | 0.15 / 0.097 | -3.800 | 167,126 | 216,249 |
| filtered_historical_alert (Daniel) | 99% | 22 (22) | 0.80 | 0.63 | n/a / 0.31 | -3.798 | 190,533 | 242,033 |
| filtered_evt_cusum (Daniel) | 99% | 25 (22) | 0.63 | 0.24 | n/a / 0.81 | -3.807 | 189,490 | 246,899 |
| **gjrx_vix_evt_d1000** | 97.5% | **72 (55)** | 0.086 | **0.024** | 0.11 / 0.40 | **-4.106** | 126,196 | 164,082 |
| gjr_evt (yours) | 97.5% | 63 (55) | 0.41 | 0.059 | 0.10 / 0.13 | -4.058 | 127,470 | 172,959 |
| **gjrx_vix_evt_d1000** | 95% | 131 (111) | 0.12 | **0.019** | 0.051 / 0.21 | **-4.290** | 98,693 | 137,505 |
| gjr_evt (yours) | 95% | 124 (111) | 0.31 | 0.023 | 0.098 / 0.19 | -4.271 | 99,056 | 142,250 |
| filtered_historical_alert (Daniel) | 95% | 111 (111) | 0.97 | 0.012 | n/a / 0.75 | -4.243 | 106,133 | 160,844 |
| **gjrx_vix_evt_d1000** | 90% | 244 (221) | **0.036** | 0.077 | **0.048** / 0.10 | **-4.518** | 70,449 | 110,261 |
| gjr_evt (yours) | 90% | 248 (221) | 0.077 | 0.077 | 0.053 / 0.25 | -4.509 | 71,741 | 113,000 |
| filtered_historical_alert (Daniel) | 90% | 232 (221) | 0.74 | 0.004 | n/a / 0.88 | -4.483 | 76,277 | 125,360 |

Bold p-values are rejections (p < 0.05). Daniel's files only store 90/95/99%, and they don't
store the full distribution, so the DE ES test can't be run on them.

Diebold-Mariano on FZ0 (negative = new model better):

| Conf. | vs gjr_evt | vs filtered_historical_alert |
|---:|---|---|
| 90% | -0.009, p = 0.41 | -0.034, **p = 0.028** |
| 95% | -0.019, p = 0.25 | -0.047, **p = 0.032** |
| 97.5% | -0.048, p = 0.061 | n/a |
| 99% | -0.081, p = 0.084 | -0.083, p = 0.135 |

All main models stay in the 90% Model Confidence Set at every level, so no one is
statistically thrown out. The new model is first or second by FZ0 at every level.

## 2. COVID and 2020

| 99% | COVID crash, 20 Feb-30 Apr 2020 | All of 2020 | Avg 99% VaR in the crash |
|---|---:|---:|---:|
| expected | 0.5 | 2.5 | |
| **gjrx_vix_evt_d1000** | 3 | **5** | $460,957 |
| gjr_evt (yours) | 3 | 8 | $496,814 |
| filtered_historical_alert (Daniel) | 2 | 3 | $700,706 |
| filtered_evt_cusum (Daniel) | 3 | 8 | $630,373 |

2020 breach dates at 99%:
* new: 02-24, 03-09, 03-12, 06-11, 09-03
* gjr_evt: 01-27, 02-24, 03-09, 03-12, 06-11, 08-11, 09-03, 10-28

So the improvement is **outside** the crash. gjr_evt had 5 breaches in 2020 outside the crash
window; three of them (01-27, 08-11, 10-28) are gone. In late January the VIX jumped on the
first virus news before returns moved. From June to October the VIX stayed high while
realised volatility came down, and the VIX term kept VaR from dropping too fast. Inside the crash, the first big days (24 Feb, 9 and 12 March)
still breach. No daily model catches the first day of a shock. The Fed note (Abboud et al.
2021) found about 6 exceptions in March 2020 even for the best daily-updated GARCH, and
regulators excluded 6-27 March 2020. None of the new model's 2020 breaches come two days in a
row (over the whole test there is one back-to-back pair at 99%).

Daniel's alert model has fewer 2020 breaches (3), but it does it by holding a much bigger
VaR in the crash ($701k vs $461k on average), and its alert rule was designed after looking
at these dates.

## 3. Calm years (average VIX below 20: 2018, 2019, 2021, 2023, 2024, 2025, 2026)

| 99%, 1,708 calm days | Breaches (17.1 expected) | Avg VaR $ | Avg ES $ |
|---|---:|---:|---:|
| **gjrx_vix_evt_d1000** | 19 | **139,923** | **170,963** |
| gjr_evt (yours) | 16 | 143,590 | 183,639 |
| filtered_historical_alert (Daniel) | 17 | 159,131 | 201,987 |

The new model is **not** inflated in calm years. If anything it's a bit lower than gjr_evt.
That's the point of the VIX: when options traders are relaxed, VaR comes down a little.
19 breaches against 17.1 expected is well within normal noise. At 97.5% the calm-year count is 48 vs 42.7 expected.

## 4. What didn't work, or got worse

* **Too many breaches at 97.5% and 95%.** 72 vs 55 expected at 97.5% (Kupiec rejects, p = 0.030;
  DQ rejects, p = 0.024), against 63 for gjr_evt. The 9 extra come from two places, and the
  cause is the same for both: the VIX said "calm" when the portfolio wasn't.
  - **2022 (+4):** bonds and stocks fell together on inflation and rate hikes, but the VIX
    stayed moderate. The VIX only measures expected S&P 500 volatility, so it doesn't see bond
    risk. It pulled VaR down (2022 average 99% VaR $232,568 vs $260,260 for gjr_evt).
  - **Low-VIX years 2023 and 2026 (+6):** very low VIX readings pushed VaR a bit too low.
* **ES at 99% is slightly small** by the McNeil-Frey test (p = 0.046, a borderline rejection).
* **2018 is worse**: 5 breaches at 99% vs 2. VIX was very low going into the Feb 2018 "Volmageddon",
  and the model trusted it.
* **The gain over gjr_evt is not significant at 5%** (DM p = 0.06 to 0.08 at the tail levels).
* **The candidates the rule rejected** (appendix of RESULTS.md, not used for choosing). Asset-level
  FHS and DCC had only 1 COVID-crash breach and the best 99% coverage (23 and 24 breaches).
  But they failed two dev gates (joint coverage and DQ at 95% on 2011-2017), so the rule
  dropped them. Switching to them now would be choosing on the test set, so we don't. They're
  the obvious thing to re-test on post-2026 data.
* **Markov-switching GJR** looked strong on dev (rank 2) and failed badly on test (47 breaches at
  99%). It shows exactly why we freeze before testing: a good dev score is not enough.
* **Daily refits on their own** (gjr_evt_d1000) and **500-day windows** didn't help.
  The 500-day window made 99% worse (30-32 breaches).

## 5. Bottom line

| Goal from the brief | Result |
|---|---|
| Fewer breaches in 2020 | Yes: 8 to 5 at 99% |
| Fewer breaches in the crash itself | No: 3 and 3 |
| Less clustering | Yes, a little: no back-to-back breaches in 2020; ES clustering test DE-CES p = 0.22 vs 0.12 |
| Not over-conservative in calm years | Yes: calm-year VaR $139.9k vs $143.6k |
| Better overall forecast | Better FZ0 at every level, but not significant at 5% vs gjr_evt; significant vs Daniel at 90/95% |
| Passes all tests | No: fails at 97.5% (count, DQ), 95% (DQ, both models fail), 90% (joint); borderline ES at 99% |

Our reading: keep `gjr_evt` as the main model for the 95% and 97.5% VaR lines, where it
covers better. Present `gjrx_vix_evt_d1000` for the 99% and ES numbers and as the
"crisis-aware" model, which is where it helps. If the team wants one model only, `gjr_evt`
is the safer choice for passing every test. The VIX version is the better forecaster by
the scoring rule. Either way, tell the judges both, and show this table.

## Caveats

* 2018-2026 has been looked at by the team before, so it's not a perfectly untouched holdout.
  Freezing on 2011-2017 limits the damage but can't undo earlier looks.
* The frozen allocation was fitted on 2007-2017, which includes the dev period.
* At 99% only about 22 breaches are expected, so every test has low power.
* The VIX closes 15 minutes after the ETFs. We use the previous day's close, so there's no look-ahead.
* Yahoo's high/low history can be revised slightly. Re-downloading can move the range models in the last decimals.
