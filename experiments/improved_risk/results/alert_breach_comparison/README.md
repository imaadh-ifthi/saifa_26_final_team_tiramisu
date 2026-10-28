# Three alert candidates and their 99% VaR breaches

[Full evaluation chart](99pct_breaches_all_dates.png) | [February-March 2020 detail](99pct_breaches_march_2020.png)

The first three rows compare one candidate with its unchanged reference. The fourth shows the original raw Monte Carlo bootstrap beside the same filtered-bootstrap alert candidate (shown again for clarity). There are three distinct new candidates, not four. All panels use the same 2,212 dates and realized losses. Red dots are VaR breaches. Hollow green circles mark reference breaches that the alert variant avoided. The plots use the same vertical scale so forecast size is directly comparable.

| Family | Original breaches | Alert breaches | Avoided | New | Alert Kupiec p | Alert independence p | Alert combined p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Filtered EVT | 31 | 25 | 6 | 0 | 0.5466 | 0.4495 | 0.6265 |
| Filtered historical | 29 | 22 | 7 | 0 | 0.9795 | 0.5061 | 0.8014 |
| Monte Carlo, filtered | 29 | 22 | 7 | 0 | 0.9795 | 0.5061 | 0.8014 |
| Monte Carlo, raw to alert | 32 | 22 | 19 | 9 | 0.9795 | 0.5061 | 0.8014 |

The three distinct alert candidates are not rejected by these three **99% VaR** tests at 5% on the previously inspected development period. This is not independent validation. Raw EVT with daily scaling plus an alert is excluded because it still rejects independence (p about 0.018). ES is not assessed by these VaR tests.
