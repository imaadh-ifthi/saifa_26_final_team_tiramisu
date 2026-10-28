# Layers 3, 4 and 5 in plain language

This guide explains what the team's risk code does, what the upgrade added, and how to answer
judges' questions. Technical details: `experiments/layers345_upgrade/README.md`.

## The starting point (main pipeline, `src/quantedge/`)

Every day from 2018 to 2026, the code predicts tomorrow's risk using only data up to today.
It then checks those predictions against what really happened.

* **Layer 2 (VaR):** "On 99 days out of 100, we will not lose more than X."
  - Historical: the 1%-worst day of the last 252 days.
  - EWMA-normal: a bell curve with recent volatility.
  - Bootstrap: resample the last 252 days 10,000 times.
* **Layer 3 (ES):** "On the 1 bad day in 100, the average loss is Y." VaR says where the bad
  zone starts; ES says how deep it goes. That is why ES is reported next to VaR.
* **Layer 4 (EVT):** fit a Generalized Pareto Distribution (GPD) to the biggest losses only.
  Extreme value theory says the tail beyond a high threshold looks GPD whatever the rest of
  the distribution does, so it extrapolates beyond what has been seen.
* **Layer 5 (backtests):**
  - Kupiec: is the number of breaches right?
  - Christoffersen: do breaches follow each other?
  - Basel traffic light: how many breaches in the last 250 days?

**Main problem found:** at 99%, every one of those models fails the joint test. The breaches
cluster, for example March 2020, because the models are slow to notice that the market has
become more volatile.

## What the upgrade adds, and why

### 1. A better volatility filter: GJR-GARCH (Layers 3 and 4)
Volatility today depends on yesterday's surprise and yesterday's volatility. GJR adds a
**bigger reaction to down days** (the gamma term). For this portfolio gamma is about 0.15 and
alpha (the up-day term) is about 0.03, so a fall moves volatility about seven times more than a rise of the same size. Parameters are
estimated by maximum likelihood each month on the previous 1,000 days.

### 2. Conditional EVT (McNeil and Frey 2000) (Layer 4)
Step 1: divide each return by its GARCH volatility. These "shocks" no longer cluster.
Step 2: fit the GPD to the worst 10% of shocks.
Step 3: tomorrow's risk = tomorrow's volatility x the shock tail.

Evidence that step 1 does its job is the **extremal index**:
* 1.0 means extremes come one at a time.
* Raw losses: about 0.5, so they come in pairs.
* GARCH shocks: about 0.9 to 1.0, close to independent, which is what EVT assumes.

### 3. Choosing the EVT threshold with data (Layer 4)
The brief says "threshold selection must be justified." We compare two methods:
* the classic fixed 90% threshold;
* the EQD method (Murphy, Tawn and Varty 2025), which bootstraps a QQ-plot distance and picks
  the threshold where the GPD fits best.

Both give almost the same risk numbers, which shows the result does not hinge on the threshold.

### 4. Backtesting ES itself (Layers 3 and 5)
Kupiec and Christoffersen only count VaR breaches. Three tests now check ES directly:
* **Du-Escanciano:** on breach days, is the loss as deep as the model says, on average? Also,
  do deep losses cluster?
* **Multinomial (Kratz, Lok and McNeil):** check breaches at four levels (97.5% to 99.4%) at
  once. ES is roughly an average of these VaRs, so if all four are right, ES is right.
* **McNeil-Frey:** on breach days, (loss - ES) / volatility should average zero.

### 5. Stronger VaR tests (Layer 5)
* **Dynamic Quantile (DQ):** can yesterday's breaches, or the VaR level, predict today's
  breach? They should not.
* **Duration test:** gaps between breaches should be random (exponential).
* **Monte Carlo p-values:** with about 22 expected breaches, chi-square p-values are rough, so
  we simulate the exact null distribution.

### 6. Picking the best model, not just pass/fail (Layer 5)
* **FZ0 score:** one number scoring VaR and ES together. Lower is better, and the true
  model wins on average.
* **Diebold-Mariano:** is model A's score really lower than model B's?
* **Model Confidence Set:** the group of models statistically tied for best. Models outside
  it are beaten.

## Result in one paragraph
The conditional EVT model (`gjr_evt`) has 26 breaches at 99% where 22 are expected. It passes
Kupiec, Christoffersen, the joint test, DQ, the duration test, both Du-Escanciano ES tests,
McNeil-Frey and the multinomial test. Every baseline model fails most of these. It is
statistically tied with the alert-based candidates in the experiments folder, but needs about
12% less VaR, roughly $23k a day, to get the same coverage. Tomorrow's 99% VaR for the current
allocation is about $130k (90% interval $118k to $142k), and 99% ES is about $166k
($144k to $190k).

## Likely judge questions

* **Why not just use historical VaR?** It reacts slowly. In the 2022 rate shock it had 11
  breaches at 99% in one year (2.5 expected), against 2 for conditional EVT.
* **Why both VaR and ES?** VaR ignores how bad the bad days are. ES measures it. Our ES tests
  check that the depth is right.
* **How did you choose the EVT threshold?** Literature default (top 10% of shocks), confirmed
  by a data-driven method (EQD) that gives nearly the same answer.
* **How confident are you?** We bootstrap the tail fit. The 90% interval for 99% VaR is about
  plus or minus 9%, and for ES about plus or minus 14%.
* **Isn't this overfitted?** Settings were fixed before the run and come from published
  papers. But 2018-2026 was looked at before, so we call it development evidence. A truly
  independent check needs future data.
* **What does the model miss?** Overnight jumps bigger than anything in the last 1,000 days,
  sudden changes in correlation between assets, liquidity and trading costs, and uncertainty
  in the GARCH parameters themselves.

## Crisis-response upgrade: making the model react faster (experiments/crisis_response)

### The problem in one line
`gjr_evt` passes every test over 2018-2026, but in 2020 it had 8 days at 99% where the loss
beat VaR, when about 2.5 are expected. The model learns about a crisis only from yesterday's
closing return, which is a slow and noisy signal.

### What we tried (12 versions, all from published research)
* **Add the VIX** (the market's "fear gauge") to the GARCH volatility equation. Options traders
  often react before prices do. This is called GARCH-X.
* **Add the daily high-low range.** A day can swing 4% and still close flat; the range catches that.
* **Refit every day** instead of every month, and a **500-day** instead of a **1,000-day** window.
* **Model each asset separately**, then rebuild the portfolio with today's volatility of each
  asset. One version also tracks **changing correlations (DCC)**, so it notices when bonds stop
  protecting us, as in March 2020.
* **Two regimes** ("calm" and "stressed") with a Markov-switching GARCH.

### How we made sure we weren't fooling ourselves
1. We wrote the list of models and the rule for picking the winner into a file **first**, and
   saved it in git.
2. We picked the winner using **2011-2017 only** (that includes the August 2011 crash). The rule:
   it has to pass the main backtests, then it needs the lowest FZ0 score (a score for VaR and ES
   together; lower is better).
3. We **froze** the winner, with a fingerprint (hash) of the settings, before ever running 2018-2026.
4. We tested on 2018-2026 **once**. We also show every other model's test numbers, so nobody
   has to wonder about cherry-picking.

### The winner and what it did
**`gjrx_vix_evt_d1000`**: GJR-GARCH plus yesterday's VIX, refit every day, with the same EVT
tail as before.

| 99%, 2018-2026 | New | gjr_evt | Daniel's alert model |
|---|---:|---:|---:|
| Breaches (22 expected) | 26 | 26 | 22 |
| Breaches in all of 2020 | **5** | 8 | 3 |
| Breaches in the COVID crash (20 Feb-30 Apr) | 3 | 3 | 2 |
| Average daily VaR | **$161.5k** | $167.1k | $190.5k |
| Average VaR in calm years | **$139.9k** | $143.6k | $159.1k |
| FZ0 score (lower is better) | **-3.881** | -3.800 | -3.798 |

How to say it: "Adding the VIX made the model better at the edges of a crisis. In 2020 it cut
the breaches from 8 to 5, and it uses about $5-6k less risk capital a day on average. It does
not stop the first days of a crash. No daily model does; even US regulators excluded 6 to 27
March 2020 for banks."

### Be upfront about the weak spots
* At 97.5% it has 72 breaches where 55 are expected, so the breach-count test rejects it. The
  reason: the VIX only measures stock-market fear. In 2022 bonds and stocks fell together on
  rate hikes while the VIX stayed moderate, so the model was too relaxed. In the very low-VIX
  years (2023, 2026) it was also a bit too low.
* Its edge over gjr_evt is real on the score but not significant at 5% (p = 0.06 to 0.08).
* The asset-level DCC model did best in the crash (1 breach), but it failed our 2011-2017 rule.
  We didn't switch to it after seeing the test, because that would be cheating. It's the first
  thing to re-test on new data.

### Likely judge questions on this part
* **"Didn't you just pick the model that looked best on COVID?"** No. The choice was frozen
  and committed in git before 2018-2026 was run, and the test script refuses to run if the
  settings change. The model with the best COVID numbers (asset DCC) is *not* the one we picked.
* **"Is the VIX a fair input?"** Yes. It's public (FRED VIXCLS), and we only use yesterday's close.
* **"Which model would you use?"** For the 95% and 97.5% VaR lines, gjr_evt, because it covers
  better there. For 99% VaR and ES, and as an early-warning view, the VIX model.

## From numbers to decisions (experiments/crisis_response/decision.py)

* **Risk budget.** We split the portfolio's ES into one piece per asset; the pieces add up to
  the total. Today gold is 15% of the money but about 28% of the tail risk, because it's
  moving with stocks (correlation 0.32, normally 0.08). XLK is 6% of the money and 16% of the
  risk, more than twice its weight, so we flag it for trimming. We build these numbers from
  every daily shock since 2007, re-played at today's volatility and correlation. That makes them
  steadier than the old 252-day split, which put 51% of ES on gold.
* **Stress tests** with today's weights: 2008 -24.8%, COVID -16.6%, 2022 -18.0%, and our own
  scenario "hedge failure", where stocks, bonds and gold are all sold at once, -13.2%.
  Reverse stress test: the S&P 500 would need to fall about 48% for us to lose 20%.
* **Trade menu.** Every 5% swap allowed by the mandate, with its effect on ES, the worst stress
  loss and expected return. Bonds and gold are already at their caps, so no swap gets a 2008
  repeat below 24.3%.
* **Memo** (`docs/INVESTMENT_COMMITTEE_MEMO.md`): accept the allocation, but raise the
  stress-loss limit from 20% to 26%, trim XLK, and keep a daily 97.5% ES limit of $200k. The 2%
  and 20% limits are our assumptions until the client brief gives its numbers.
