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
