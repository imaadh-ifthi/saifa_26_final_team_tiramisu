# Crisis response: making VaR and ES react faster in a sudden crash

Our best model so far, `gjr_evt` (GJR-GARCH filter + EVT tail, refit monthly on 1,000 days),
passes every backtest on 2018-2026. Its weak spot is 2020: 8 breaches at 99% in one year
when about 2.5 are expected. This folder tests a set of ideas from the research literature
that should help a model notice a crisis sooner. It then picks one with a rule written down
in advance and tests it once.

The results are in [REVIEW.md](REVIEW.md) (what worked and what didn't), [results/dev/DEV_RESULTS.md](results/dev/DEV_RESULTS.md)
(how the model was chosen) and [results/test/RESULTS.md](results/test/RESULTS.md) (the one-shot test).
The investor decision layer is in [results/decision/DECISION.md](results/decision/DECISION.md), and the
one-page committee memo is [docs/INVESTMENT_COMMITTEE_MEMO.md](../../docs/INVESTMENT_COMMITTEE_MEMO.md).

## How to run it

From the repository root, inside the project's `.venv`:

```bash
python -m experiments.crisis_response.download_data      # once: VIX (FRED + Yahoo) and OHLC into data/raw/
python -m experiments.crisis_response.run --stage dev    # about 10 min: picks the model on 2011-2017, writes frozen_choice.yaml
python -m experiments.crisis_response.run --stage test   # about 15 min: one evaluation on 2018-01-02 .. 2026-10-20
python -m experiments.crisis_response.decision           # about 1 min: risk budget, stress tests, memo
pytest tests/test_crisis_response.py
```

`data/raw/` is git-ignored, like the rest of the team's raw data, so run `download_data`
once on a new machine. Yahoo sometimes revises old high/low prices a little, so a re-download
can shift results in the last decimal places.

## The candidates

All of them forecast tomorrow's return distribution using data up to today only, then turn
it into VaR and ES with the same code (`src/quantedge/tail_grid.py`). Unless it says
otherwise, the tail is a GPD fitted on the worst 10% of standardised shocks, as in `gjr_evt`.

| Name | What's different | Why it might help | Source |
|---|---|---|---|
| `gjr_evt_m1000` | nothing: this is Bhavee's `gjr_evt`, rebuilt in the new engine | the reference | McNeil and Frey (2000) |
| `gjr_evt_d1000` | parameters refitted every day, not every 21 days | the model can learn about a new regime within days | |
| `gjr_evt_m500`, `gjr_evt_d500` | 500-day window instead of 1,000 | older calm years weigh less | |
| `gjrx_vix_evt_*` | yesterday's VIX in the variance equation (GARCH-X) | the VIX jumps before realised volatility catches up | Lynch, Hasan and Siddique (2023); Bams, Blanchard and Lehnert (2017) |
| `gjrx_range_evt_*` | yesterday's high-low range variance in the variance equation | a wild intraday swing shows up even when the close is flat | Molnar (2016); Hansen, Huang and Shek (2012) |
| `har_range_evt_m1000` | sigma from a HAR model on the range variance (day / week / month averages) | a fast, simple realised-volatility forecast | Corsi (2009) |
| `asset_fhs_evt_m1000` | GJR filter on each asset, 1,000 portfolio scenarios rebuilt at today's asset volatilities | if only bonds suddenly get volatile, the portfolio model sees it | Barone-Adesi, Giannopoulos and Vosper (1999) |
| `asset_dcc_evt_m1000` | as above, and the shocks are re-correlated with today's DCC correlation | shows a broken stock-bond hedge straight away | Engle (2002) |
| `ms_gjr_t_m1000` | two-regime Markov-switching GJR with Student-t shocks (no EVT) | a separate "stress" regime with its own dynamics | Haas, Mittnik and Paolella (2004); Ardia et al. (2018) |

The VIX is FRED `VIXCLS` (Yahoo `^VIX` only fills gaps). OHLC prices come from Yahoo through
`yfinance`. Each input for day t is the value from day t-1 (`shift(1)` in `models.py`), and
the tests check that changing anything from day t onwards doesn't change the forecast for t.

## How we avoided overfitting

1. **Everything was written down first.** The candidate list, the windows, the selection
   rule and the reporting windows are in `config.yaml`. That file went into a git commit
   before any result existed (`git log -- experiments/crisis_response/config.yaml`).
2. **The choice used 2011-2017 only.** The dev stage forecasts 2011-01-03 to 2017-12-29
   (1,761 days, includes the August 2011 crash and the 2015-16 sell-off). The rule:
   - gates: not rejected (p >= 0.05) by joint coverage or DQ at 95% and 99%, or by
     Du-Escanciano or McNeil-Frey ES tests at 97.5%;
   - score: the average of mean FZ0 loss at 95%, 97.5% and 99%;
   - winner: lowest score among models passing every gate.
3. **Then it was frozen.** `frozen_choice.yaml` stores the winner, the time, and SHA-256
   hashes of `config.yaml` and the model code. The test stage won't run if the config has
   changed, and it reports any code file that changed.
4. **The test runs once.** Every candidate is shown on the test period too, in an appendix,
   so nobody has to guess whether a different pick would have looked better. They were not
   used to choose.

Two caveats here. The frozen 2007-2017 allocation was fitted on data that includes the dev
period, as the brief requires. And 2018-2026 has been looked at before by the team, so it
is not a perfectly untouched holdout. Details are in REVIEW.md.

## Files

| File | What it is |
|---|---|
| `config.yaml` | candidates, periods, selection rule (pre-registered) |
| `frozen_choice.yaml` | the frozen winner plus hashes (written by the dev stage) |
| `models.py` | the forecast engine for every candidate |
| `run.py` | dev and test stages, all backtests and reports |
| `decision.py`, `decision.yaml` | risk budget, stress tests, trade menu, committee memo |
| `download_data.py` | downloads VIX and OHLC into `data/raw/` |
| `results/dev/` | development forecasts, backtests, ranking |
| `results/test/` | test forecasts, backtests, crisis windows, yearly tables |
| `results/decision/` | risk budget, stress tests, trade menu |

Reusable code is in `src/quantedge/`: `garch_x.py`, `range_vol.py`, `dcc.py`, `ms_garch.py`,
`model_selection.py` and `decision.py`.
