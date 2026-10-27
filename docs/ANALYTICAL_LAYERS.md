# Analytical layers and output contract

Run `python -m src.quantedge.main` for the complete pipeline. Run `python -m src.quantedge.analytics --risk-free data/raw/risk_free.csv` to use saved prices and current portfolio JSON without network access. Omit `--risk-free` to download public FRED DGS3MO history. Both commands run from the repository root.

For a complete offline pipeline run, including regenerating the current allocation, use `python -m src.quantedge.main --prices outputs/tables/prices.csv --risk-free data/raw/risk_free.csv`.

## Allocation and timing

`validation` declares the training and test periods. Defaults are 2010-01-01 through 2017-12-31 for training and 2018-01-01 onward for testing. Means, CAPM inputs, covariance and portfolio optimization for the validation allocation use only complete training observations. The risk-free quote used for that optimization must be dated no later than the final training observation. An infeasible training target is an error; no full-sample substitution occurs.

The validation allocation is frozen. Applying those same weights to each day's simple asset returns assumes daily rebalancing without costs. Risk models may update within the test period, always using information strictly before the forecast date. Training data supply warmup. The EVT and other model warmups must be covered before the test period begins. All methods/confidences are validated over the same complete evaluation dates; internal availability gaps are errors rather than being silently bridged for independence tests.

The current allocation in `selected_portfolio.json` has a separate purpose: retrospective stress replay and current-position risk budgeting. Do not label that allocation's full-history performance as out-of-sample. The standalone analytical command does not regenerate this allocation or the original retrospective VaR files.

## VaR and Expected Shortfall

The shared forecast table includes `date`, `method`, `confidence`, `estimation_end`, `allocation_training_end`, `evaluation_basis`, `actual_log_return`, `actual_simple_return`, `actual_loss_usd`, `var_log`, `es_log`, `var_simple_pct`, `es_simple_pct`, `var_usd`, `es_usd`, `exception`, and `evaluation_index`. Additional EVT columns describe threshold, shape, scale and selection status. Dollar forecasts use the constant configured notional; this is not a simulation of changing portfolio NAV. A forecast on date t is for the return realized on t, using observations through t-1. The export does not include an unevaluated next-business-day forecast.

* Historical: trailing empirical quantile of log returns, with NumPy's linear quantile convention. ES averages exactly the worst `(1-c)*n` empirical probability mass, including a fractional boundary observation. This gives stable behavior when the tail sample size is not integral.
* EWMA parametric: zero conditional mean normal log returns; volatility is pandas' bias-corrected EWMA standard deviation, shifted one observation. This is an EWMA volatility estimator, not a claim to reproduce the zero-mean RiskMetrics variance recursion. Log VaR is `z_c*sigma` and log ES is `sigma*phi(z_c)/(1-c)`.
* Bootstrap Monte Carlo: resample the prior empirical window with replacement. One seeded simulation draw is reused across confidence levels on a date. This is an empirical bootstrap, not a separate structural return process; its similarity to historical risk is expected.
* EVT: a GPD fitted to positive excesses of log losses over a selected threshold.

Simple loss is `1-exp(log_return)`. VaR is monotone under this transformation, but **transforming mean log ES does not give dollar ES**. Historical/bootstrap ES averages transformed tail losses. Normal simple-loss ES is `1-exp(sigma²/2)*Phi(-z_c-sigma)/(1-c)`. EVT simple ES integrates the conditional GPD survival function after transforming losses. Loss risk measures are floored at zero for reporting; portfolio contributions retain signed values so Euler sums remain valid.

## EVT threshold evidence

Every configured candidate threshold exports its quantile, log-loss threshold, number and fraction of exceedances, sample mean excess, fitted shape/scale, modified scale `beta-xi*u`, and fitted KS distance. The KS statistic is a descriptive fit diagnostic: no unadjusted fitted-parameter KS p-value is treated as a valid test.

A threshold is marked stable when its shape and modified scale are sufficiently close to the adjacent higher threshold, under declared tolerances. Selection chooses the lowest stable eligible threshold; it must support all requested confidence levels. If no eligible stable threshold exists, the eligible fit with the lowest KS distance is used with `warning_no_stable_threshold`. This warning requires human review and is not evidence that the threshold is appropriate. If no eligible fit exists, evaluation fails rather than inventing a tail estimate. These are transparent heuristics for review, not proof of a unique optimal threshold.

Fits use a trailing 1,000 observations and refresh every 21 observations by default. Between fits the parameters and risk estimates are held constant; `estimation_end` records the actual fit cutoff. The sensitivity table includes VaR and ES at every supported confidence for every candidate threshold at each refit. Use it to review mean-excess behavior, parameter stability, and sensitivity before defending the final threshold rule.

For exceedance probability `p_u`, threshold `u`, shape `xi` and scale `beta`, the extrapolated loss quantile is `u + beta/xi * ((p_u/(1-c))^xi - 1)`, with the exponential limit at zero shape. Log ES is `(VaR + beta - xi*u)/(1-xi)` when `xi<1`; otherwise it is infinite and `log_es_finite` is false. Transformed simple loss is bounded by 100%, so its ES can remain finite even when log-loss ES diverges. The survival integral is truncated beyond excess log loss 40, bounding the omitted contribution by `exp(-40)` for nonnegative VaR. Volatility clustering, dependent extremes and threshold uncertainty remain limitations; this implementation does not claim filtered-EVT or uncertainty intervals.

## Formal validation

Kupiec POF compares the realized exception rate with `1-c` using a one-degree-of-freedom likelihood ratio. Christoffersen independence compares transition probabilities following exception/nonexception states and exports `n00`, `n01`, `n10`, `n11`. Conditional coverage sums the POF and independence likelihood ratios and uses two degrees of freedom. The significance level is configurable (default 5%). Zero-count likelihood terms use their limiting values, avoiding `log(0)` arithmetic errors. When one transition origin state is unobserved, independence and joint coverage are marked unavailable instead of reporting a spurious pass. Chi-square p-values are asymptotic; rare exceptions and short samples require caution. Non-rejection does not prove calibration.

The legacy Basel traffic light uses the **last actual 250 evaluation observations at 99%**: 0–4 green, 5–9 yellow, 10+ red. Shorter windows are `insufficient_data`; other confidences are `not_applicable`. Full-history exception counts are never prorated. This is the legacy comparison requested in the challenge, not a regulatory compliance claim.

## Stress replay and position sizing

Historical scenarios replay 2008–March 2009, the February–March 2020 crash and 2022. Daily rebalancing or buy-and-hold may be configured. Exports include the full wealth/drawdown paths, cumulative P&L, worst day and maximum drawdown measured against initial wealth. Missing boundaries or incomplete source rows are flagged `partial_data`; an absent scenario is `missing_data`. These scenarios use today's allocation and are retrospective sensitivity exercises, not implementable historical investment decisions.

The configured stagflation scenario applies one simultaneous simple-return shock to every asset. Its values are team-editable assumptions, not estimated probabilities. Missing asset shocks fail validation. Position shock P&L contributions sum exactly to total scenario P&L.

Risk contributions use the current allocation and latest historical risk window. Historical simple-return VaR gradients interpolate the asset-loss vectors at the portfolio quantile; historical ES gradients use the same fractional tail weights as portfolio ES. Euler components are exposure times the unconstrained marginal derivative, and sum to portfolio risk. The normal simple-return approximation includes sample mean and covariance and supplies analytical gradients. It is labeled `parametric_normal_simple` because it is distinct from the forecast model's EWMA-normal log-return distribution. Negative components may reflect hedging. These derivatives describe changes in notional exposure, not a constrained trade that automatically finances itself; use marginal differences when comparing financed reallocations. Contributions support decisions but do not prescribe an automatic allocation change.

## Files

| File under `outputs/tables/` | Contents |
|---|---|
| `risk_forecasts_oos.csv` | Shared evaluated VaR/ES forecasts for four models |
| `risk_summary_oos.csv` | Latest and average risk, exception counts, observation counts |
| `backtest_results.csv` | POF, independence, conditional coverage and traffic light |
| `evt_threshold_diagnostics.csv` | Each refit's candidate fits, selection flags and sensitivity estimates |
| `oos_portfolio_weights.csv` | Training-only allocation, explicit `asset,weight` columns |
| `oos_performance.csv` | Portfolio, policy benchmark and market daily returns; wealth and as-of risk-free rate |
| `oos_kpi_summary.csv` | Performance KPIs using historical as-of risk-free observations |
| `stress_summary.csv`, `stress_paths.csv` | Historical/hypothetical impacts and historical wealth paths |
| `stress_shock_contributions.csv` | Assumed shocks and position P&L contributions |
| `risk_contributions.csv` | Marginal/component VaR and ES, dollar components and risk shares |

`outputs/models/analytical_run.json` records the exact configuration, training dates, quote date, both allocations and evaluation dates. Retain it with the exported tables. Prices and FRED data remain third-party observations that may be revised. Dependency pinning and immutable source snapshots are still needed for exact long-term reproducibility.

## Method references

* [SciPy GPD definition and fitting API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html)
* [MathWorks conditional coverage documentation and reference example](https://www.mathworks.com/help/risk/varbacktest.cc.html); the tests reproduce its published POF and independence likelihood ratios.
* [Basel Committee's original supervisory backtesting framework](https://www.bis.org/publ/bcbs22.pdf)

The challenge also requires team-authored interpretation, an investment memo, a tear sheet and AI-use disclosure. These outputs provide evidence for those deliverables; they do not supply a team-approved investment recommendation.
