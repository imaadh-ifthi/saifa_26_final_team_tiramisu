# Training-start change: 2010 to 2007

Both `estimation.core_start_date` and `validation.train_start` are now 2007-01-01. Validation still freezes its allocation after 2017 and evaluates 2018 onward. The source prices and FRED snapshot are unchanged. Saved-price readers use round-trip float parsing to avoid rewriting their numerical values. The earlier run is preserved in commit `303ba13`.

## Current allocation

This allocation uses all available observations from the configured estimation start, through the latest saved data. It is distinct from the validation allocation.

| Asset | Previous 2010-start weight | New 2007-start weight |
|---|---:|---:|
| SPY | 23.17% | 22.81% |
| EFA | 11.84% | 9.61% |
| EEM | 1.33% | 0.00% |
| AGG | 45.00% | 45.00% |
| GLD | 15.00% | 15.00% |
| XLE | 2.48% | 1.27% |
| XLK | 1.17% | 6.31% |

## Historical stress replay

| Scenario | Previous cumulative return | New cumulative return | Previous maximum drawdown | New maximum drawdown |
|---|---:|---:|---:|---:|
| GFC 2008 | -21.34% | -21.03% | -25.40% | -24.84% |
| COVID crash | -15.82% | -15.28% | -17.24% | -16.55% |
| 2022 rate shock | -10.85% | -12.18% | -17.00% | -17.99% |

The 2008 replay improves modestly, while the 2022 replay worsens. Including a crisis in estimation does not guarantee lower losses in every scenario. Both current-allocation replays are retrospective. The 2008 crisis is now in the estimation/training sample and cannot be presented as an independent crisis test.

## Frozen validation allocation

The training-only portfolio uses 2007-01-04 through 2017-12-29 (2768 complete observations). It retains the configured 6.5% estimated-return target and all exposure limits.

| Asset | Validation weight |
|---|---:|
| SPY | 32.09% |
| EFA | 0.00% |
| EEM | 0.00% |
| AGG | 37.91% |
| GLD | 15.00% |
| XLE | 0.00% |
| XLK | 15.00% |

The optimizer still minimizes estimated variance subject to its return target and constraints. It does not directly minimize the 2008 loss or maximum drawdown. Crisis data influence the estimated means and covariance; an explicit crisis-loss constraint would be a separate portfolio-design decision.

Updated baseline backtests are in `outputs/tables/backtest_results.csv`. The separate experimental comparison must be read with its regenerated manifest and results review; the previous 2010-start statistics do not describe this new allocation.

## Out-of-sample portfolio performance, 2018 onward

These compare the separately frozen training allocations on the same later evaluation dates. They are distinct from the current-allocation stress replays above.

| Metric | 2010-2017 training | 2007-2017 training |
|---|---:|---:|
| Annualized return | 10.16% | 11.28% |
| Annualized volatility | 10.01% | 11.15% |
| Maximum drawdown | -19.33% | -19.95% |

## Numerical implementation checks

The training-start update passed 38 software tests when it was made; the expanded current suite passes 42. Saved-price readers preserve round-trip precision. The GPD dollar-ES survival integral now evaluates the same closed-form survival function directly instead of repeatedly invoking the general SciPy distribution interface. Six reference tests compare the result with SciPy quadrature across bounded, exponential, near-zero-shape and heavy-tailed cases to an absolute tolerance of 1e-12. This is a calculation-speed change, not a different tail model or a change to test thresholds.
