# Methodology and Limitations

## Portfolio construction
The engine minimizes annualized portfolio variance subject to a configured target return and long-only client constraints. This is preferred to a maximum-Sharpe objective because the mandate prioritizes downside control over return maximization. Expected returns blend annualized historical simple returns with CAPM estimates. The blend reduces dependence on noisy sample means. Covariance is estimated from log returns and shrunk toward its diagonal to reduce unstable correlations.

## Returns and risk
Log returns are used for statistical estimation and VaR. Simple returns are used for portfolio wealth and performance KPIs. VaR forecasts are shifted one observation so date t uses information only through t-1. Bootstrap Monte Carlo resamples the trailing empirical return window and preserves its observed skew and tail behavior, but cannot extrapolate beyond the sample. Historical simulation has the same limitation; EWMA normal VaR depends on its distributional assumption.

> VaR is not a guarantee of maximum loss. It is a probabilistic threshold estimate. Tail events can exceed VaR.

Expected Shortfall, EVT diagnostics, frozen-allocation out-of-sample backtests, stress replay and risk contributions are implemented; see ANALYTICAL_LAYERS.md for definitions and output contracts. The model excludes fees, taxes, liquidity and trading costs. Hypothetical shocks are assumptions rather than forecasts. CAPM and covariance estimates are sample-dependent.

Separate [risk-model development experiments](../experiments/improved_risk/README.md) test volatility-filtered historical, bootstrap and EVT forecasts, including a training-calibrated next-day alert. Their 99% alert variants are not rejected by the recorded Kupiec, independence or joint coverage tests on the already-inspected 2018–2026 development period. This does not validate them independently, and they have not replaced the baseline methods. Filtered historical and bootstrap alert variants have 90% and 95% checks; filtered EVT + alert currently has only a 99% comparison. See [ANALYTICAL_LAYERS.md](ANALYTICAL_LAYERS.md) and the experiment reviews for exact results and remaining limitations.
