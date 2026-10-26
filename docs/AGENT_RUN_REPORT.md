# Agent Run Report

## Request
Implement the Quant Edge analytical layers, including ES, EVT, out-of-sample validation, stress testing and risk contributions.

## Work completed
- Implemented data access, cleaning, ADF diagnostics, return estimation, constrained optimization, investment KPIs, VaR/ES, EVT threshold diagnostics, formal OOS backtests, stress tests, risk contributions and exports.
- Created: config, source package, methodology, data governance, living notes, report, tests, app placeholder, and notebook placeholders.
- Dependencies added: only the packages listed in requirements.txt from the requested specification.

## Run evidence
- Price period: 2007-01-03 to 2026-10-22.
- Data quality: 2 missing price cells and 95 flagged extreme returns.
- ADF: 7 of 7 series stationary at 1%.
- Weights: {'SPY': 0.23174078548670018, 'EFA': 0.11838737709252461, 'EEM': 0.013300417436364715, 'AGG': 0.45, 'GLD': 0.15, 'XLE': 0.024825087025660126, 'XLK': 0.011746332958750171}.
- Expected return: 8.0137%; annualized volatility: 7.5736%.
- VaR summary: [{'method': 'historical', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.006984678900430162, 'latest_var_usd': 69846.78900430162, 'latest_es_pct': 0.010617806266118524, 'latest_es_usd': 106178.06266118523, 'average_var_pct': 0.005768874130371565, 'number_of_exceptions': 477, 'observations': 4728}, {'method': 'historical', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.009546362797628632, 'latest_var_usd': 95463.62797628633, 'latest_es_pct': 0.01304822301445321, 'latest_es_usd': 130482.2301445321, 'average_var_pct': 0.00818842703248068, 'number_of_exceptions': 265, 'observations': 4728}, {'method': 'historical', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.01559011933696229, 'latest_var_usd': 155901.1933696229, 'latest_es_pct': 0.01774814321131852, 'latest_es_usd': 177481.4321131852, 'average_var_pct': 0.014226600335450762, 'number_of_exceptions': 77, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.007148445033878947, 'latest_var_usd': 71484.45033878947, 'latest_es_pct': 0.010720638130604667, 'latest_es_usd': 107206.38130604666, 'average_var_pct': 0.005826095486558898, 'number_of_exceptions': 471, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.010267210757006168, 'latest_var_usd': 102672.10757006168, 'latest_es_pct': 0.013079734046683871, 'latest_es_usd': 130797.34046683871, 'average_var_pct': 0.00829815147892869, 'number_of_exceptions': 259, 'observations': 4728}, {'method': 'monte_carlo_bootstrap', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.015998049733315173, 'latest_var_usd': 159980.49733315172, 'latest_es_pct': 0.01782872355812315, 'latest_es_usd': 178287.2355812315, 'average_var_pct': 0.014817694833083515, 'number_of_exceptions': 68, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.9, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.005830988082123963, 'latest_var_usd': 58309.880821239625, 'latest_es_pct': 0.007974719868080848, 'latest_es_usd': 79747.19868080849, 'average_var_pct': 0.006379588190372638, 'number_of_exceptions': 411, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.95, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.007477798009763555, 'latest_var_usd': 74777.98009763555, 'latest_es_pct': 0.009367113805821448, 'latest_es_usd': 93671.13805821448, 'average_var_pct': 0.00817784273916285, 'number_of_exceptions': 259, 'observations': 4728}, {'method': 'parametric_ewma', 'confidence': 0.99, 'latest_date': Timestamp('2026-10-20 00:00:00'), 'latest_var_pct': 0.01055958519832819, 'latest_var_usd': 105595.8519832819, 'latest_es_pct': 0.01208741445962147, 'latest_es_usd': 120874.14459621471, 'average_var_pct': 0.01153893474795337, 'number_of_exceptions': 88, 'observations': 4728}].
- Tests: run `pytest`.

## Assumptions made
- Retrospective KPIs use the latest DGS3MO quote; OOS KPIs use as-of historical quotes.
- All configured tickers are required; absent downloads fail the pipeline.
- Observations with any missing asset return are omitted for portfolio-level calculations.
- Historical, CAPM, and diagonal-shrinkage approaches follow the provided specification.

## Known issues and recommendations
Refresh public data before relying on outputs; inspect flagged outliers, infeasible targets, formal backtests and EVT stability warnings. Analytical layers include ES, EVT, OOS validation, stress tests and risk contributions. Review ANALYTICAL_LAYERS.md and acknowledge that fees, taxes, liquidity and transaction costs are excluded.
