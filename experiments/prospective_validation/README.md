# Forward evaluation of the frozen portfolio

This separate workflow tests the **same 2007–2017 training-only allocation** used in the historical risk comparison. It records forecasts before future outcomes, then scores them after the next complete seven-ETF price row arrives. The current full-history allocation is different and is not used here.

The protocol includes the four baseline models, three filtered references, and three next-day alert models at 90%, 95%, and 99% confidence. Its code, configuration, allocation, alert threshold, and starting price source are hashed in `records/protocol.json`. A changed model or configuration requires a new records directory. Forecast and outcome files are create-once; keep an external timestamped copy or Git commit of each forecast if independent proof of timing matters. Local files and their timestamps alone are not tamper-proof.

Run from the repository root:

```powershell
python -m experiments.prospective_validation.run freeze
python -m experiments.prospective_validation.run forecast --prices path/to/fresh_prices.csv --target-date YYYY-MM-DD
python -m experiments.prospective_validation.run settle --prices path/to/prices_with_target.csv --target-date YYYY-MM-DD
python -m experiments.prospective_validation.run report
```

Use `--root path/to/versioned_records` on every command for a separate protocol version. The default root is `experiments/prospective_validation/records`. The price CSV must have a date index and columns `SPY,EFA,EEM,AGG,GLD,XLE,XLK` containing the same adjusted-close series as the original price file. Forecasting requires a complete latest row and rejects target dates already present in the data, stale snapshots, duplicate dates, or changes to frozen model files. For the first run, use the first future trading day after the latest complete portfolio return. Settlement rejects a missing target return, changed cutoff prices, and missing intervening portfolio returns. Record the forecast **before** the target-day outcome exists; after the target closes and adjusted prices are available, settle it. Continue every complete trading day. Do not use a historical date to fill a missed forecast.

`report` writes `reports/scored_forecasts.csv` (each day's forecasts and outcomes), `reports/backtests.csv` (Kupiec, Christoffersen independence, joint coverage, and the 99% Basel traffic light), `reports/scores.csv` (quantile loss, mean VaR/ES, breach severity), and `reports/summary.md`. Run it at any point to see pending forecasts. A missing day is an error rather than an omitted observation. The report compares all models on the same dates. Early p-values have very low power; a non-rejection is not proof of calibration. Quantile loss compares forecast quality, while mean VaR shows conservatism. There is no ES-specific backtest.

**Current status:** this script creates a protocol, but no genuine future forecast/outcome pair exists yet. The saved price file's latest complete portfolio return is 2026-10-20; later rows are incomplete. Refresh all seven adjusted prices before making a prospective forecast. Historical 2018–2026 results remain development evidence and are not copied into this forward report.
