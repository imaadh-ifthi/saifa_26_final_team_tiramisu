# Data Governance

Yahoo Finance adjusted close is the sole price input. The explicit `Adj Close` field is selected with automatic adjustment disabled. FRED DGS3MO supplies the public short-rate series. Raw source downloads are cached under `data/raw/` when enabled. Calendars are not forward-filled. Missing values are retained in source exports and reported; complete rows are used for portfolio returns. Potential outliers are flagged using an eight-MAD threshold and are not silently removed. ADF diagnostics are written to `outputs/tables/adf_report.csv`.

Latest downloaded price date: 2026-10-22. Assets with missing prices: 2 asset-days. Potential extreme-return flags: 95. Treat downloads as third-party source data and verify them before publication.
