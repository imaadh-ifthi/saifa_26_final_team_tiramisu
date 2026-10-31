"""Check the prospective ledger with synthetic prices, not inspected study outcomes."""

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.prospective_validation.run import report, settle


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_settlement_and_report_reject_changed_forecast(tmp_path):
    root = tmp_path / "records"
    _save(root / "protocol.json", {
        "protocol_id": "synthetic", "weights": {"SPY": 1.0},
        "portfolio_value_usd": 10_000_000, "methods": ["historical"],
        "confidence_levels": [.99]})
    prediction = {
        "protocol_id": "synthetic", "target_date": "2024-01-03",
        "created_utc": "2024-01-02T12:00:00+00:00",
        "information_cutoff": "2024-01-02", "cutoff_prices": {"SPY": 100.0},
        "forecasts": [{"target_date": "2024-01-03", "method": "historical",
                       "confidence": .99, "evaluation_index": 1, "var_log": .01,
                       "var_usd": 99_501.66, "es_usd": 150_000}]}
    forecast_path = root / "forecasts/2024-01-03.json"
    _save(forecast_path, prediction)
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame({"SPY": [100.0, 100.0, 98.0]},
                 index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
                 ).to_csv(prices_path)
    outcome = settle(root, prices_path, pd.Timestamp("2024-01-03").date())
    assert outcome["actual_loss_usd"] == pytest.approx(200_000)
    assert outcome["exceptions"]["historical:0.99"] is True
    assert report(root)["settled_days"] == 1
    assert (root / "reports/backtests.csv").exists()
    with pytest.raises(FileExistsError):
        settle(root, prices_path, pd.Timestamp("2024-01-03").date())
    forecast_path.write_text(forecast_path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed or mismatched forecast"):
        report(root)


def test_settlement_rejects_revised_cutoff(tmp_path):
    root = tmp_path / "records"
    _save(root / "protocol.json", {"protocol_id": "synthetic", "weights": {"SPY": 1.0},
                                    "portfolio_value_usd": 10_000_000})
    _save(root / "forecasts/2024-01-03.json", {
        "protocol_id": "synthetic", "target_date": "2024-01-03",
        "created_utc": "2024-01-02T12:00:00+00:00",
        "information_cutoff": "2024-01-02", "cutoff_prices": {"SPY": 100.0},
        "forecasts": []})
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame({"SPY": [100.0, 101.0, 98.0]},
                 index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
                 ).to_csv(prices_path)
    with pytest.raises(ValueError, match="revised"):
        settle(root, prices_path, pd.Timestamp("2024-01-03").date())
