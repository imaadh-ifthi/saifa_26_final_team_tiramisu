"""Freeze, forecast, settle, and report a genuinely forward risk-model evaluation."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.config import load_config
from src.quantedge.risk import build_risk_forecasts
from src.quantedge.tail import empirical_risk
from experiments.improved_risk.evt_response import calibrate_threshold, detector_path
from experiments.improved_risk.models import (build_improved_forecasts, causal_volatility,
                                              filtered_gpd_risk)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_FILES = (
    "config.yaml", "src/quantedge/analytics.py", "src/quantedge/backtesting.py",
    "src/quantedge/cleaning.py", "src/quantedge/config.py", "src/quantedge/evt.py",
    "src/quantedge/risk.py", "src/quantedge/tail.py",
    "experiments/improved_risk/config.yaml", "experiments/improved_risk/models.py",
    "experiments/improved_risk/evt_response.yaml",
    "experiments/improved_risk/evt_response.py",
    "experiments/prospective_validation/run.py",
)
BASELINE_METHODS = ("historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd")
REFERENCE_METHODS = ("filtered_historical", "filtered_bootstrap", "filtered_evt")
ALERT_METHODS = ("filtered_historical_alert", "filtered_bootstrap_alert", "filtered_evt_cusum")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(now: datetime | None = None) -> datetime:
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("Clock must be timezone-aware")
    return instant.astimezone(timezone.utc)


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_hashes() -> dict[str, str]:
    return {name: _hash(REPO / name) for name in MODEL_FILES}


def _load_prices(path: Path, tickers: list[str], *, trim_incomplete_tail: bool = False) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
    if (prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing
            or not set(tickers).issubset(prices.columns)):
        raise ValueError("Price snapshot needs sorted unique dates and all frozen assets")
    prices = prices.loc[:, tickers]
    if trim_incomplete_tail:
        complete = prices.notna().all(axis=1) & prices.shift(1).notna().all(axis=1)
        if not complete.any():
            raise ValueError("No complete portfolio return exists")
        prices = prices.loc[:complete[complete].index[-1]]
    if (prices.iloc[-1].isna().any() or not np.isfinite(prices.iloc[-1].to_numpy(dtype=float)).all()
            or (prices.iloc[-1] <= 0).any()):
        raise ValueError("Latest price row is incomplete; refresh all frozen assets before forecasting")
    return prices


def _returns(prices: pd.DataFrame, weights: pd.Series, start: str) -> pd.Series:
    _, simple = compute_returns(prices)
    complete = simple.loc[:, weights.index].dropna()
    result = np.log1p(complete.loc[pd.Timestamp(start):] @ weights)
    if result.empty or result.index.max() != prices.index.max():
        raise ValueError("Latest price row has no complete portfolio return")
    return result


def freeze(root: Path, prices_path: Path, study_manifest_path: Path,
           now: datetime | None = None) -> dict:
    """Create a versioned protocol. Refuse to replace an existing freeze."""
    if (root / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; create a new directory for a new version")
    base = load_config(str(REPO / "config.yaml"))
    experiment = yaml.safe_load((REPO / "experiments/improved_risk/config.yaml").read_text(encoding="utf-8"))
    alert = yaml.safe_load((REPO / "experiments/improved_risk/evt_response.yaml").read_text(encoding="utf-8"))
    study = _read(study_manifest_path)
    if study.get("status") != "complete" or study["base_configuration"] != base:
        raise ValueError("The completed study manifest must match the current base configuration")
    if study["experiment_configuration"] != experiment:
        raise ValueError("The study manifest and experimental configuration differ")
    weights = pd.Series(study["training"]["weights"], dtype=float).reindex(base["data"]["tickers"])
    if weights.isna().any() or abs(weights.sum()-1) > 1e-8:
        raise ValueError("Frozen training allocation is invalid")
    prices = _load_prices(prices_path, base["data"]["tickers"], trim_incomplete_tail=True)
    returns = _returns(prices, weights, base["validation"]["train_start"])
    vol = causal_volatility(returns, alert["volatility"]["reference_ewma_lambda"],
                            alert["volatility"]["min_periods"], alert["volatility"]["floor"])
    threshold, training_triggers, training_days = calibrate_threshold(
        returns, vol, pd.Timestamp(base["validation"]["train_end"]), alert["detector"])
    hashes = _model_hashes()
    identity = json.dumps({"hashes": hashes, "weights": weights.to_dict(),
                           "threshold": threshold}, sort_keys=True).encode("utf-8")
    protocol = {"protocol_id": hashlib.sha256(identity).hexdigest()[:16],
                "created_utc": _utc(now).isoformat(),
                "purpose": "prospective_same_frozen_portfolio_daily_constant_weights_no_costs",
                "portfolio_value_usd": float(base["project"]["portfolio_value_usd"]),
                "weights": weights.to_dict(),
                "training_start": base["validation"]["train_start"],
                "training_end": base["validation"]["train_end"],
                "evaluation_start": base["validation"]["test_start"],
                "confidence_levels": base["var"]["confidence_levels"],
                "methods": list(BASELINE_METHODS + REFERENCE_METHODS + ALERT_METHODS),
                "detector_threshold": threshold,
                "training_trigger_count": training_triggers,
                "training_volatility_days": training_days,
                "model_sha256": hashes,
                "starting_price_cutoff": str(prices.index.max().date()),
                "starting_price_sha256": _hash(prices_path),
                "study_manifest_sha256": _hash(study_manifest_path)}
    _write_once(root / "protocol.json", protocol)
    return protocol


def _protocol(root: Path, *, check_code: bool) -> dict:
    result = _read(root / "protocol.json")
    if check_code and _model_hashes() != result["model_sha256"]:
        raise ValueError("Model code or configuration changed after freeze; start a new protocol version")
    return result


def _risk_record(date_value: date, method: str, confidence: float, risk: dict,
                 value: float, evaluation_index: int) -> dict:
    return {"target_date": date_value.isoformat(), "method": method,
            "confidence": float(confidence), "evaluation_index": int(evaluation_index),
            "var_log": float(risk["var_log"]), "es_log": float(risk["es_log"]),
            "var_usd": float(risk["var_simple_pct"] * value),
            "es_usd": float(risk["es_simple_pct"] * value)}


def _forecast_rows(returns: pd.Series, target: pd.Timestamp, protocol: dict) -> tuple[list[dict], dict]:
    base = load_config(str(REPO / "config.yaml"))
    exp = yaml.safe_load((REPO / "experiments/improved_risk/config.yaml").read_text(encoding="utf-8"))
    alert_spec = yaml.safe_load((REPO / "experiments/improved_risk/evt_response.yaml").read_text(encoding="utf-8"))
    if (base["var"]["confidence_levels"] != protocol["confidence_levels"] or
            exp["filter"]["ewma_lambda"] != alert_spec["volatility"]["reference_ewma_lambda"]):
        raise ValueError("Model settings do not match the frozen protocol")
    if target <= returns.index.max() or target in returns.index:
        raise ValueError("Target must follow all observed returns")
    # Both forecasters create a row for each return date. The appended value is
    # only a placeholder; causal tests verify target forecasts use earlier data.
    extended = pd.concat([returns, pd.Series([0.0], index=pd.DatetimeIndex([target]))])
    evaluation_start = pd.Timestamp(protocol["evaluation_start"])
    baseline, _ = build_risk_forecasts(extended, base, evaluation_start, include_evt=True)
    improved, diagnostics = build_improved_forecasts(extended, base, exp, evaluation_start)
    baseline = baseline[baseline.date.eq(target) & baseline.method.isin(BASELINE_METHODS)]
    improved = improved[improved.date.eq(target) & improved.method.isin(REFERENCE_METHODS)]
    expected = len(protocol["confidence_levels"])
    if len(baseline) != 4*expected or len(improved) != 3*expected:
        raise ValueError("Missing one-day forecast for a frozen method/confidence")
    if (baseline.estimation_end >= target).any() or (improved.estimation_end >= target).any():
        raise ValueError("Forecast information cutoff is not strictly before target")
    vcfg = alert_spec["volatility"]
    slow = causal_volatility(extended, vcfg["reference_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    fast = causal_volatility(extended, vcfg["fast_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    signal = detector_path(returns, slow.loc[returns.index], protocol["detector_threshold"],
                           alert_spec["detector"]["standardized_loss_drift"],
                           alert_spec["detector"]["active_forecast_days"])
    last_signal = signal.iloc[-1]
    active = bool(last_signal.active_through_index >= len(returns))
    slow_today, fast_today = float(slow.loc[target]), float(fast.loc[target])
    overlay_vol = max(slow_today, fast_today) if active else slow_today
    ratio = overlay_vol / slow_today
    residuals = extended / slow
    i = len(returns)
    window = int(exp["historical_window"])
    trailing = residuals.iloc[i-window:i].to_numpy() * slow_today
    simulated = np.random.default_rng(base["project"]["seed"]+i).choice(
        trailing, size=int(exp["monte_carlo_simulations"]), replace=True)
    if not np.isfinite(trailing).all():
        raise ValueError("Nonfinite standardized shocks before target")
    fits = diagnostics.loc[diagnostics.model.eq("filtered_evt") &
                           diagnostics.selected.eq(True)].sort_values("fit_date")
    if fits.empty or pd.Timestamp(fits.iloc[-1].estimation_end) >= target:
        raise ValueError("Missing causal filtered EVT tail fit")
    fit = fits.iloc[-1].copy()
    fit["threshold_log"] = fit["threshold_standardized"]
    value = float(protocol["portfolio_value_usd"])
    rows = []
    for source in (baseline, improved):
        for item in source.itertuples():
            risk = {"var_log": item.var_log, "es_log": item.es_log,
                    "var_simple_pct": item.var_simple_pct,
                    "es_simple_pct": item.es_simple_pct}
            rows.append(_risk_record(target.date(), item.method, item.confidence,
                                     risk, value, i))
    for confidence in protocol["confidence_levels"]:
        for method, sample in (("filtered_historical_alert", trailing),
                               ("filtered_bootstrap_alert", simulated)):
            updated = empirical_risk(sample * ratio, confidence)
            rows.append(_risk_record(target.date(), method, confidence, updated, value, i))
        evt = filtered_gpd_risk(fit, overlay_vol, confidence)
        rows.append(_risk_record(target.date(), "filtered_evt_cusum", confidence,
                                 evt, value, i))
    if (len(rows) != len(protocol["methods"])*expected or
            {(row["method"], row["confidence"]) for row in rows} !=
            {(method, float(c)) for method in protocol["methods"]
             for c in protocol["confidence_levels"]}):
        raise ValueError("Forecast rows do not match frozen model set")
    if any(not np.isfinite([row["var_log"], row["var_usd"], row["es_usd"]]).all()
           or row["es_usd"] + 1e-6 < row["var_usd"] for row in rows):
        raise ValueError("Invalid VaR/ES forecast")
    details = {"alert_active_for_target": active,
               "last_trigger_close": str(signal.loc[signal.trigger_after_close, "date"].max().date())
               if signal.trigger_after_close.any() else None,
               "cusum_score_after_cutoff_close": float(max(0.0, last_signal.score_before_close +
                   last_signal.standardized_loss_after_close -
                   alert_spec["detector"]["standardized_loss_drift"]))
               if not last_signal.trigger_after_close else 0.0,
               "slow_volatility": slow_today, "fast_volatility": fast_today,
               "response_ratio": ratio}
    return sorted(rows, key=lambda row: (row["method"], row["confidence"])), details


def forecast(root: Path, prices_path: Path, target: date,
             now: datetime | None = None) -> dict:
    protocol = _protocol(root, check_code=True)
    instant = _utc(now)
    if target <= instant.date():
        raise ValueError("Prospective forecasts require a future UTC calendar date; backdating is refused")
    path = root / "forecasts" / f"{target.isoformat()}.json"
    if path.exists():
        raise FileExistsError("Forecast for this date is already recorded")
    weights = pd.Series(protocol["weights"], dtype=float)
    prices = _load_prices(prices_path, list(weights.index))
    cutoff = prices.index.max()
    if target <= cutoff.date():
        raise ValueError("Target date is already in the price snapshot")
    if (pd.Timestamp(target) - cutoff).days > 7:
        raise ValueError("Price snapshot is too stale for a one-day forecast")
    returns = _returns(prices, weights, protocol["training_start"])
    rows, details = _forecast_rows(returns, pd.Timestamp(target), protocol)
    result = {"protocol_id": protocol["protocol_id"],
              "created_utc": instant.isoformat(),
              "target_date": target.isoformat(),
              "information_cutoff": str(cutoff.date()),
              "price_input_sha256": _hash(prices_path),
              "cutoff_prices": {ticker: float(prices.loc[cutoff, ticker]) for ticker in weights.index},
              "weights": protocol["weights"],
              "portfolio_value_usd": protocol["portfolio_value_usd"],
              "model_sha256": protocol["model_sha256"],
              "alert": details, "forecasts": rows}
    _write_once(path, result)
    return result


def settle(root: Path, prices_path: Path, target: date,
           now: datetime | None = None) -> dict:
    protocol = _protocol(root, check_code=False)
    if target > _utc(now).date():
        raise ValueError("Cannot settle a future date")
    forecast_path = root / "forecasts" / f"{target.isoformat()}.json"
    outcome_path = root / "outcomes" / f"{target.isoformat()}.json"
    if outcome_path.exists():
        raise FileExistsError("Outcome for this target is already recorded")
    prediction = _read(forecast_path)
    if prediction["protocol_id"] != protocol["protocol_id"] or prediction["target_date"] != target.isoformat():
        raise ValueError("Forecast does not belong to this frozen protocol/date")
    if datetime.fromisoformat(prediction["created_utc"]).astimezone(timezone.utc).date() >= target:
        raise ValueError("Forecast timestamp is not before the target date")
    tickers = list(protocol["weights"])
    prices = _load_prices(prices_path, tickers)
    cutoff = pd.Timestamp(prediction["information_cutoff"])
    target_ts = pd.Timestamp(target)
    if cutoff not in prices.index or target_ts not in prices.index:
        raise ValueError("Price snapshot is missing cutoff or target prices")
    old = np.array([prediction["cutoff_prices"][ticker] for ticker in tickers])
    if not np.allclose(prices.loc[cutoff, tickers].to_numpy(dtype=float), old,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Cutoff adjusted prices were revised; review data before settlement")
    _, simple = compute_returns(prices)
    complete = simple.loc[:, tickers].dropna()
    newer = complete.loc[complete.index > cutoff]
    if newer.empty or newer.index[0] != target_ts:
        raise ValueError("Target is not the first complete portfolio return after forecast cutoff")
    weights = pd.Series(protocol["weights"], dtype=float).reindex(tickers)
    actual_simple = float(newer.loc[target_ts] @ weights)
    actual_log = float(np.log1p(actual_simple))
    value = float(protocol["portfolio_value_usd"])
    outcome = {"protocol_id": protocol["protocol_id"],
               "recorded_utc": _utc(now).isoformat(),
               "target_date": target.isoformat(),
               "forecast_sha256": _hash(forecast_path),
               "outcome_price_sha256": _hash(prices_path),
               "target_prices": {ticker: float(prices.loc[target_ts, ticker]) for ticker in tickers},
               "actual_simple_return": actual_simple, "actual_log_return": actual_log,
               "actual_loss_usd": -actual_simple*value,
               "exceptions": {f'{row["method"]}:{row["confidence"]:.2f}':
                              bool(actual_log < -row["var_log"])
                              for row in prediction["forecasts"]}}
    _write_once(outcome_path, outcome)
    return outcome


def report(root: Path) -> dict:
    protocol = _protocol(root, check_code=False)
    forecast_paths = sorted((root / "forecasts").glob("*.json"))
    outcome_paths = sorted((root / "outcomes").glob("*.json"))
    pending = sorted(set(path.stem for path in forecast_paths) -
                     set(path.stem for path in outcome_paths))
    if set(path.stem for path in outcome_paths) - set(path.stem for path in forecast_paths):
        raise ValueError("Outcome exists without a forecast")
    rows = []
    for outcome_path in outcome_paths:
        prediction_path = root / "forecasts" / outcome_path.name
        prediction = _read(prediction_path)
        outcome = _read(outcome_path)
        if (outcome["protocol_id"] != protocol["protocol_id"] or
                outcome["forecast_sha256"] != _hash(prediction_path)):
            raise ValueError(f"Changed or mismatched forecast: {prediction_path}")
        for item in prediction["forecasts"]:
            rows.append({"date": pd.Timestamp(item["target_date"]),
                         "method": item["method"], "confidence": item["confidence"],
                         "evaluation_index": item["evaluation_index"],
                         "var_log": item["var_log"], "var_usd": item["var_usd"],
                         "es_usd": item["es_usd"],
                         "actual_log_return": outcome["actual_log_return"],
                         "actual_loss_usd": outcome["actual_loss_usd"],
                         "exception": bool(outcome["actual_log_return"] < -item["var_log"])})
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        result = {"settled_days": 0, "pending_dates": pending,
                  "status": "awaiting_future_outcomes"}
        (out / "summary.md").write_text(
            "# Prospective evaluation\n\nNo future forecasts have been settled yet. "
            "This is a forward-test protocol, not historical backtest evidence.\n\n"
            f"Pending dates: {', '.join(pending) if pending else 'none'}.\n",
            encoding="utf-8")
        return result
    frame = pd.DataFrame(rows).sort_values(["date", "method", "confidence"])
    if frame.duplicated(["date", "method", "confidence"]).any():
        raise ValueError("Duplicate scored forecasts")
    dates = frame.date.drop_duplicates().sort_values()
    if len(frame) != len(dates)*len(protocol["methods"])*len(protocol["confidence_levels"]):
        raise ValueError("Incomplete method/confidence records")
    # The stored evaluation indices must advance by one complete portfolio day.
    # backtest_forecasts independently checks this within every method/confidence.
    backtests = backtest_forecasts(frame)
    loss = -frame.actual_log_return
    frame["quantile_loss"] = (frame.confidence - (loss < frame.var_log).astype(float)) * (
        loss - frame.var_log)
    scores = frame.groupby(["method", "confidence"], as_index=False).agg(
        mean_quantile_loss=("quantile_loss", "mean"),
        mean_var_usd=("var_usd", "mean"),
        mean_es_usd=("es_usd", "mean"))
    excess = frame[frame.exception].assign(excess_usd=lambda x: x.actual_loss_usd-x.var_usd)
    severity = excess.groupby(["method", "confidence"], as_index=False).excess_usd.mean()
    scores = scores.merge(severity.rename(columns={"excess_usd": "mean_breach_excess_usd"}),
                          on=["method", "confidence"], how="left")
    backtests["evaluation_status"] = np.where(backtests.observations < 250,
                                              "early_monitoring_low_power", "review_sample_size")
    frame.to_csv(out / "scored_forecasts.csv", index=False)
    backtests.to_csv(out / "backtests.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    lines = ["# Prospective evaluation", "",
             f"Frozen protocol `{protocol['protocol_id']}`; {len(dates)} settled days from "
             f"{dates.iloc[0].date()} through {dates.iloc[-1].date()}.",
             f"Pending forecast dates: {', '.join(pending) if pending else 'none'}.", "",
             "| Method | Confidence | Breaches / days | Kupiec p | Independence p | Joint p | Mean quantile loss | Mean VaR USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    combined = backtests.merge(scores, on=["method", "confidence"])
    for item in combined.itertuples():
        fmt = lambda value: f"{value:.4g}" if pd.notna(value) else "unavailable"
        lines.append(f"| {item.method} | {item.confidence:.0%} | {item.exceptions}/{item.observations} | "
                     f"{fmt(item.kupiec_p_value)} | {fmt(item.independence_p_value)} | "
                     f"{fmt(item.conditional_coverage_p_value)} | "
                     f"{item.mean_quantile_loss:.6g} | {item.mean_var_usd:,.0f} |")
    lines += ["", "All forecasts were stored before their target dates and scored against the same frozen daily-rebalanced $10 million portfolio. P-values on fewer than 250 days are early monitoring statistics with very low power; non-rejection does not validate a model. At 99%, even 250 days imply only about 2.5 expected breaches. VaR backtests do not validate ES. The scored CSV includes every recorded date and forecast; missing days are errors, not silently omitted.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"settled_days": len(dates), "pending_dates": pending,
            "status": "early_monitoring" if len(dates) < 250 else "review_results"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "records")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--prices", type=Path, default=REPO / "outputs/tables/prices.csv")
    freeze_parser.add_argument("--study-manifest", type=Path,
                               default=REPO / "experiments/improved_risk/results/manifest.json")
    forecast_parser = sub.add_parser("forecast")
    forecast_parser.add_argument("--prices", type=Path, required=True)
    forecast_parser.add_argument("--target-date", type=date.fromisoformat, required=True)
    settle_parser = sub.add_parser("settle")
    settle_parser.add_argument("--prices", type=Path, required=True)
    settle_parser.add_argument("--target-date", type=date.fromisoformat, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.root, args.prices, args.study_manifest)
    elif args.command == "forecast":
        result = forecast(args.root, args.prices, args.target_date)
    elif args.command == "settle":
        result = settle(args.root, args.prices, args.target_date)
    else:
        result = report(args.root)
    print(json.dumps({"command": args.command,
                      "protocol_id": result.get("protocol_id"),
                      "target_date": result.get("target_date"),
                      "status": result.get("status"),
                      "settled_days": result.get("settled_days"),
                      "pending_dates": result.get("pending_dates")}, indent=2))


if __name__ == "__main__":
    main()
