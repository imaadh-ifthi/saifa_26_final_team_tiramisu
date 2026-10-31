"""Compare two frozen training windows on middle dates; one arm intentionally looks ahead."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.quantedge.analytics import fit_training_portfolio
from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.config import load_config
from src.quantedge.data import load_fred_csv
from src.quantedge.evt import fit_thresholds, gpd_risk, select_threshold
from src.quantedge.tail import empirical_risk, normal_risk, tail_weights
from .evt_response import calibrate_threshold, detector_path
from .models import causal_volatility


HERE = Path(__file__).resolve().parent
WINDOWS = {"early_2007_2012": ("2007-01-01", "2012-12-31"),
           "future_2021_2026": ("2021-01-01", "2026-12-31")}
TEST_START, TEST_END = "2013-01-01", "2020-12-31"
METHODS = ("historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd",
           "filtered_historical", "filtered_bootstrap", "filtered_evt",
           "filtered_historical_alert", "filtered_bootstrap_alert", "filtered_evt_cusum")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_evt(sample: pd.Series, spec: dict, confidences: list[float]) -> tuple[pd.Series, str]:
    report = fit_thresholds(sample, spec["threshold_quantiles"], spec["min_exceedances"],
                            spec["shape_tolerance"], spec["scale_tolerance"])
    fit, status = select_threshold(report, min(confidences))
    if fit is None:
        raise ValueError("No eligible EVT fit for the frozen training window")
    return fit, status


def _portfolio_returns(simple: pd.DataFrame, weights: pd.Series,
                       start: str, end: str) -> pd.Series:
    sample = simple.loc[start:end, weights.index].dropna()
    if sample.empty or not np.isfinite(sample.to_numpy()).all():
        raise ValueError("Empty or invalid portfolio returns")
    return np.log1p(sample @ weights)


def _empirical_cache(sample: np.ndarray, confidences: list[float]) -> dict:
    """Cache quantiles and nonzero tail masses for exact scaled empirical ES."""
    cache = {}
    for c in confidences:
        weights = tail_weights(-sample, c)
        used = weights > 0
        cache[c] = (max(0.0, -float(np.quantile(sample, 1-c))),
                    sample[used], weights[used])
    return cache


def _scaled_empirical(cache: dict, confidence: float, scale: float) -> dict:
    var_base, tail_sample, weights = cache[confidence]
    var_log = var_base*scale
    return {"var_log": var_log,
            "es_log": max(0.0, float(weights @ (-tail_sample*scale))),
            "var_simple_pct": max(0.0, float(-np.expm1(-var_log))),
            "es_simple_pct": max(0.0, float(weights @ -np.expm1(tail_sample*scale)))}


def evaluate_window(train: pd.Series, test: pd.Series, label: str,
                    base: dict, experiment: dict, alert: dict) -> tuple[pd.DataFrame, dict]:
    """Fit tail shapes once; use observed earlier test returns for next-day volatility/alerts."""
    if train.empty or test.empty or train.index.intersection(test.index).size:
        raise ValueError("Training and test windows must be nonempty and disjoint")
    if not test.index.is_monotonic_increasing or not train.index.is_monotonic_increasing:
        raise ValueError("Window dates must be sorted")
    h = int(base["var"]["historical_window"])
    mc = int(base["var"]["monte_carlo_window"])
    fh = int(experiment["historical_window"])
    ew = int(experiment["evt"]["window"])
    if len(train) < max(h, mc, fh+alert["volatility"]["min_periods"],
                        ew+alert["volatility"]["min_periods"]):
        raise ValueError("Training window is too short for all fixed model warmups")
    confidences = base["var"]["confidence_levels"]
    value = float(base["project"]["portfolio_value_usd"])
    seed = int(base["project"]["seed"])
    # The RangeIndex is an artificial sequence in the future-trained arm. This is
    # deliberate look-ahead, not a time-respecting validation scheme.
    sequence = pd.Series(np.r_[train.to_numpy(), test.to_numpy()], dtype=float)
    n = len(train)
    vcfg = alert["volatility"]
    slow = causal_volatility(sequence, vcfg["reference_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    fast = causal_volatility(sequence, vcfg["fast_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    threshold, training_triggers, training_days = calibrate_threshold(
        sequence.iloc[:n], slow.iloc[:n], n-1,
        alert["detector"])
    signals = detector_path(sequence, slow, threshold,
                            alert["detector"]["standardized_loss_drift"],
                            alert["detector"]["active_forecast_days"])
    raw_hist = train.iloc[-h:].to_numpy()
    raw_boot = np.random.default_rng(seed).choice(train.iloc[-mc:].to_numpy(),
                                                   size=int(base["var"]["monte_carlo_simulations"]),
                                                   replace=True)
    raw_fit, raw_status = _fit_evt(train.iloc[-int(base["evt"]["window"]):],
                                   base["evt"], confidences)
    residuals = sequence.iloc[:n] / slow.iloc[:n]
    filtered_hist = residuals.iloc[-fh:].to_numpy()
    filtered_boot = np.random.default_rng(seed+1).choice(
        filtered_hist, size=int(experiment["monte_carlo_simulations"]), replace=True)
    filtered_fit, filtered_status = _fit_evt(residuals.iloc[-ew:],
                                             experiment["evt"], confidences)
    if not np.isfinite(filtered_hist).all():
        raise ValueError("Nonfinite standardized training shocks")
    raw_evt = {c: gpd_risk(float(raw_fit.threshold_log), float(raw_fit["shape"]),
                           float(raw_fit.scale), float(raw_fit.exceedance_probability), c)
               for c in confidences}
    raw_historical = {c: empirical_risk(raw_hist, c) for c in confidences}
    raw_bootstrap = {c: empirical_risk(raw_boot, c) for c in confidences}
    filtered_historical_cache = _empirical_cache(filtered_hist, confidences)
    filtered_bootstrap_cache = _empirical_cache(filtered_boot, confidences)
    rows = []
    for j, target_date in enumerate(test.index):
        i = n+j
        sigma = float(slow.iloc[i])
        alert_active = bool(signals.iloc[i].alert_active_for_forecast)
        alert_sigma = max(sigma, float(fast.iloc[i])) if alert_active else sigma
        for c in confidences:
            estimates = {
                "historical": raw_historical[c],
                "parametric_ewma": normal_risk(sigma, c),
                "monte_carlo_bootstrap": raw_bootstrap[c],
                "evt_gpd": raw_evt[c],
                "filtered_historical": _scaled_empirical(filtered_historical_cache, c, sigma),
                "filtered_bootstrap": _scaled_empirical(filtered_bootstrap_cache, c, sigma),
                "filtered_evt": gpd_risk(float(filtered_fit.threshold_log)*sigma,
                                         float(filtered_fit["shape"]), float(filtered_fit.scale)*sigma,
                                         float(filtered_fit.exceedance_probability), c),
                "filtered_historical_alert": _scaled_empirical(filtered_historical_cache, c, alert_sigma),
                "filtered_bootstrap_alert": _scaled_empirical(filtered_bootstrap_cache, c, alert_sigma),
                "filtered_evt_cusum": gpd_risk(float(filtered_fit.threshold_log)*alert_sigma,
                                                float(filtered_fit["shape"]),
                                                float(filtered_fit.scale)*alert_sigma,
                                                float(filtered_fit.exceedance_probability), c),
            }
            actual = float(test.iloc[j])
            for method, risk in estimates.items():
                rows.append({"training_window": label, "date": target_date,
                             "method": method, "confidence": c, "evaluation_index": j,
                             "actual_log_return": actual,
                             "actual_loss_usd": -np.expm1(actual)*value,
                             "var_log": risk["var_log"], "es_log": risk["es_log"],
                             "var_usd": risk["var_simple_pct"]*value,
                             "es_usd": risk["es_simple_pct"]*value,
                             "exception": actual < -risk["var_log"],
                             "alert_active": alert_active})
    frame = pd.DataFrame(rows)
    if (set(frame.method) != set(METHODS) or frame.date.nunique() != len(test)
            or len(frame) != len(test)*len(METHODS)*len(confidences)):
        raise ValueError("Incomplete backward-window forecasts")
    return frame, {"training_window": label, "training_start": str(train.index.min().date()),
                   "training_end": str(train.index.max().date()), "training_days": n,
                   "test_start": str(test.index.min().date()),
                   "test_end": str(test.index.max().date()), "test_days": len(test),
                   "detector_threshold": threshold, "training_triggers": training_triggers,
                   "training_volatility_days": training_days,
                   "raw_evt_status": raw_status, "filtered_evt_status": filtered_status,
                   "alert_active_test_days": int(signals.iloc[n:].alert_active_for_forecast.sum())}


def run(prices_path: Path, rates_path: Path, output: Path) -> dict:
    base = load_config("config.yaml")
    experiment = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    alert = yaml.safe_load((HERE / "evt_response.yaml").read_text(encoding="utf-8"))
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True, float_precision="round_trip")
    _, simple = compute_returns(prices)
    simple = simple.loc[:, base["data"]["tickers"]].dropna()
    rates = load_fred_csv(rates_path, base["data"]["fred_risk_free_series"])
    early_config = json.loads(json.dumps(base))
    early_config["validation"].update(train_end=WINDOWS["early_2007_2012"][1],
                                       test_start=TEST_START)
    weights, allocation = fit_training_portfolio(simple, rates, early_config)
    test = _portfolio_returns(simple, weights, TEST_START, TEST_END)
    all_rows, metadata = [], []
    for label, (start, end) in WINDOWS.items():
        train = _portfolio_returns(simple, weights, start, end)
        frame, meta = evaluate_window(train, test, label, base, experiment, alert)
        all_rows.append(frame)
        metadata.append(meta)
    forecasts = pd.concat(all_rows, ignore_index=True)
    tests = []
    for label, group in forecasts.groupby("training_window", sort=False):
        result = backtest_forecasts(group, base["validation"]["significance"])
        result.insert(0, "training_window", label)
        tests.append(result)
    backtests = pd.concat(tests, ignore_index=True)
    loss = -forecasts.actual_log_return
    forecasts["quantile_loss"] = (forecasts.confidence -
                                  (loss < forecasts.var_log).astype(float))*(loss-forecasts.var_log)
    scores = forecasts.groupby(["training_window", "method", "confidence"], as_index=False).agg(
        mean_quantile_loss=("quantile_loss", "mean"),
        mean_var_usd=("var_usd", "mean"), mean_es_usd=("es_usd", "mean"))
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "daily_forecasts.csv.gz", index=False)
    backtests.to_csv(output / "backtests.csv", index=False)
    scores.to_csv(output / "scores.csv", index=False)
    manifest = {"experiment": "retrospective_backward_window_v1",
                "warning": "2021-2026 training uses future information relative to 2013-2020 scoring; neither arm is an untouched holdout",
                "portfolio": allocation, "windows": metadata,
                "tail_fit_policy": "fit_once_per_training_block; update_only_volatility_and_alert_state_with_prior_test_returns",
                "price_sha256": _hash(prices_path), "rates_sha256": _hash(rates_path),
                "script_sha256": _hash(Path(__file__)),
                "confidence_levels": base["var"]["confidence_levels"], "methods": METHODS}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    merged = backtests.merge(scores, on=["training_window", "method", "confidence"])
    lines = ["# Backward-window robustness comparison", "",
             "**The 2021–2026 arm looks ahead to future events when scoring 2013–2020. "
             "These are descriptive, retrospective results, not valid out-of-sample evidence. "
             "Both windows and the test years have already been inspected during model development.**", "",
             "The portfolio weights were estimated once from 2007–2012 and held identical in both arms. "
             "Each arm fits its historical shocks and EVT tail once on its named training block. "
             "Only volatility and the alert state update from preceding test-day returns. "
             "This differs from the existing rolling-refit models; compare patterns, not their headline p-values.", "",
             "| Training window | Model | 99% breaches / days | Kupiec p | Independence p | Joint p | Mean quantile loss | Mean VaR USD |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in merged[merged.confidence.eq(.99)].itertuples():
        fmt = lambda x: f"{x:.4f}" if pd.notna(x) else "unavailable"
        lines.append(f"| {row.training_window} | {row.method} | {row.exceptions}/{row.observations} | "
                     f"{fmt(row.kupiec_p_value)} | {fmt(row.independence_p_value)} | "
                     f"{fmt(row.conditional_coverage_p_value)} | "
                     f"{row.mean_quantile_loss:.6g} | {row.mean_var_usd:,.0f} |")
    lines += ["", "All 90%, 95%, and 99% results are in `backtests.csv` and `scores.csv`. "
              "P-values below 0.05 reject their null hypotheses. Non-rejection does not validate a model. "
              "The Basel traffic light is included in the CSV. VaR backtests do not validate ES.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output), "test_days": len(test), "methods": len(METHODS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("outputs/tables/prices.csv"))
    parser.add_argument("--risk-free", type=Path, default=Path("data/raw/risk_free.csv"))
    parser.add_argument("--output", type=Path, default=HERE / "results/backward_window")
    args = parser.parse_args()
    print(json.dumps(run(args.prices, args.risk_free, args.output), indent=2))


if __name__ == "__main__":
    main()
