"""Apply the existing training-calibrated next-day alert to filtered empirical VaR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.tail import empirical_risk
from .evt_response import calibrate_threshold, detector_path
from .models import causal_volatility


HERE = Path(__file__).resolve().parent


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_filtered_response(saved: pd.DataFrame, returns: pd.Series,
                            sigma_slow: pd.Series, sigma_fast: pd.Series,
                            alert_path: pd.DataFrame, value: float,
                            seed: int, window: int, simulations: int,
                            models: list[str], confidences: list[float]) -> pd.DataFrame:
    """Recompute empirical tails on alert dates; copy and verify other dates."""
    reference = saved[saved.method.isin(models) & saved.confidence.isin(confidences)].copy()
    reference["date"] = pd.to_datetime(reference.date)
    if reference.duplicated(["date", "method", "confidence"]).any():
        raise ValueError("Duplicate reference forecasts")
    expected_dates = returns.loc[reference.date.min():].index
    for _, group in reference.groupby(["method", "confidence"]):
        if not group.sort_values("date").date.reset_index(drop=True).equals(
                pd.Series(expected_dates).reset_index(drop=True)):
            raise ValueError("Reference model has missing or mismatched dates")
    alert = alert_path.set_index("date")
    sigma_at_dates = sigma_slow.loc[reference.date].to_numpy()
    if not np.allclose(reference.forecast_volatility.to_numpy(), sigma_at_dates,
                       rtol=1e-10, atol=1e-12):
        raise ValueError("Reconstructed EWMA volatility differs from saved reference")
    reference["alert_active_for_forecast"] = reference.date.map(alert.alert_active_for_forecast).astype(bool)
    reference["alert_trigger_after_close"] = reference.date.map(alert.trigger_after_close).astype(bool)
    reference["forecast_volatility_fast"] = reference.date.map(sigma_fast)
    reference["forecast_volatility_slow"] = reference.date.map(sigma_slow)
    reference["response_ratio"] = 1.0
    reference["variant"] = "reference"
    candidate = reference.copy()
    candidate["variant"] = "alert"
    candidate["response_ratio"] = np.where(
        candidate.alert_active_for_forecast,
        np.maximum(candidate.forecast_volatility_fast,
                   candidate.forecast_volatility_slow) / candidate.forecast_volatility_slow,
        1.0)
    residuals = returns / sigma_slow
    # One simulation per date, as in the unchanged candidate model. This also
    # verifies that the original tail calculations reproduce the saved rows.
    for date in candidate.loc[candidate.response_ratio.gt(1), "date"].unique():
        i = returns.index.get_loc(date)
        if i < window:
            raise ValueError("Insufficient standardized-shock warmup")
        slow = float(sigma_slow.iloc[i])
        ratio = float(candidate.loc[candidate.date.eq(date), "response_ratio"].iloc[0])
        trailing = residuals.iloc[i-window:i].to_numpy() * slow
        bootstrap = np.random.default_rng(seed+i).choice(
            trailing, size=simulations, replace=True)
        if not np.isfinite(trailing).all():
            raise ValueError(f"Invalid trailing shocks at {date}")
        for model, shocks in (("filtered_historical", trailing),
                              ("filtered_bootstrap", bootstrap)):
            for confidence in confidences:
                selector = (candidate.date.eq(date) & candidate.method.eq(model) &
                            candidate.confidence.eq(confidence))
                if selector.sum() != 1:
                    raise ValueError("Missing candidate forecast")
                old = empirical_risk(shocks, confidence)
                saved_row = reference.loc[selector].iloc[0]
                if (not np.isclose(old["var_log"], saved_row.var_log, rtol=1e-9, atol=1e-12)
                        or not np.isclose(old["es_simple_pct"] * value, saved_row.es_usd,
                                          rtol=1e-9, atol=.01)):
                    raise ValueError(f"Empirical reference does not reproduce {model} on {date}")
                updated = empirical_risk(shocks * ratio, confidence)
                for field, result in (("var_log", updated["var_log"]),
                                      ("es_log", updated["es_log"]),
                                      ("var_simple_pct", updated["var_simple_pct"]),
                                      ("es_simple_pct", updated["es_simple_pct"]),
                                      ("var_usd", updated["var_simple_pct"] * value),
                                      ("es_usd", updated["es_simple_pct"] * value)):
                    candidate.loc[selector, field] = result
                candidate.loc[selector, "forecast_volatility"] = max(
                    float(sigma_fast.iloc[i]), slow)
                candidate.loc[selector, "exception"] = (
                    saved_row.actual_log_return < -updated["var_log"])
    result = pd.concat([reference, candidate], ignore_index=True)
    result["method_original"] = result.method
    result["method"] = result.method + "_" + result.variant
    result = result.sort_values(["date", "method", "confidence"]).reset_index(drop=True)
    if (result.estimation_end >= result.date.astype(str)).any():
        raise ValueError("A forecast uses an observation from its own date")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--prices", type=Path, default=Path("outputs/tables/prices.csv"))
    parser.add_argument("--config", type=Path, default=HERE / "filtered_response.yaml")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    alert_config = HERE / spec["reference_alert_protocol"]
    alert_spec = yaml.safe_load(alert_config.read_text(encoding="utf-8"))
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Reference experiment must be complete")
    original = manifest["experiment_configuration"]
    base = manifest["base_configuration"]
    if (spec["models"] != ["filtered_historical", "filtered_bootstrap"]
            or spec["confidence_levels"] != base["var"]["confidence_levels"]
            or alert_spec["volatility"]["reference_ewma_lambda"] != original["filter"]["ewma_lambda"]
            or alert_spec["volatility"]["min_periods"] != original["filter"]["min_periods"]):
        raise ValueError("Fixed response protocol differs from the saved model protocol")
    saved_price_hash = next((hash_value for name, hash_value in
                             manifest.get("input_sha256", {}).items() if name.replace("\\", "/").endswith(
                                 "outputs/tables/prices.csv")), None)
    if saved_price_hash is not None and _hash(args.prices) != saved_price_hash:
        raise ValueError("Price snapshot differs from the reference run")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True, float_precision="round_trip")
    _, simple = compute_returns(prices)
    complete = simple.loc[:, base["data"]["tickers"]].dropna()
    weights = pd.Series(manifest["training"]["weights"])
    returns = np.log1p(complete.loc[pd.Timestamp(base["validation"]["train_start"]):] @ weights)
    vcfg = alert_spec["volatility"]
    slow = causal_volatility(returns, vcfg["reference_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    fast = causal_volatility(returns, vcfg["fast_ewma_lambda"],
                             vcfg["min_periods"], vcfg["floor"])
    threshold, training_triggers, training_days = calibrate_threshold(
        returns, slow, pd.Timestamp(base["validation"]["train_end"]),
        alert_spec["detector"])
    alerts = detector_path(returns, slow, threshold,
                           alert_spec["detector"]["standardized_loss_drift"],
                           alert_spec["detector"]["active_forecast_days"])
    saved = pd.read_csv(args.results / "forecasts.csv.gz", parse_dates=["date"])
    output = build_filtered_response(
        saved, returns, slow, fast, alerts, base["project"]["portfolio_value_usd"],
        base["project"]["seed"], original["historical_window"],
        original["monte_carlo_simulations"], spec["models"], spec["confidence_levels"])
    tests = backtest_forecasts(output, base["validation"].get("significance", .05))
    loss = -output.actual_log_return
    output["quantile_loss"] = (output.confidence - (loss < output.var_log).astype(float)) * (
        loss - output.var_log)
    scores = output.groupby(["method", "confidence"], as_index=False).agg(
        mean_quantile_loss=("quantile_loss", "mean"),
        average_var_usd=("var_usd", "mean"), average_es_usd=("es_usd", "mean"))
    tests = tests.merge(scores, on=["method", "confidence"], validate="one_to_one")
    yearly = output.assign(year=output.date.dt.year).groupby(
        ["method", "confidence", "year"], as_index=False).agg(
            observations=("exception", "size"), breaches=("exception", "sum"))
    impacts = []
    for model in spec["models"]:
        for confidence in spec["confidence_levels"]:
            part = output[output.confidence.eq(confidence)]
            reference = part[part.method.eq(model + "_reference")].set_index("date")
            candidate = part[part.method.eq(model + "_alert")].set_index("date")
            uplift = candidate.var_usd - reference.var_usd
            active = candidate.alert_active_for_forecast
            impacts.append({"model": model, "confidence": confidence,
                            "alert_active_days": int(active.sum()),
                            "days_with_higher_var": int((uplift > .01).sum()),
                            "mean_uplift_active_usd": float(uplift[active].mean()),
                            "mean_uplift_all_days_usd": float(uplift.mean()),
                            "reference_breaches_avoided": int((reference.exception &
                                                               ~candidate.exception).sum()),
                            "new_breaches": int((~reference.exception &
                                                 candidate.exception).sum())})
    impacts = pd.DataFrame(impacts)
    out = args.results / "filtered_response"
    out.mkdir(parents=True, exist_ok=True)
    output.to_csv(out / "daily_forecasts.csv.gz", index=False)
    alerts.loc[alerts.date.ge(output.date.min())].to_csv(out / "detector_timeline.csv", index=False)
    tests.to_csv(out / "backtests.csv", index=False)
    yearly.to_csv(out / "yearly.csv", index=False)
    impacts.to_csv(out / "alert_impact.csv", index=False)
    metadata = {"protocol": spec, "alert_protocol": alert_spec,
                "training_only_threshold": threshold, "training_trigger_count": training_triggers,
                "training_volatility_days": training_days,
                "input_sha256": {str(path): _hash(path) for path in
                                 (args.config, alert_config, args.prices,
                                  args.results / "forecasts.csv.gz")},
                "already_inspected_development_period": True}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Filtered empirical next-day response", "",
             "The detector rule and fast/slow EWMA settings are exactly those declared in the separate EVT response protocol. Its threshold is calibrated using 2007-2017 returns only. Both filtered models and all three confidences use the same 2,212 evaluation dates. This 2018-2026 period was already inspected before the idea was proposed; these are development results.",
             f"The training-calibrated threshold is {threshold:g} ({training_triggers} training triggers in {training_days} valid volatility days). A close-of-day alert can first change the next day's forecast. The candidate uses max(slow, fast) EWMA only while the alert is active; historical standardized shocks and bootstrap draws are otherwise unchanged.", ""]
    for confidence in spec["confidence_levels"]:
        lines += [f"## {confidence:.0%} VaR", "",
                  "| Model | Breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Mean VaR USD |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in tests[tests.confidence.eq(confidence)].itertuples():
            lines.append(f"| {row.method} | {row.exceptions} | {row.kupiec_p_value:.5g} | "
                         f"{row.independence_p_value:.5g} | {row.conditional_coverage_p_value:.5g} | "
                         f"{row.mean_quantile_loss:.7g} | {row.average_var_usd:,.0f} |")
        lines.append("")
    lines += ["## Alert impact", "",
              "| Model | Confidence | Active days | Days with higher VaR | Mean uplift on active days USD | Breaches avoided | New breaches |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in impacts.itertuples():
        lines.append(f"| {row.model} | {row.confidence:.0%} | {row.alert_active_days} | "
                     f"{row.days_with_higher_var} | {row.mean_uplift_active_usd:,.0f} | "
                     f"{row.reference_breaches_avoided} | {row.new_breaches} |")
    lines += ["", "P-values below 0.05 reject the corresponding null. Non-rejection is not proof of correctness. A higher p-value after examining this same history does not independently validate the rule. VaR backtests do not validate ES. The alert raises forecasts on some days and may reduce sensitivity to future losses; use the quantile loss and average VaR columns alongside breach counts.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Filtered-response comparison written to {out}")


if __name__ == "__main__":
    main()
