"""Compare one declared faster EWMA response for the two empirical filtered models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.tail import empirical_risk
from .models import causal_volatility


HERE = Path(__file__).resolve().parent


def forecasts_for_lambda(returns: pd.Series, start: pd.Timestamp, config: dict,
                         base: dict, decay: float) -> pd.DataFrame:
    f = base["filter"]
    sigma = causal_volatility(returns, decay, f["min_periods"], f["volatility_floor"])
    residuals = returns / sigma
    window = int(base["historical_window"])
    simulations = int(base["monte_carlo_simulations"])
    value = float(config["project"]["portfolio_value_usd"])
    seed = int(config["project"]["seed"])
    rows = []
    for i, date in enumerate(returns.index):
        if date < start:
            continue
        if i < f["min_periods"] + window or not np.isfinite(sigma.iloc[i]):
            raise ValueError(f"Insufficient causal volatility warmup on {date}")
        volatility = float(sigma.iloc[i])
        shocks = residuals.iloc[i-window:i].to_numpy() * volatility
        if not np.isfinite(shocks).all():
            raise ValueError(f"Invalid standardized shock window on {date}")
        simulated = np.random.default_rng(seed+i).choice(shocks, size=simulations, replace=True)
        for model, sample in (("filtered_historical", shocks),
                              ("filtered_bootstrap", simulated)):
            risk = empirical_risk(sample, .99)
            actual = float(returns.iloc[i])
            rows.append({"date": date, "method": model, "confidence": .99,
                         "evaluation_index": i, "estimation_end": returns.index[i-1],
                         "forecast_volatility": volatility, "var_log": risk["var_log"],
                         "var_usd": risk["var_simple_pct"] * value,
                         "actual_log_return": actual,
                         "actual_loss_usd": -np.expm1(actual) * value,
                         "exception": actual < -risk["var_log"],
                         "lambda": decay})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--prices", type=Path, default=Path("outputs/tables/prices.csv"))
    parser.add_argument("--config", type=Path, default=HERE / "volatility_response.yaml")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if spec["confidence"] != .99 or spec["models"] != ["filtered_historical", "filtered_bootstrap"]:
        raise ValueError("This diagnostic probe is limited to the declared 99% empirical filtered models")
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("The reference comparison must be complete")
    base, config = manifest["experiment_configuration"], manifest["base_configuration"]
    if spec["reference_ewma_lambda"] != base["filter"]["ewma_lambda"]:
        raise ValueError("Reference lambda differs from saved experiment")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True, float_precision="round_trip")
    _, simple = compute_returns(prices)
    complete = simple.loc[:, config["data"]["tickers"]].dropna()
    weights = pd.Series(manifest["training"]["weights"])
    returns = np.log1p(complete.loc[pd.Timestamp(config["validation"]["train_start"]):] @ weights)
    start = pd.Timestamp(config["validation"]["test_start"])
    saved = pd.read_csv(args.results / "forecasts.csv.gz", parse_dates=["date"])
    saved = saved[saved.confidence.eq(.99) & saved.method.isin(spec["models"])]
    if not returns.loc[start:].index.equals(pd.Index(sorted(saved.date.unique()))):
        raise ValueError("Saved and reconstructed evaluation dates differ")
    reference = forecasts_for_lambda(returns, start, config, base, spec["reference_ewma_lambda"])
    check = reference.merge(saved[["date", "method", "var_usd", "actual_loss_usd", "exception"]],
                            on=["date", "method"], validate="one_to_one", suffixes=("", "_saved"))
    if len(check) != len(reference) or not np.allclose(check.var_usd, check.var_usd_saved, atol=.01,
                                                        rtol=1e-9) or not np.allclose(
                                                            check.actual_loss_usd, check.actual_loss_usd_saved,
                                                            atol=.01, rtol=1e-9) or not (check.exception ==
                                                                                        check.exception_saved).all():
        raise ValueError("Reference implementation does not reproduce saved forecasts")
    alternative = forecasts_for_lambda(returns, start, config, base,
                                       spec["alternative_ewma_lambda"])
    all_rows = pd.concat([reference, alternative], ignore_index=True)
    all_rows["variant"] = np.where(all_rows["lambda"].eq(spec["reference_ewma_lambda"]),
                                   "reference_0.94", "faster_0.90")
    all_rows["method_original"] = all_rows.method
    all_rows["method"] = all_rows.method + "_" + all_rows.variant
    tests = backtest_forecasts(all_rows, config["validation"].get("significance", .05))
    loss = -all_rows.actual_log_return
    all_rows["quantile_loss"] = (all_rows.confidence - (loss < all_rows.var_log).astype(float)) * (
        loss - all_rows.var_log)
    score = all_rows.groupby("method", as_index=False).agg(
        mean_quantile_loss=("quantile_loss", "mean"), average_var_usd=("var_usd", "mean"))
    tests = tests.merge(score, on="method", validate="one_to_one")
    all_rows["year"] = all_rows.date.dt.year
    yearly = all_rows.groupby(["method", "year"], as_index=False).agg(
        observations=("exception", "size"), breaches=("exception", "sum"))
    output = args.results / "volatility_response"
    output.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output / "daily_forecasts.csv.gz", index=False)
    tests.to_csv(output / "backtests.csv", index=False)
    yearly.to_csv(output / "yearly.csv", index=False)
    lines = ["# Fixed EWMA response probe", "",
             "The 2018-2026 period is already inspected. This is a descriptive development comparison, not an independent selection or an automatically promoted model.",
             "The declared comparison changes only EWMA lambda from 0.94 to 0.90 for filtered historical and filtered bootstrap. It recomputes each historical standardized shock using its own prior-information volatility estimate. The original 0.94 forecasts are reproduced from the saved data before comparing.",
             "", "| Model | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Average VaR USD |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in tests.itertuples():
        lines.append(f"| {r.method} | {r.exceptions} | {r.kupiec_p_value:.5g} | "
                     f"{r.independence_p_value:.5g} | {r.conditional_coverage_p_value:.5g} | "
                     f"{r.mean_quantile_loss:.7g} | {r.average_var_usd:,.0f} |")
    lines += ["", "P-values below 0.05 reject the corresponding null. Non-rejection is not proof of adequacy. Quantile loss is lower-is-better; average VaR shows forecast conservatism. Review dated breaches and yearly counts as well as aggregate p-values.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Fixed volatility comparison written to {output}")


if __name__ == "__main__":
    main()
