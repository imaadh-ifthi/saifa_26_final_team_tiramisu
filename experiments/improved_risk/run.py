"""Run fixed experimental candidates and unchanged baselines on identical dates."""
import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import yaml
from src.quantedge.analytics import fit_training_portfolio
from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.config import load_config
from src.quantedge.data import load_fred_csv
from src.quantedge.risk import build_risk_forecasts, common_evaluation_sample, summarize_risk
from .models import build_improved_forecasts

PACKAGE = Path(__file__).resolve().parent


def compare_models(returns: pd.Series, base_config: dict, experiment: dict,
                   verbose: bool = False) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(base_config["validation"]["test_start"])
    if verbose:
        print("Computing unchanged baseline forecasts...", flush=True)
    baseline, base_diagnostics = build_risk_forecasts(returns, base_config, start, include_evt=True)
    if verbose:
        print("Computing fixed experimental candidates and sensitivities...", flush=True)
    candidates, diagnostics = build_improved_forecasts(returns, base_config, experiment, start)
    if verbose:
        print("Comparing coverage, independence, forecasting scores and yearly exceptions...", flush=True)
    baseline["category"] = "baseline"
    candidates["category"] = np.where(candidates.method.str.startswith("historical_"),
                                      "sensitivity", "candidate")
    all_forecasts = pd.concat([baseline, candidates], ignore_index=True)
    common = common_evaluation_sample(all_forecasts, list(all_forecasts.method.unique()))
    expected_dates = returns.loc[start:].index
    if set(common.date) != set(expected_dates):
        raise ValueError("All models must cover every declared evaluation date; extend training warmup")
    common["evaluation_basis"] = experiment["evaluation_label"]
    common = common.sort_values(["date", "method", "confidence"]).reset_index(drop=True)
    backtests = backtest_forecasts(common, base_config["validation"].get("significance", .05))
    category = common.groupby("method").category.first()
    backtests["category"] = backtests.method.map(category)
    loss = -common.actual_log_return
    error = loss-common.var_log
    common["quantile_loss"] = (common.confidence-(loss < common.var_log).astype(float))*error
    scores = common.groupby(["method", "confidence"], as_index=False).agg(
        mean_quantile_loss=("quantile_loss", "mean"), average_var_pct=("var_simple_pct", "mean"),
        average_es_pct=("es_simple_pct", "mean"))
    scores["category"] = scores.method.map(category)
    breaches = common[common.exception].copy()
    breaches["excess_loss_usd"] = breaches.actual_loss_usd-breaches.var_usd
    severity = breaches.groupby(["method", "confidence"], as_index=False).agg(
        mean_exception_loss_usd=("actual_loss_usd", "mean"),
        mean_excess_loss_usd=("excess_loss_usd", "mean"), worst_exception_loss_usd=("actual_loss_usd", "max"))
    scores = scores.merge(severity, on=["method", "confidence"], how="left")
    yearly = common.assign(year=common.date.dt.year).groupby(["method", "confidence", "year"], as_index=False).agg(
        observations=("exception", "size"), exceptions=("exception", "sum"))
    yearly["exception_rate"] = yearly.exceptions/yearly.observations
    yearly["expected_exceptions"] = yearly.observations*(1-yearly.confidence)
    return {"forecasts": common, "backtests": backtests, "scores": scores,
            "yearly_exceptions": yearly, "risk_summary": summarize_risk(common),
            "fit_diagnostics": diagnostics, "baseline_evt_diagnostics": base_diagnostics}


def export_comparison(tables: dict, output: Path, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        suffix = ".csv.gz" if name == "forecasts" else ".csv"
        frame.to_csv(output/f"{name}{suffix}", index=False)
    (output/"manifest.json").write_text(json.dumps({**metadata, "status": "complete"}, indent=2, allow_nan=False),
                                        encoding="utf-8")
    backtests = tables["backtests"]
    scores = tables["scores"]
    lines = ["# Fixed-candidate development comparison", "",
             "This historical period was already inspected. Results are development evidence, not an untouched holdout.",
             "No candidate is automatically promoted and no parameter search is performed.", "",
             "## 99% results", "",
             "| Model | Category | Exceptions / observations | Kupiec p | Independence p | Conditional coverage p | Mean quantile loss |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in backtests[backtests.confidence.eq(.99)].itertuples():
        score = scores[(scores.method == row.method) & scores.confidence.eq(.99)].iloc[0]
        lines.append(f"| {row.method} | {row.category} | {row.exceptions} / {row.observations} | "
                     f"{row.kupiec_p_value:.5g} | {row.independence_p_value:.5g} | "
                     f"{row.conditional_coverage_p_value:.5g} | {score.mean_quantile_loss:.6g} |")
    lines.extend(["", "P-values below the configured significance level reject the relevant null. Non-rejection is not proof of correctness.",
                  "Quantile loss is a proper VaR forecasting score (lower is better); compare it only at the same confidence and dates.",
                  "ES is calculated but not independently validated by the VaR coverage tests.", "",
                  "All confidences, yearly exception counts, fit warnings and severity metrics are available in the adjacent CSVs.",
                  "See ../README.md for model definitions, fixed settings, remaining limitations and rerun commands."])
    (output/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config.yaml")
    parser.add_argument("--experiment-config", default=str(PACKAGE/"config.yaml"))
    parser.add_argument("--prices", default="outputs/tables/prices.csv")
    parser.add_argument("--risk-free", default="data/raw/risk_free.csv")
    parser.add_argument("--output-dir", default=str(PACKAGE/"results"))
    args = parser.parse_args()
    base = load_config(args.base_config)
    experiment = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
    output = Path(args.output_dir).resolve()
    for protected in (Path("outputs"), Path("data"), Path("src")):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError("Experimental exports must be outside baseline outputs, data and source folders")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    _, simple = compute_returns(prices)
    complete = simple.loc[:, base["data"]["tickers"]].dropna()
    rates = load_fred_csv(args.risk_free, base["data"]["fred_risk_free_series"])
    weights, training = fit_training_portfolio(complete, rates, base)
    validation = base["validation"]
    end = validation.get("test_end") or complete.index.max()
    history = complete.loc[pd.Timestamp(validation["train_start"]):pd.Timestamp(end)]
    returns = np.log1p(history @ weights)
    metadata = {"protocol": experiment["protocol"], "evaluation_label": experiment["evaluation_label"],
                "base_configuration": base, "experiment_configuration": experiment, "training": training,
                "input_sha256": {str(path): _hash(Path(path)) for path in
                                  (args.prices, args.risk_free, args.base_config, args.experiment_config)},
                "implementation_sha256": {name: _hash(PACKAGE/name) for name in ("models.py", "run.py")},
                "dependency_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "pyyaml")},
                "data_cutoff": str(returns.index.max().date()),
                "selection_policy": "fixed_candidates_no_search_no_automatic_promotion"}
    output.mkdir(parents=True, exist_ok=True)
    # Write the declared protocol before computing results, and retain it if a run fails.
    (output/"manifest.json").write_text(json.dumps({**metadata, "status": "running"}, indent=2), encoding="utf-8")
    try:
        tables = compare_models(returns, base, experiment, verbose=True)
    except Exception as exc:
        (output/"manifest.json").write_text(json.dumps({**metadata, "status": "failed", "error": str(exc)},
                                                     indent=2), encoding="utf-8")
        raise
    export_comparison(tables, output, metadata)
    print(f"Comparison complete: {tables['forecasts'].date.nunique()} dates, "
          f"{tables['backtests'].method.nunique()} models; results in {output}")


if __name__ == "__main__":
    main()
