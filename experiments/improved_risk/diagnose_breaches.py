"""Describe saved 99% VaR breaches without fitting or selecting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def diagnose(forecasts: pd.DataFrame, weights: dict[str, float],
             prices: pd.DataFrame, portfolio_value_usd: float) -> dict[str, pd.DataFrame]:
    data = forecasts.loc[forecasts.confidence.eq(.99) &
                         ~forecasts.category.eq("sensitivity")].copy()
    data["date"] = pd.to_datetime(data.date)
    data = data.sort_values(["method", "date"])
    if data.duplicated(["method", "date"]).any():
        raise ValueError("Duplicate model/date forecast")
    date_sets = data.groupby("method").date.agg(lambda dates: tuple(dates))
    if date_sets.nunique() != 1:
        raise ValueError("Models must share a common evaluation sample")
    data["breach_margin_usd"] = data.actual_loss_usd - data.var_usd
    data["breach_margin_pct"] = 100 * (data.actual_loss_usd - data.var_usd) / data.var_usd
    data["year"] = data.date.dt.year
    data["evaluation_day"] = data.groupby("method").cumcount()
    data["prior_breach"] = data.groupby("method").exception.shift(fill_value=False)
    data["adjacent_breach"] = data.exception & data.prior_breach

    # A 5-observation episode joins breaches no more than five trading days apart.
    breach = data.loc[data.exception].copy()
    breach["gap_days"] = breach.groupby("method").evaluation_day.diff()
    breach["episode_number"] = (breach.gap_days.isna() | breach.gap_days.gt(5)).groupby(
        breach.method).cumsum()
    episodes = breach.groupby(["method", "episode_number"], as_index=False).agg(
        start=("date", "min"), end=("date", "max"), breaches=("date", "size"),
        worst_loss_usd=("actual_loss_usd", "max"),
        largest_margin_usd=("breach_margin_usd", "max"))
    episodes = episodes.sort_values(["breaches", "largest_margin_usd"], ascending=False)

    yearly = data.groupby(["method", "year"], as_index=False).agg(
        observations=("exception", "size"), breaches=("exception", "sum"),
        adjacent_breaches=("adjacent_breach", "sum"))
    yearly["expected_breaches"] = yearly.observations * .01
    yearly["breach_rate"] = yearly.breaches / yearly.observations
    summary = data.groupby("method", as_index=False).agg(
        observations=("exception", "size"), breaches=("exception", "sum"),
        adjacent_breaches=("adjacent_breach", "sum"),
        average_var_usd=("var_usd", "mean"))
    ep = episodes.groupby("method").agg(
        episodes=("episode_number", "size"), largest_episode=("breaches", "max"))
    summary = summary.join(ep, on="method")

    # Same lagged volatility forecast on every method's date; labels are descriptive.
    vol = data.loc[data.method.eq("filtered_bootstrap"),
                   ["date", "forecast_volatility"]].copy()
    if not vol.empty and vol.forecast_volatility.notna().all():
        vol["volatility_quartile"] = pd.qcut(
            vol.forecast_volatility, 4, labels=["Q1 calm", "Q2", "Q3", "Q4 turbulent"])
        data = data.merge(vol[["date", "volatility_quartile"]], on="date", how="left")
        breach = breach.merge(vol[["date", "volatility_quartile"]], on="date", how="left")
        regimes = data.groupby(["method", "volatility_quartile"], observed=True,
                               as_index=False).agg(observations=("exception", "size"),
                                                   breaches=("exception", "sum"))
        regimes["expected_breaches"] = regimes.observations * .01
        regimes["breach_rate"] = regimes.breaches / regimes.observations
    else:
        regimes = pd.DataFrame()

    # Realized asset arithmetic attributes portfolio loss on each date. It does not
    # establish which asset caused the risk model's forecast error.
    prices = prices.sort_index()
    asset_returns = prices[list(weights)].pct_change(fill_method=None)
    components = asset_returns.mul(pd.Series(weights), axis=1).mul(-portfolio_value_usd)
    components.index.name = "date"
    components = components.rename(columns=lambda name: f"loss_{name}_usd").reset_index()
    breach = breach.merge(components, on="date", how="left", validate="many_to_one")
    asset_cols = [f"loss_{name}_usd" for name in weights]
    if breach[asset_cols].isna().any().any():
        raise ValueError("Price snapshot does not cover all breach dates and assets")
    if not np.allclose(breach[asset_cols].sum(axis=1), breach.actual_loss_usd,
                       atol=.01, rtol=1e-10):
        raise ValueError("Realized asset contributions do not reconcile to losses")
    breach["largest_loss_asset"] = breach[asset_cols].idxmax(axis=1).str.removeprefix("loss_").str.removesuffix("_usd")
    return {"timeline": data, "breach_events": breach, "episodes": episodes,
            "yearly": yearly, "summary": summary, "volatility_regimes": regimes}


def export(tables: dict[str, pd.DataFrame], output: Path, source: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name != "timeline":
            table.to_csv(output / f"{name}.csv", index=False)

    timeline = tables["timeline"]
    models = sorted(timeline.method.unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 2.45 * len(models)),
                             sharex=True, layout="constrained")
    for ax, name in zip(axes, models):
        group = timeline[timeline.method.eq(name)]
        ax.plot(group.date, group.actual_loss_usd / 1e6, color="#777777", lw=.55, label="actual loss")
        ax.plot(group.date, group.var_usd / 1e6, color="#1769aa", lw=.75, label="99% VaR")
        hits = group[group.exception]
        ax.scatter(hits.date, hits.actual_loss_usd / 1e6, color="#cf351e", s=12,
                   zorder=3, label="breach")
        ax.set_ylabel(f"{name}\nUSD m")
        ax.grid(alpha=.2)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Evaluation date (2018 onward)")
    fig.savefig(output / "var_breach_timeline.png", dpi=150)
    plt.close(fig)

    summary, yearly, regimes, episodes, breach = (tables[k] for k in
        ("summary", "yearly", "volatility_regimes", "episodes", "breach_events"))
    lines = ["# 99% VaR breach diagnosis", "",
             "This describes saved, already-inspected 2018-2026 forecasts. It fits no models and does not select parameters.",
             "All amounts use the frozen training allocation. A breach is actual loss > same-day forecast VaR.",
             "Adjacent means consecutive trading-day breaches. An episode joins breaches up to five trading days apart; this is descriptive, not a statistical test.",
             "", "## Overview", "",
             "| Model | Breaches | Adjacent pairs | Largest 5-day-gap episode | Mean VaR, USD |",
             "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.breaches}/{r.observations} | {r.adjacent_breaches} | {r.largest_episode} | {r.average_var_usd:,.0f} |")
    lines += ["", "## Years with most breaches", ""]
    for name in sorted(yearly.method.unique()):
        rows = yearly[yearly.method.eq(name)].sort_values("breaches", ascending=False).head(3)
        lines.append(f"- **{name}:** " + ", ".join(
            f"{int(r.year)}: {int(r.breaches)}/{int(r.observations)} (expected {r.expected_breaches:.1f})"
            for r in rows.itertuples()))
    lines += ["", "## Close-together breach episodes", "",
              "Largest episodes across models (overlapping dates are the same market event):", "",
              "| Model | Start | End | Breaches | Largest excess over VaR, USD |",
              "|---|---|---|---:|---:|"]
    for r in episodes.head(15).itertuples():
        lines.append(f"| {r.method} | {r.start.date()} | {r.end.date()} | {r.breaches} | {r.largest_margin_usd:,.0f} |")
    if not regimes.empty:
        lines += ["", "## Forecast-volatility regimes", "",
                  "Quartiles are defined once from filtered-bootstrap's prior-information EWMA volatility over evaluation dates. They describe conditions, not an independent validation.", "",
                  "| Model | Calm Q1 | Q2 | Q3 | Turbulent Q4 |", "|---|---:|---:|---:|---:|"]
        for name in models:
            rows = regimes[regimes.method.eq(name)]
            vals = [f"{int(r.breaches)}/{int(r.observations)}" for r in rows.itertuples()]
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines += ["", "## How to read the files", "",
              "`var_breach_timeline.png` plots daily loss, forecast VaR, and breach dates. `breach_events.csv` gives each breach's forecast, realized loss, excess, volatility regime, and realized asset loss contributions. `episodes.csv`, `yearly.csv`, and `volatility_regimes.csv` provide aggregations.",
              "Realized asset contributions are weight times realized asset return; they do not identify the cause of model error. Large losses after a volatility jump suggest a hypothesis to investigate, not a proved mechanism.",
              "These VaR diagnostics do not validate ES. The 2018-2026 sample is already inspected; no p-value here is an untouched holdout result.", ""]
    (output / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "source.json").write_text(json.dumps({"forecast_file": str(source),
        "confidence": .99, "episode_gap_trading_days": 5,
        "allocation": "frozen training weights from experiment manifest"}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--prices", type=Path, default=Path("outputs/tables/prices.csv"))
    args = parser.parse_args()
    forecasts = args.results / "forecasts.csv.gz"
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    weights = manifest["training"]["weights"]
    portfolio_value_usd = manifest["base_configuration"]["project"]["portfolio_value_usd"]
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True, float_precision="round_trip")
    tables = diagnose(pd.read_csv(forecasts, parse_dates=["date"]), weights, prices,
                      portfolio_value_usd)
    export(tables, args.results / "diagnostics", forecasts)
    print(f"Breach diagnostics written to {args.results / 'diagnostics'}")


if __name__ == "__main__":
    main()
