"""Side-by-side causal EVT volatility scaling and next-day CUSUM response."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.cleaning import compute_returns
from src.quantedge.evt import gpd_risk
from .models import causal_volatility


HERE = Path(__file__).resolve().parent


def detector_path(returns: pd.Series, volatility: pd.Series, threshold: float,
                  drift: float, active_days: int) -> pd.DataFrame:
    """Forecast-day state is set only by returns observed on earlier dates."""
    if (not returns.index.equals(volatility.index) or threshold <= 0 or drift <= 0
            or active_days < 1):
        raise ValueError("Invalid detector inputs")
    score = 0.0
    active_until = -1
    rows = []
    for i, (date, actual) in enumerate(returns.items()):
        active = i <= active_until
        prior_score = score
        v = volatility.iloc[i]
        trigger = False
        standardized_loss = np.nan
        if np.isfinite(v):
            standardized_loss = -float(actual) / float(v)
            score = max(0.0, score + standardized_loss - drift)
            if score >= threshold:
                trigger = True
                score = 0.0
                active_until = max(active_until, i + active_days)
        rows.append({"date": date, "evaluation_index": i,
                     "alert_active_for_forecast": active,
                     "score_before_close": prior_score,
                     "standardized_loss_after_close": standardized_loss,
                     "trigger_after_close": trigger,
                     "active_through_index": active_until})
    return pd.DataFrame(rows)


def calibrate_threshold(returns: pd.Series, volatility: pd.Series, train_end: pd.Timestamp,
                        spec: dict) -> tuple[float, int, int]:
    training = returns.index <= train_end
    count = int((training & volatility.notna()).sum())
    if count < 252:
        raise ValueError("Insufficient training history for detector calibration")
    target = float(spec["maximum_training_triggers_per_year"])
    grid = np.arange(spec["threshold_grid_start"],
                     spec["threshold_grid_stop"] + spec["threshold_grid_step"] / 2,
                     spec["threshold_grid_step"])
    allowed = count / 252 * target
    for threshold in grid:
        path = detector_path(returns.loc[training], volatility.loc[training],
                             float(threshold), spec["standardized_loss_drift"],
                             spec["active_forecast_days"])
        triggers = int(path.trigger_after_close.sum())
        if triggers <= allowed:
            return float(threshold), triggers, count
    raise ValueError("Training alarm target unavailable on the declared threshold grid")


def build_arms(saved: pd.DataFrame, raw_fits: pd.DataFrame,
               filtered_fits: pd.DataFrame, returns: pd.Series,
               sigma_slow: pd.Series, sigma_fast: pd.Series, alerts: pd.DataFrame,
               notional: float) -> pd.DataFrame:
    selected = saved[saved.confidence.eq(.99) &
                     saved.method.isin(["evt_gpd", "filtered_evt"])].copy()
    selected["date"] = pd.to_datetime(selected.date)
    if selected.duplicated(["date", "method"]).any():
        raise ValueError("Duplicate reference EVT forecasts")
    expected = returns.loc[selected.date.min():].index
    for _, frame in selected.groupby("method"):
        if not frame.sort_values("date").date.reset_index(drop=True).equals(
                pd.Series(expected).reset_index(drop=True)):
            raise ValueError("Reference EVT methods do not cover identical dates")
    raw = selected[selected.method.eq("evt_gpd")].sort_values("date").copy()
    filtered = selected[selected.method.eq("filtered_evt")].sort_values("date").copy()
    raw_fit_dates = raw.groupby("estimation_end").date.min().to_dict()
    filtered_fit_dates = filtered.groupby("tail_fit_end").date.min().to_dict()
    alert = alerts.set_index("date").alert_active_for_forecast
    raw_fits = raw_fits[raw_fits.selected.eq(True) & raw_fits.confidence.eq(.99)].copy()
    raw_fits["fit_date"] = pd.to_datetime(raw_fits.fit_date)
    raw_fits = raw_fits.set_index("fit_date")
    if raw_fits.index.has_duplicates:
        raise ValueError("Duplicate selected raw EVT fits")
    filtered_fits = filtered_fits[filtered_fits.selected.eq(True) &
                                  filtered_fits.model.eq("filtered_evt")].copy()
    filtered_fits["fit_date"] = pd.to_datetime(filtered_fits.fit_date)
    filtered_fits = filtered_fits.set_index("fit_date")
    if filtered_fits.index.has_duplicates:
        raise ValueError("Duplicate selected filtered EVT fits")
    if not np.allclose(filtered.set_index("date").forecast_volatility,
                       sigma_slow.loc[filtered.date], rtol=1e-10, atol=1e-12):
        raise ValueError("Reconstructed EWMA volatility differs from filtered EVT reference")
    rows = []
    for item in raw.itertuples():
        date = item.date
        # estimation_end is the last return in the fitted sample; the first
        # forecast bearing that value is the fit date.
        fit_date = raw_fit_dates[item.estimation_end]
        if fit_date not in raw_fits.index:
            raise ValueError(f"Missing selected raw EVT parameters for {fit_date}")
        fit = raw_fits.loc[fit_date]
        base_sigma = float(sigma_slow.loc[fit_date])
        slow = float(sigma_slow.loc[date])
        fast = float(sigma_fast.loc[date])
        active = bool(alert.loc[date])
        ratios = {"raw_evt_daily_scale": slow/base_sigma,
                  "raw_evt_daily_scale_cusum": max(slow, fast)/base_sigma if active else slow/base_sigma}
        reference_risk = gpd_risk(float(fit.threshold_log), float(fit["shape"]),
                                  float(fit.scale), float(fit.exceedance_probability), .99)
        if not np.isclose(reference_risk["var_log"], item.var_log, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Raw EVT parameters do not reproduce saved VaR on {date}")
        rows.append(_reference_row(item, "raw_evt_reference", active, slow, fast, fit_date))
        for arm, ratio in ratios.items():
            risk = gpd_risk(float(fit.threshold_log)*ratio, float(fit["shape"]),
                            float(fit.scale)*ratio, float(fit.exceedance_probability), .99)
            rows.append(_risk_row(item, arm, risk, notional, active, slow, fast, fit_date))
    for item in filtered.itertuples():
        date = item.date
        fit_date = filtered_fit_dates[item.tail_fit_end]
        if fit_date not in filtered_fits.index:
            raise ValueError(f"Missing selected filtered EVT parameters for {fit_date}")
        fit = filtered_fits.loc[fit_date]
        slow = float(sigma_slow.loc[date])
        fast = float(sigma_fast.loc[date])
        active = bool(alert.loc[date])
        rows.append(_reference_row(item, "filtered_evt_reference", active, slow, fast, fit_date))
        base_risk = gpd_risk(float(fit.threshold_standardized)*slow,
                             float(fit["shape"]), float(fit.scale)*slow,
                             float(fit.exceedance_probability), .99)
        if not np.isclose(base_risk["var_log"], item.var_log, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Filtered EVT parameters do not reproduce saved VaR on {date}")
        overlay_vol = max(slow, fast) if active else slow
        risk = gpd_risk(float(fit.threshold_standardized)*overlay_vol,
                        float(fit["shape"]), float(fit.scale)*overlay_vol,
                        float(fit.exceedance_probability), .99)
        rows.append(_risk_row(item, "filtered_evt_cusum", risk, notional,
                              active, slow, fast, fit_date))
    output = pd.DataFrame(rows).sort_values(["date", "method"]).reset_index(drop=True)
    if output.groupby("method").date.nunique().nunique() != 1:
        raise ValueError("EVT arms have differing evaluation dates")
    return output


def _reference_row(item, name: str, active: bool, slow: float,
                   fast: float, fit_date: pd.Timestamp) -> dict:
    return {"date": item.date, "method": name, "confidence": .99,
            "evaluation_index": item.evaluation_index, "estimation_end": item.estimation_end,
            "var_log": item.var_log, "es_log": item.es_log,
            "var_usd": item.var_usd, "es_usd": item.es_usd,
            "actual_log_return": item.actual_log_return,
            "actual_loss_usd": item.actual_loss_usd, "exception": item.exception,
            "alert_active_for_forecast": active,
            "forecast_volatility_slow": slow,
            "forecast_volatility_fast": fast, "tail_fit_date": fit_date}


def _risk_row(item, name: str, risk: dict, value: float, active: bool,
              slow: float, fast: float, fit_date: pd.Timestamp) -> dict:
    return {"date": item.date, "method": name, "confidence": .99,
            "evaluation_index": item.evaluation_index, "estimation_end": item.estimation_end,
            "var_log": risk["var_log"], "es_log": risk["es_log"],
            "var_usd": risk["var_simple_pct"] * value,
            "es_usd": risk["es_simple_pct"] * value,
            "actual_log_return": item.actual_log_return,
            "actual_loss_usd": item.actual_loss_usd,
            "exception": item.actual_log_return < -risk["var_log"],
            "alert_active_for_forecast": active,
            "forecast_volatility_slow": slow, "forecast_volatility_fast": fast,
            "tail_fit_date": fit_date}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot_crisis(arms: pd.DataFrame, output: Path) -> None:
    selected = arms[arms.date.between("2020-02-20", "2020-03-20")]
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    actual = selected[selected.method.eq("raw_evt_reference")]
    ax.plot(actual.date, actual.actual_loss_usd / 1e6, color="black", lw=1.6,
            marker="o", markersize=2.5, label="Actual loss")
    for name, color, style in (
        ("raw_evt_reference", "#ae5c14", "--"),
        ("raw_evt_daily_scale", "#ef9b18", "-"),
        ("raw_evt_daily_scale_cusum", "#c63420", "-"),
        ("filtered_evt_reference", "#247aa7", "--"),
        ("filtered_evt_cusum", "#154888", "-"),
    ):
        frame = selected[selected.method.eq(name)]
        ax.plot(frame.date, frame.var_usd / 1e6, color=color, linestyle=style,
                lw=1.4, label=name)
    ax.set_ylabel("USD millions")
    ax.set_title("One-day 99% VaR and realized loss, February-March 2020")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--prices", type=Path, default=Path("outputs/tables/prices.csv"))
    parser.add_argument("--config", type=Path, default=HERE / "evt_response.yaml")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Reference experiment must be complete")
    base = manifest["base_configuration"]
    expected_price_hash = manifest.get("input_sha256", {}).get(str(args.prices))
    if expected_price_hash is not None and _hash(args.prices) != expected_price_hash:
        raise ValueError("Price snapshot differs from the saved reference experiment")
    if spec["volatility"]["reference_ewma_lambda"] != manifest["experiment_configuration"]["filter"]["ewma_lambda"]:
        raise ValueError("Reference EWMA setting differs from saved experiment")
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True, float_precision="round_trip")
    _, simple = compute_returns(prices)
    weights = pd.Series(manifest["training"]["weights"])
    returns = np.log1p(simple.loc[:, base["data"]["tickers"]].dropna().loc[
        pd.Timestamp(base["validation"]["train_start"]):] @ weights)
    vcfg = spec["volatility"]
    slow = causal_volatility(returns, vcfg["reference_ewma_lambda"], vcfg["min_periods"], vcfg["floor"])
    fast = causal_volatility(returns, vcfg["fast_ewma_lambda"], vcfg["min_periods"], vcfg["floor"])
    threshold, training_triggers, training_days = calibrate_threshold(
        returns, slow, pd.Timestamp(base["validation"]["train_end"]), spec["detector"])
    alerts = detector_path(returns, slow, threshold, spec["detector"]["standardized_loss_drift"],
                           spec["detector"]["active_forecast_days"])
    saved = pd.read_csv(args.results / "forecasts.csv.gz", parse_dates=["date"])
    raw_fits = pd.read_csv(args.results / "baseline_evt_diagnostics.csv", parse_dates=["fit_date"])
    filtered_fits = pd.read_csv(args.results / "fit_diagnostics.csv", parse_dates=["fit_date"])
    arms = build_arms(saved, raw_fits, filtered_fits, returns, slow, fast, alerts,
                      float(base["project"]["portfolio_value_usd"]))
    tests = backtest_forecasts(arms, base["validation"].get("significance", .05))
    loss = -arms.actual_log_return
    arms["quantile_loss"] = (arms.confidence - (loss < arms.var_log).astype(float)) * (loss-arms.var_log)
    scores = arms.groupby("method", as_index=False).agg(mean_quantile_loss=("quantile_loss", "mean"),
        average_var_usd=("var_usd", "mean"), average_es_usd=("es_usd", "mean"))
    tests = tests.merge(scores, on="method", validate="one_to_one")
    out = args.results / "evt_response"
    out.mkdir(parents=True, exist_ok=True)
    arms.to_csv(out / "daily_forecasts.csv.gz", index=False)
    _plot_crisis(arms, out / "feb_mar_2020.png")
    alerts.loc[alerts.date.ge(pd.Timestamp(base["validation"]["test_start"]))].to_csv(
        out / "detector_timeline.csv", index=False)
    tests.to_csv(out / "backtests.csv", index=False)
    arms.assign(year=arms.date.dt.year).groupby(["method", "year"], as_index=False).agg(
        observations=("exception", "size"), breaches=("exception", "sum")
    ).to_csv(out / "yearly.csv", index=False)
    eval_alerts = alerts.loc[alerts.date.ge(arms.date.min())].copy()
    reference = arms[arms.method.eq("filtered_evt_reference")].set_index("evaluation_index")
    alert_rows = []
    for event in eval_alerts[eval_alerts.trigger_after_close].itertuples():
        next_days = reference.loc[(reference.index > event.evaluation_index) &
                                  (reference.index <= event.evaluation_index + spec["detector"]["active_forecast_days"])]
        alert_rows.append({"trigger_close": event.date,
                           "next_forecast_date": next_days.date.min() if len(next_days) else pd.NaT,
                           "next_days_observed": len(next_days),
                           "reference_breach_next_5_days": bool(next_days.exception.any()),
                           "largest_actual_loss_next_5_days_usd": next_days.actual_loss_usd.max()
                           if len(next_days) else np.nan})
    alert_review = pd.DataFrame(alert_rows)
    alert_review.to_csv(out / "alert_review.csv", index=False)
    uplift_rows = []
    for root, candidate in (("raw_evt_daily_scale", "raw_evt_daily_scale_cusum"),
                            ("filtered_evt_reference", "filtered_evt_cusum")):
        base_rows = arms[arms.method.eq(root)].set_index("date")
        new_rows = arms[arms.method.eq(candidate)].set_index("date")
        active = base_rows.alert_active_for_forecast
        delta = new_rows.var_usd - base_rows.var_usd
        uplift_rows.append({"reference": root, "candidate": candidate,
                            "active_days": int(active.sum()),
                            "days_with_higher_var": int((delta > .01).sum()),
                            "average_uplift_active_days_usd": float(delta[active].mean()),
                            "maximum_uplift_usd": float(delta.max()),
                            "reference_breaches_avoided": int((base_rows.exception &
                                                              ~new_rows.exception).sum()),
                            "new_breaches": int((~base_rows.exception &
                                                 new_rows.exception).sum())})
    uplift = pd.DataFrame(uplift_rows)
    uplift.to_csv(out / "alert_impact.csv", index=False)
    provenance = {"protocol": spec, "detector_threshold_training_only": threshold,
                  "training_trigger_count": training_triggers, "training_volatility_days": training_days,
                  "input_sha256": {str(p): _hash(p) for p in
                                   (args.prices, args.config, args.results / "forecasts.csv.gz",
                                    args.results / "baseline_evt_diagnostics.csv")},
                  "already_inspected_development_period": True}
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    lines = ["# EVT daily response experiment", "",
             "Five predeclared EVT paths use the same frozen allocation and evaluation dates. Detector threshold was calibrated only on training returns. The 2018-2026 period is already inspected development evidence.",
             f"Training-only CUSUM threshold: **{threshold:g}**; {training_triggers} triggers in {training_days} valid training days. The signal computed after close first changes the next day's forecast, and stays active for five forecast days unless another trigger extends it.",
             "", "| Arm | 99% breaches | Kupiec p | Independence p | Combined p | Mean quantile loss | Average VaR USD |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in tests.itertuples():
        lines.append(f"| {r.method} | {r.exceptions} | {r.kupiec_p_value:.5g} | "
                     f"{r.independence_p_value:.5g} | {r.conditional_coverage_p_value:.5g} | "
                     f"{r.mean_quantile_loss:.7g} | {r.average_var_usd:,.0f} |")
    complete_alerts = alert_review[alert_review.next_days_observed.eq(
        spec["detector"]["active_forecast_days"])]
    lines += ["", "## Signal and forecast impact", "",
              f"The detector triggered after {len(alert_review)} evaluation-period closes. It was active for {int(eval_alerts.alert_active_for_forecast.sum())} forecast days. Among {len(complete_alerts)} triggers with a full five-day follow-up, {int(complete_alerts.reference_breach_next_5_days.sum())} were followed by a filtered-EVT reference breach. The others are not automatically false alarms: an alert can respond to an unusually large return even when later losses stay below VaR.",
              "", "| Candidate overlay | Days with higher VaR | Mean uplift on active days | Reference breaches avoided | New breaches |",
              "|---|---:|---:|---:|---:|"]
    for r in uplift.itertuples():
        lines.append(f"| {r.candidate} | {r.days_with_higher_var} | ${r.average_uplift_active_days_usd:,.0f} | {r.reference_breaches_avoided} | {r.new_breaches} |")
    lines += ["", "The overlay uses max(slow EWMA, fast EWMA) only during an active alert. Raw daily scaling keeps the 21-day GPD fit but multiplies its threshold and scale by today's slow volatility divided by the volatility at fit time. Filtered EVT retains its standardized GPD fit. VaR/ES are recomputed in the original GPD units; there is no fitted VaR multiplier.",
              "", "A higher p-value on this inspected period is not validation. Check signal timing, false alerts, risk forecast size, quantile loss, and crisis paths; ES is not validated by VaR tests.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EVT response comparison written to {out}")


if __name__ == "__main__":
    main()
