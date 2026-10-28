"""Run the Layer 3-5 upgrade: new models, ES backtests, extra VaR tests, scoring and MCS.

From the repository root:
    python -m experiments.layers345_upgrade.run            # full run (about 8-10 minutes)
    python -m experiments.layers345_upgrade.run --reuse    # reuse saved GARCH forecasts

Everything is written to experiments/layers345_upgrade/results/. Nothing in outputs/ or in
Daniel's experiments folder is changed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.quantedge.backtesting import backtest_forecasts
from src.quantedge.backtests_extra import dynamic_quantile, duration_test, monte_carlo_pvalues
from src.quantedge.es_backtests import du_escanciano, kratz_multinomial, exceedance_residual_test
from src.quantedge.scoring import fz0_loss, quantile_loss, diebold_mariano, model_confidence_set
from src.quantedge.evt_extra import extremal_index, bootstrap_evt_interval, eqd_threshold
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from .models import load_portfolio, build_garch_family, baseline_pits

HERE = Path(__file__).resolve().parent
NEW = ["gjr_t", "gjr_fhs", "gjr_evt", "gjr_evt_eqd", "combo"]
BASE = ["historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd"]
KEEP = ["date", "method", "confidence", "var_log", "es_log", "var_simple_pct", "es_simple_pct",
        "actual_log_return", "evaluation_index"]


def collect_forecasts(root, base_cfg, cfg, reuse):
    r, run, complete, current = load_portfolio(root, base_cfg)
    test_start = pd.Timestamp(run["test_start"])
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    cached = out / "garch_family_forecasts.csv.gz"
    if reuse and cached.exists():
        garch = pd.read_csv(cached, parse_dates=["date", "estimation_end", "fit_end"])
        fits = pd.read_csv(out / "garch_fits.csv", parse_dates=["fit_date", "estimation_end"])
    else:
        garch, fits = build_garch_family(r, test_start, cfg, cfg["tests"]["confidences"])
        garch.to_csv(cached, index=False)
        fits.to_csv(out / "garch_fits.csv", index=False)
    garch["family"] = "new_gjr_garch"

    baseline = pd.read_csv(root / "outputs/tables/risk_forecasts_oos.csv", parse_dates=["date", "estimation_end"])
    pits = baseline_pits(r, baseline, base_cfg)
    check = baseline[baseline.confidence == 0.99].merge(pits, on=["date", "method"])
    rebuild_error = check.groupby("method").apply(lambda f: float(np.abs(f.var_log - f.check_var99).max()),
                                                  include_groups=False)
    base = baseline[KEEP].merge(pits[["date", "method", "pit", "sigma"]], on=["date", "method"])
    base["family"] = "baseline_main_pipeline"

    ewma_sigma = pits[pits.method == "parametric_ewma"].set_index("date").sigma
    extra = []
    for item in cfg["daniel_candidates"]:
        frame = pd.read_csv((HERE / item["file"]).resolve(), parse_dates=["date"])
        frame = frame[frame.method.isin(item["methods"])].copy()
        # the EVT-response file stores dollar amounts instead of percentages
        for col in ("var", "es"):
            if f"{col}_simple_pct" not in frame:
                frame[f"{col}_simple_pct"] = frame[f"{col}_usd"] / base_cfg["project"]["portfolio_value_usd"]
        frame = frame[KEEP]
        frame["pit"] = np.nan
        frame["sigma"] = frame.date.map(ewma_sigma)
        frame["family"] = "daniel_experiment"
        extra.append(frame)
    cols = KEEP + ["pit", "sigma", "family"]
    allf = pd.concat([garch[cols], base[cols], *extra], ignore_index=True)
    dates = set(garch.date) & set(base.date)
    allf = allf[allf.date.isin(dates)].sort_values(["method", "confidence", "date"]).reset_index(drop=True)
    allf["exception"] = allf.actual_log_return < -allf.var_log
    return allf, fits, rebuild_error, r, complete, current, run


def run_tests(allf, cfg):
    tc = cfg["tests"]
    classic = backtest_forecasts(allf[["date", "method", "confidence", "exception", "evaluation_index"]],
                                 tc["significance"])
    rows = []
    for (method, c), f in allf.groupby(["method", "confidence"]):
        a = 1 - c
        hits = f.exception.to_numpy()
        row = {"method": method, "confidence": c, "family": f.family.iloc[0],
               "avg_var_usd": f.var_simple_pct.mean() * 1e7, "avg_es_usd": f.es_simple_pct.mean() * 1e7,
               "mean_fz0": float(fz0_loss(f.actual_log_return, f.var_log, f.es_log, c).mean()),
               "mean_quantile_loss": float(quantile_loss(f.actual_log_return, f.var_log, c).mean())}
        row.update(dynamic_quantile(hits, f.var_log.to_numpy(), a, tc["dq_lags"]))
        row.update(duration_test(hits))
        row.update(monte_carlo_pvalues(hits, a, tc["monte_carlo_sims"]))
        row.update(exceedance_residual_test(f.actual_log_return, f.var_log, f.es_log, f.sigma))
        if f.pit.notna().all():
            row.update(du_escanciano(f.pit.to_numpy(), a, tc["de_lags"]))
        rows.append(row)
    extra = pd.DataFrame(rows)
    table = classic.merge(extra, on=["method", "confidence"])
    multinomial = []
    for method, f in allf[allf.confidence == tc["confidences"][0]].groupby("method"):
        if f.pit.notna().all():
            multinomial.append({"method": method, **kratz_multinomial(f.pit.to_numpy(), tc["multinomial_base"],
                                                                      tc["multinomial_levels"])})
    return table, pd.DataFrame(multinomial)


def run_scoring(allf, cfg):
    tc = cfg["tests"]
    mcs_tables, dm_rows = [], []
    for c in tc["confidences"]:
        f = allf[allf.confidence == c]
        loss = f.assign(loss=fz0_loss(f.actual_log_return, f.var_log, f.es_log, c)).pivot(
            index="date", columns="method", values="loss").dropna(axis=1)
        mcs = model_confidence_set(loss, tc["mcs_level"], tc["mcs_bootstrap"], tc["mcs_block"])
        mcs["confidence"] = c
        mcs_tables.append(mcs)
        for ref in ("historical", "filtered_historical_alert"):
            if ref not in loss:
                continue
            for m in loss.columns:
                if m != ref:
                    dm_rows.append({"confidence": c, "model": m, "reference": ref,
                                    **diebold_mariano(loss[m], loss[ref])})
    return pd.concat(mcs_tables, ignore_index=True), pd.DataFrame(dm_rows)


def evt_diagnostics(allf, fits, r, cfg):
    """Clustering of extremes before/after GARCH filtering, and threshold choices."""
    test = allf[(allf.method == "gjr_evt") & (allf.confidence == 0.99)]
    raw_losses = -test.actual_log_return.to_numpy()
    # mean is tiny next to sigma, so loss/sigma is a fair shock scale here
    shocks = raw_losses / test.sigma.to_numpy()
    rows = []
    for q in (0.95, 0.975, 0.99):
        rows.append({"series": "raw portfolio loss", "threshold_quantile": q,
                     "extremal_index": extremal_index(raw_losses, np.quantile(raw_losses, q))})
        rows.append({"series": "GJR-GARCH standardised shock", "threshold_quantile": q,
                     "extremal_index": extremal_index(shocks, np.quantile(shocks, q))})
    choice = fits.eqd_threshold_quantile.value_counts().sort_index().rename_axis("threshold_quantile")
    return pd.DataFrame(rows), choice.rename("refits").reset_index()


def current_allocation_forecast(complete, current, cfg):
    """Tomorrow's 95%/99% VaR and ES for the CURRENT allocation, with 90% bootstrap intervals."""
    r = np.log1p(complete[current.index] @ current).to_numpy()
    window = int(cfg["garch"]["window"])
    params = fit_gjr_garch(r[-window:])
    sigma = float(filter_volatility(r[-window:], params)[-1])
    z = params["residuals"]
    rows = []
    for c in (0.95, 0.99):
        for label, q in (("fixed 90% threshold", cfg["evt"]["fixed_threshold_quantile"]),
                         ("EQD threshold", eqd_threshold(-z, cfg["evt"]["eqd_candidates"], cfg["evt"]["eqd_bootstrap"],
                                                         cfg["evt"]["eqd_min_excesses"])["best"]["threshold_quantile"])):
            ci = bootstrap_evt_interval(-z, q, c, cfg["evt"]["interval_bootstrap"], scale_factor=sigma,
                                        location=-params["mu"])
            to_usd = lambda x: float(-np.expm1(-x)) * 1e7
            rows.append({"model": f"gjr_evt ({label})", "confidence": c, "threshold_quantile": q,
                         "sigma_daily": sigma, "var_usd": to_usd(ci["var_log"]),
                         "var_usd_5pct": to_usd(ci["var_log_5"]), "var_usd_95pct": to_usd(ci["var_log_95"]),
                         "es_usd": to_usd(ci["es_log"]), "es_usd_5pct": to_usd(ci["es_log_5"]),
                         "es_usd_95pct": to_usd(ci["es_log_95"]), "tail_shape": ci["shape"],
                         "data_end": str(complete.index.max().date())})
    # same interval for the raw (unfiltered) EVT on the last 1000 days, for comparison
    for c in (0.95, 0.99):
        ci = bootstrap_evt_interval(-r[-window:], 0.90, c, cfg["evt"]["interval_bootstrap"])
        to_usd = lambda x: float(-np.expm1(-x)) * 1e7
        rows.append({"model": "raw evt (no filter, 90% threshold)", "confidence": c, "threshold_quantile": 0.90,
                     "sigma_daily": np.nan, "var_usd": to_usd(ci["var_log"]), "var_usd_5pct": to_usd(ci["var_log_5"]),
                     "var_usd_95pct": to_usd(ci["var_log_95"]), "es_usd": to_usd(ci["es_log"]),
                     "es_usd_5pct": to_usd(ci["es_log_5"]), "es_usd_95pct": to_usd(ci["es_log_95"]),
                     "tail_shape": ci["shape"], "data_end": str(complete.index.max().date())})
    return pd.DataFrame(rows)


def yearly_breaches(allf, c=0.99):
    f = allf[allf.confidence == c].assign(year=lambda d: d.date.dt.year)
    table = f.pivot_table(index="year", columns="method", values="exception", aggfunc="sum")
    table["expected"] = f.groupby("year").date.nunique() * (1 - c)
    return table


def fmt(x, d=3):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{d}g}" if abs(x) < 1e-3 else f"{x:.{d}f}"
    return str(x)


def write_report(table, multinomial, mcs, dm, ext_index, eqd_choice, current, rebuild_error, yearly, out):
    lines = ["# Layer 3-5 upgrade: results tables", "",
             "Generated by `python -m experiments.layers345_upgrade.run`. Evaluation: 2,212 days, "
             "2018-01-02 to 2026-10-20, frozen 2007-2017 allocation (same as the baseline). "
             "p < 0.05 means the test rejects the model. Read `REVIEW.md` for the interpretation.", "",
             "Baseline rebuild check (max |VaR99 difference| vs saved baseline, log units): "
             + ", ".join(f"{k} {v:.1e}" for k, v in rebuild_error.items()), ""]
    for c in sorted(table.confidence.unique()):
        f = table[table.confidence == c].sort_values("mean_fz0")
        lines += [f"## {c:.0%} confidence", "",
                  "| Model | Family | Breaches | Kupiec p | Indep. p | Joint p | DQ p | Duration p | DE-UES p | DE-CES p | MF ES p | Mean FZ0 | Avg VaR $ | Avg ES $ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, x in f.iterrows():
            lines.append(f"| {x.method} | {x.family} | {int(x.exceptions)} | {fmt(x.kupiec_p_value)} | "
                         f"{fmt(x.independence_p_value)} | {fmt(x.conditional_coverage_p_value)} | {fmt(x.dq_p)} | "
                         f"{fmt(x.dur_p)} | {fmt(x.get('de_ues_p', np.nan))} | {fmt(x.get('de_ces_p', np.nan))} | "
                         f"{fmt(x.mf_p)} | {x.mean_fz0:.4f} | {x.avg_var_usd:,.0f} | {x.avg_es_usd:,.0f} |")
        m = mcs[mcs.confidence == c]
        lines += ["", f"Model Confidence Set (90%) on FZ0 loss: **{', '.join(m[m.in_mcs].model)}**", ""]
    lines += ["## Multinomial ES backtest (Kratz-Lok-McNeil, levels 97.5/98.125/98.75/99.375%)", "",
              "| Model | Observed (none .. worst bin) | Expected | Pearson p | Light |", "|---|---|---|---:|---|"]
    for _, x in multinomial.sort_values("mn_pearson_p", ascending=False).iterrows():
        lines.append(f"| {x.method} | {x.mn_observed} | {x.mn_expected} | {fmt(x.mn_pearson_p)} | {x.mn_light} |")
    lines += ["", "## Diebold-Mariano on FZ0 (negative = model beats reference)", "",
              "| Confidence | Model | Reference | Mean diff | DM stat | p |", "|---:|---|---|---:|---:|---:|"]
    for _, x in dm[dm.model.isin(NEW)].iterrows():
        lines.append(f"| {x.confidence:.0%} | {x.model} | {x.reference} | {x.dm_mean_diff:.5f} | {x.dm_stat:.2f} | {fmt(x.dm_p)} |")
    lines += ["", "## Clustering of extremes (extremal index, 1 = no clustering)", "",
              "| Series | Threshold | Extremal index |", "|---|---:|---:|"]
    for _, x in ext_index.iterrows():
        lines.append(f"| {x.series} | {x.threshold_quantile:.1%} | {x.extremal_index:.2f} |")
    lines += ["", "EQD threshold choices across monthly refits: "
              + ", ".join(f"{q:.2f}: {n}" for q, n in eqd_choice.itertuples(index=False)), "",
              "## Next-day risk for the CURRENT allocation, with 90% bootstrap intervals", "",
              "| Model | Conf. | VaR $ | VaR 90% interval | ES $ | ES 90% interval | Tail shape |",
              "|---|---:|---:|---|---:|---|---:|"]
    for _, x in current.iterrows():
        lines.append(f"| {x.model} | {x.confidence:.0%} | {x.var_usd:,.0f} | {x.var_usd_5pct:,.0f} - {x.var_usd_95pct:,.0f} | "
                     f"{x.es_usd:,.0f} | {x.es_usd_5pct:,.0f} - {x.es_usd_95pct:,.0f} | {x.tail_shape:.3f} |")
    header = "| Year | " + " | ".join(yearly.columns) + " |"
    lines += ["", "## 99% breaches by year", "", header, "|" + "---:|" * (len(yearly.columns) + 1)]
    for year, x in yearly.iterrows():
        lines.append(f"| {year} | " + " | ".join(f"{v:.0f}" if k != "expected" else f"{v:.1f}" for k, v in x.items()) + " |")
    lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="reuse saved GARCH-family forecasts")
    args = parser.parse_args()
    root = Path.cwd()
    base_cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    out = HERE / "results"
    allf, fits, rebuild_error, r, complete, current, run = collect_forecasts(root, base_cfg, cfg, args.reuse)
    table, multinomial = run_tests(allf, cfg)
    mcs, dm = run_scoring(allf, cfg)
    ext_index, eqd_choice = evt_diagnostics(allf, fits, r, cfg)
    cur = current_allocation_forecast(complete, current, cfg)
    yearly = yearly_breaches(allf)
    table.to_csv(out / "backtests_all.csv", index=False)
    multinomial.to_csv(out / "multinomial_es_backtest.csv", index=False)
    mcs.to_csv(out / "model_confidence_set.csv", index=False)
    dm.to_csv(out / "diebold_mariano.csv", index=False)
    ext_index.to_csv(out / "extremal_index.csv", index=False)
    eqd_choice.to_csv(out / "eqd_threshold_choices.csv", index=False)
    cur.to_csv(out / "current_allocation_next_day.csv", index=False)
    yearly.to_csv(out / "yearly_breaches_99.csv")
    write_report(table, multinomial, mcs, dm, ext_index, eqd_choice, cur, rebuild_error, yearly, out)
    print("done:", out / "RESULTS.md")


if __name__ == "__main__":
    main()
