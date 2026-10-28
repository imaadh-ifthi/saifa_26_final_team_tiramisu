"""Run the crisis-response experiment in two separate steps.

From the repository root:
    python -m experiments.crisis_response.download_data      # once, saves VIX and OHLC to data/raw/
    python -m experiments.crisis_response.run --stage dev    # 2011-2017 only, picks and freezes a winner
    python -m experiments.crisis_response.run --stage test   # one evaluation on 2018-01-02 .. 2026-10-20

The dev step writes frozen_choice.yaml. The test step refuses to run without it, and refuses
if config.yaml has changed since the choice was frozen. That way the 2018-2026 numbers can't
feed back into the choice.
"""
import argparse
import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from experiments.layers345_upgrade.run import run_tests, run_scoring
from src.quantedge.model_selection import apply_selection_rule
from src.quantedge.scoring import fz0_loss, diebold_mariano, model_confidence_set
from .models import load_inputs, run_candidate

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
RESULTS = HERE / "results"
FROZEN = HERE / "frozen_choice.yaml"
CODE_FILES = ["experiments/crisis_response/models.py", "src/quantedge/garch_x.py", "src/quantedge/range_vol.py",
              "src/quantedge/dcc.py", "src/quantedge/ms_garch.py", "src/quantedge/garch.py",
              "src/quantedge/tail_grid.py", "src/quantedge/evt_extra.py"]
KEEP = ["date", "method", "confidence", "var_log", "es_log", "var_simple_pct", "es_simple_pct",
        "actual_log_return", "evaluation_index", "pit", "sigma"]
NOTIONAL = 10_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(args):
    spec, start, end, confidences = args
    warnings.filterwarnings("ignore")
    base_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = load_inputs(ROOT, base_cfg)
    forecasts, fits = run_candidate(spec, data, start, end, confidences)
    return spec["name"], forecasts, fits


def run_all(specs, start, end, confidences):
    jobs = [(s, start, end, confidences) for s in specs]
    out, fits = [], []
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, f, fl in pool.map(_worker, jobs):
            print(f"  finished {name}: {f.date.nunique()} days", flush=True)
            out.append(f)
            fits.append(fl)
    return pd.concat(out, ignore_index=True), pd.concat(fits, ignore_index=True)


def test_config(cfg, multinomial_confidence=0.99):
    tc = dict(cfg["tests"])
    tc["confidences"] = [multinomial_confidence]   # run_tests uses the first level to pick one row per day
    return {"tests": tc}


def finish_frame(f, family):
    f = f.copy()
    f["family"] = family
    f["exception"] = f.actual_log_return < -f.var_log
    return f.sort_values(["method", "confidence", "date"]).reset_index(drop=True)


def window_summary(allf, windows, extra_years=None):
    rows = []
    for w in windows:
        f = allf[(allf.date >= pd.Timestamp(w["start"])) & (allf.date <= pd.Timestamp(w["end"]))]
        for (m, c), g in f.groupby(["method", "confidence"]):
            hits = g.sort_values("date").exception.to_numpy()
            runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], hits.astype(int), [0]])) != 0).reshape(-1, 2), axis=1)
            rows.append({"window": w["name"], "method": m, "confidence": c, "days": len(g),
                         "breaches": int(hits.sum()), "expected": len(g) * (1 - c),
                         "longest_breach_run": int(runs.max()) if len(runs) else 0,
                         "avg_var_usd": g.var_simple_pct.mean() * NOTIONAL,
                         "avg_es_usd": g.es_simple_pct.mean() * NOTIONAL,
                         "mean_fz0": float(fz0_loss(g.actual_log_return, g.var_log, g.es_log, c).mean())})
    return pd.DataFrame(rows)


def fmt(x, d=3):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{d}g}" if abs(x) < 1e-3 else f"{x:.{d}f}"
    return str(x)


def backtest_table_md(table, confidences, order=None):
    lines = []
    for c in confidences:
        f = table[table.confidence == c].sort_values("mean_fz0")
        if f.empty:
            continue
        lines += [f"### {c:.1%} confidence", "",
                  "| Model | Breaches (exp.) | Kupiec p | Indep. p | Joint p | DQ p | Duration p | DE-UES p | DE-CES p | MF ES p | Mean FZ0 | Avg VaR $ | Avg ES $ |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, x in f.iterrows():
            lines.append(f"| {x.method} | {int(x.exceptions)} ({x.observations * (1 - c):.0f}) | {fmt(x.kupiec_p_value)} | "
                         f"{fmt(x.independence_p_value)} | {fmt(x.conditional_coverage_p_value)} | {fmt(x.dq_p)} | "
                         f"{fmt(x.dur_p)} | {fmt(x.get('de_ues_p', np.nan))} | {fmt(x.get('de_ces_p', np.nan))} | "
                         f"{fmt(x.mf_p)} | {x.mean_fz0:.4f} | {x.avg_var_usd:,.0f} | {x.avg_es_usd:,.0f} |")
        lines.append("")
    return lines


# ---------------------------------------------------------------- development stage

def stage_dev(cfg):
    out = RESULTS / "dev"
    out.mkdir(parents=True, exist_ok=True)
    p = cfg["periods"]
    print(f"development run {p['dev_start']} .. {p['dev_end']} for {len(cfg['candidates'])} candidates")
    forecasts, fits = run_all(cfg["candidates"], p["dev_start"], p["dev_end"], cfg["confidences"])
    forecasts = finish_frame(forecasts, "crisis_candidate")
    if pd.Timestamp(forecasts.date.max()) > pd.Timestamp(p["dev_end"]):
        raise RuntimeError("a development forecast leaked past dev_end")
    forecasts.to_csv(out / "dev_forecasts.csv.gz", index=False)
    fits.to_csv(out / "dev_fits.csv", index=False)
    table, multinomial = run_tests(forecasts, test_config(cfg))
    ranking = apply_selection_rule(table, cfg["candidates"], cfg["selection"])
    winner = ranking.iloc[0]
    ref = cfg["reference"]
    dm_rows, mcs_rows = [], []
    for c in cfg["selection"]["score_confidences"]:
        f = forecasts[forecasts.confidence == c]
        loss = f.assign(loss=fz0_loss(f.actual_log_return, f.var_log, f.es_log, c)).pivot(
            index="date", columns="method", values="loss")
        for m in loss.columns:
            if m != ref:
                dm_rows.append({"confidence": c, "model": m, "reference": ref, **diebold_mariano(loss[m], loss[ref])})
        mcs = model_confidence_set(loss, cfg["tests"]["mcs_level"], cfg["tests"]["mcs_bootstrap"], cfg["tests"]["mcs_block"])
        mcs["confidence"] = c
        mcs_rows.append(mcs)
    dm, mcs = pd.DataFrame(dm_rows), pd.concat(mcs_rows, ignore_index=True)
    stress = window_summary(forecasts, cfg["report"]["dev_stress_windows"])
    for name, frame in (("dev_backtests", table), ("dev_multinomial", multinomial), ("dev_ranking", ranking),
                        ("dev_diebold_mariano", dm), ("dev_mcs", mcs), ("dev_stress_windows", stress)):
        frame.to_csv(out / f"{name}.csv", index=False)

    spec = next(s for s in cfg["candidates"] if s["name"] == winner.method)
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "winner": spec, "reference": ref,
              "dev_period": [p["dev_start"], p["dev_end"]],
              "dev_score_winner": float(winner.score),
              "dev_score_reference": float(ranking.set_index("method").score[ref]),
              "ranking": [f"{r.method}: score {r.score:.5f}, gates {r.gates_passed}/{r.gates_total}"
                          for r in ranking.itertuples()],
              "config_sha256": sha256(HERE / "config.yaml"),
              "code_sha256": {f: sha256(ROOT / f) for f in CODE_FILES},
              "dev_forecasts_sha256": sha256(out / "dev_forecasts.csv.gz"),
              "note": "Chosen on 2011-2017 only. Do not edit; the test stage checks the config hash."}
    FROZEN.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    write_dev_report(table, ranking, dm, mcs, stress, frozen, cfg, out)
    print("winner:", winner.method, "->", FROZEN)


def write_dev_report(table, ranking, dm, mcs, stress, frozen, cfg, out):
    p = cfg["periods"]
    lines = ["# Development results (2011-2017, inside the training period)", "",
             f"Generated by `python -m experiments.crisis_response.run --stage dev`. Forecast days "
             f"{p['dev_start']} to {p['dev_end']}, frozen 2007-2017 allocation, each forecast uses data up to the day before.",
             "These are the only numbers used to pick the model.", "",
             f"**Winner frozen in `frozen_choice.yaml`: `{frozen['winner']['name']}`** "
             f"(score {frozen['dev_score_winner']:.5f} vs reference {frozen['dev_score_reference']:.5f}; lower is better).", "",
             "## Ranking by the pre-registered rule", "",
             "| Rank | Model | Gates passed | Failed gates | Score (avg FZ0 95/97.5/99%) |", "|---:|---|---:|---|---:|"]
    for r in ranking.itertuples():
        lines.append(f"| {r.rank} | {r.method} | {r.gates_passed}/{r.gates_total} | {r.failed_gates} | {r.score:.5f} |")
    lines += ["", "## Diebold-Mariano vs the reference (negative = better than gjr_evt_m1000)", "",
              "| Conf. | Model | Mean diff | DM stat | p |", "|---:|---|---:|---:|---:|"]
    for _, x in dm.sort_values(["confidence", "dm_mean_diff"]).iterrows():
        lines.append(f"| {x.confidence:.1%} | {x.model} | {x.dm_mean_diff:.5f} | {x.dm_stat:.2f} | {fmt(x.dm_p)} |")
    for c in cfg["selection"]["score_confidences"]:
        m = mcs[mcs.confidence == c]
        lines += ["", f"Model Confidence Set ({c:.1%}): " + ", ".join(m[m.in_mcs].model)]
    lines += ["", "## Backtests on the development period", ""] + backtest_table_md(table, cfg["confidences"])
    lines += ["## Stress windows inside development (99%)", "",
              "| Window | Model | Days | Breaches | Expected | Longest run | Avg VaR $ |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, x in stress[stress.confidence == 0.99].sort_values(["window", "breaches"]).iterrows():
        lines.append(f"| {x.window} | {x.method} | {x.days} | {x.breaches} | {x.expected:.1f} | {x.longest_breach_run} | {x.avg_var_usd:,.0f} |")
    (out / "DEV_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- test stage

def load_saved_models(cfg, test_dates, r):
    frames = []
    ewma = r.ewm(alpha=0.06, min_periods=60, adjust=False).std().shift(1)
    for item in cfg["saved_models"]:
        f = pd.read_csv((HERE / item["file"]).resolve(), parse_dates=["date"])
        f = f[f.method.isin(item["methods"])].copy()
        for col in ("var", "es"):
            if f"{col}_simple_pct" not in f:
                f[f"{col}_simple_pct"] = f[f"{col}_usd"] / NOTIONAL
        if "pit" not in f:
            f["pit"] = np.nan
        if "sigma" not in f:
            f["sigma"] = f.date.map(ewma)
        f["method"] = f.method.replace(item.get("rename", {}))
        frames.append(f[KEEP])
    out = pd.concat(frames, ignore_index=True)
    return out[out.date.isin(test_dates)]


def stage_test(cfg):
    if not FROZEN.exists():
        raise SystemExit("frozen_choice.yaml is missing. Run --stage dev first.")
    frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != sha256(HERE / "config.yaml"):
        raise SystemExit("config.yaml changed after the choice was frozen. Refusing to run the test stage.")
    changed = [f for f, h in frozen["code_sha256"].items() if sha256(ROOT / f) != h]
    out = RESULTS / "test"
    out.mkdir(parents=True, exist_ok=True)
    p = cfg["periods"]
    winner, ref = frozen["winner"]["name"], frozen["reference"]
    print(f"test run {p['test_start']} .. {p['test_end']}; frozen winner {winner}")
    forecasts, fits = run_all(cfg["candidates"], p["test_start"], p["test_end"], cfg["confidences"])
    forecasts.to_csv(out / "test_forecasts.csv.gz", index=False)
    fits.to_csv(out / "test_fits.csv", index=False)
    warnings.filterwarnings("ignore")
    base_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = load_inputs(ROOT, base_cfg)
    test_dates = set(forecasts.date)
    saved = load_saved_models(cfg, test_dates, data["r"])
    allf = pd.concat([finish_frame(forecasts, "crisis_candidate"), finish_frame(saved, "saved_earlier_work")],
                     ignore_index=True)
    allf.loc[allf.method == winner, "family"] = "frozen_winner"
    table, multinomial = run_tests(allf, test_config(cfg))
    mcs, dm = scoring_vs(allf, cfg, [ref, "gjr_evt_saved", "filtered_historical_alert"])
    windows = window_summary(allf, cfg["report"]["test_windows"])
    yearly = yearly_table(allf, data["vix"], cfg["report"]["calm_year_vix_below"])
    breach_dates = allf[(allf.confidence == 0.99) & allf.exception & (allf.date.dt.year == 2020)]
    breach_dates = breach_dates.groupby("method").date.apply(lambda d: ", ".join(x.strftime("%m-%d") for x in d))
    for name, frame in (("test_backtests", table), ("test_multinomial", multinomial), ("test_mcs", mcs),
                        ("test_diebold_mariano", dm), ("test_windows", windows), ("test_yearly", yearly)):
        frame.to_csv(out / f"{name}.csv", index=False)
    breach_dates.to_csv(out / "breach_dates_2020_99.csv")
    write_test_report(table, multinomial, mcs, dm, windows, yearly, breach_dates, frozen, changed, cfg, out)
    print("done:", out / "RESULTS.md")


def scoring_vs(allf, cfg, references):
    tc = cfg["tests"]
    mcs_rows, dm_rows = [], []
    for c in cfg["confidences"]:
        f = allf[allf.confidence == c]
        loss = f.assign(loss=fz0_loss(f.actual_log_return, f.var_log, f.es_log, c)).pivot(
            index="date", columns="method", values="loss").dropna(axis=1)
        mcs = model_confidence_set(loss, tc["mcs_level"], tc["mcs_bootstrap"], tc["mcs_block"])
        mcs["confidence"] = c
        mcs_rows.append(mcs)
        for ref in references:
            if ref in loss:
                for m in loss.columns:
                    if m != ref:
                        dm_rows.append({"confidence": c, "model": m, "reference": ref, **diebold_mariano(loss[m], loss[ref])})
    return pd.concat(mcs_rows, ignore_index=True), pd.DataFrame(dm_rows)


def yearly_table(allf, vix, calm_below):
    f = allf.assign(year=allf.date.dt.year)
    g = f.groupby(["method", "confidence", "year"]).agg(days=("date", "size"), breaches=("exception", "sum"),
                                                        avg_var_usd=("var_simple_pct", "mean"),
                                                        avg_es_usd=("es_simple_pct", "mean")).reset_index()
    g[["avg_var_usd", "avg_es_usd"]] *= NOTIONAL
    g["expected"] = g.days * (1 - g.confidence)
    vix_year = vix.groupby(vix.index.year).mean()
    g["avg_vix"] = g.year.map(vix_year)
    g["calm_year"] = g.avg_vix < calm_below
    return g


def write_test_report(table, multinomial, mcs, dm, windows, yearly, breach_dates, frozen, changed, cfg, out):
    p = cfg["periods"]
    winner, ref = frozen["winner"]["name"], frozen["reference"]
    focus = [winner, ref, "gjr_evt_saved", "filtered_historical_alert", "filtered_evt_cusum"]
    focus = list(dict.fromkeys(focus))
    lines = ["# Test results (2018-01-02 to 2026-10-20, evaluated once)", "",
             f"Generated by `python -m experiments.crisis_response.run --stage test`. The model was frozen on "
             f"{frozen['frozen_at_utc']} using 2011-2017 only (see `frozen_choice.yaml`). Same frozen 2007-2017 "
             "allocation and the same 2,212 days as the earlier work. p < 0.05 means the test rejects the model.", "",
             f"* Frozen winner: **{winner}**. Reference: {ref} (Bhavee's gjr_evt run through the new engine); "
             "`gjr_evt_saved` is the exact saved gjr_evt forecast from experiments/layers345_upgrade.",
             "* Daniel's models (`filtered_historical_alert`, `filtered_evt_cusum`) are read from his saved files. "
             "Their alert rule was designed after looking at 2018-2026, and they only stored 90/95/99%, so ES tests "
             "that need the full distribution show n/a.",
             "* Code changed after freezing: " + (", ".join(changed) if changed else "none") + ".", ""]
    lines += ["## Main models", ""] + backtest_table_md(table[table.method.isin(focus)], cfg["confidences"])
    lines += ["## Model Confidence Set (90%) on FZ0, all models", ""]
    for c in cfg["confidences"]:
        m = mcs[mcs.confidence == c]
        lines.append(f"* {c:.1%}: " + ", ".join(m[m.in_mcs].model))
    lines += ["", "## Diebold-Mariano on FZ0 for the frozen winner (negative = winner better)", "",
              "| Conf. | Reference | Mean diff | DM stat | p |", "|---:|---|---:|---:|---:|"]
    for _, x in dm[dm.model == winner].iterrows():
        lines.append(f"| {x.confidence:.1%} | {x.reference} | {x.dm_mean_diff:.5f} | {x.dm_stat:.2f} | {fmt(x.dm_p)} |")
    lines += ["", "## Crisis windows", "",
              "| Window | Conf. | Model | Breaches | Expected | Longest run | Avg VaR $ | Avg ES $ |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    w = windows[windows.method.isin(focus) & windows.confidence.isin([0.95, 0.99])]
    for _, x in w.sort_values(["window", "confidence", "method"]).iterrows():
        lines.append(f"| {x.window} | {x.confidence:.0%} | {x.method} | {x.breaches} | {x.expected:.1f} | "
                     f"{x.longest_breach_run} | {x.avg_var_usd:,.0f} | {x.avg_es_usd:,.0f} |")
    lines += ["", "99% breach dates in 2020:", ""]
    for m in focus:
        if m in breach_dates:
            lines.append(f"* {m}: {breach_dates[m]}")
    y = yearly[yearly.method.isin(focus) & (yearly.confidence == 0.99)]
    calm = y[y.calm_year]
    calm_years = sorted(calm.year.unique())
    lines += ["", f"## Calm years (average VIX below {cfg['report']['calm_year_vix_below']}: "
              f"{', '.join(map(str, calm_years))}), 99%", "",
              "Checks that the model is not simply bigger all the time.", "",
              "| Model | Days | Breaches | Expected | Avg VaR $ | Avg ES $ |", "|---|---:|---:|---:|---:|---:|"]
    for m, g in calm.groupby("method"):
        lines.append(f"| {m} | {g.days.sum()} | {g.breaches.sum()} | {g.expected.sum():.1f} | "
                     f"{(g.avg_var_usd * g.days).sum() / g.days.sum():,.0f} | {(g.avg_es_usd * g.days).sum() / g.days.sum():,.0f} |")
    piv = y.pivot_table(index="year", columns="method", values="breaches")
    lines += ["", "## 99% breaches by year", "", "| Year | avg VIX | " + " | ".join(piv.columns) + " | expected |",
              "|---:|---:|" + "---:|" * (len(piv.columns) + 1)]
    for year, row in piv.iterrows():
        e = y[y.year == year].expected.iloc[0]
        v = y[y.year == year].avg_vix.iloc[0]
        lines.append(f"| {year} | {v:.1f} | " + " | ".join(f"{x:.0f}" for x in row) + f" | {e:.1f} |")
    lines += ["", "## Appendix: every candidate on the test period (NOT used for choosing)", "",
              "Shown so nobody has to wonder whether a different pick would have looked better or worse.", ""]
    lines += backtest_table_md(table[table.family == "crisis_candidate"], [0.99, 0.975])
    lines += ["### COVID crash window (2020-02-20 to 2020-04-30), 99%, every candidate", "",
              "| Model | Breaches | Longest run | Avg VaR $ |", "|---|---:|---:|---:|"]
    cw = windows[(windows.window == "COVID crash") & (windows.confidence == 0.99)]
    for _, x in cw.sort_values("breaches").iterrows():
        lines.append(f"| {x.method} | {x.breaches} | {x.longest_breach_run} | {x.avg_var_usd:,.0f} |")
    lines += ["", "## Multinomial ES test (Kratz-Lok-McNeil)", "",
              "| Model | Observed (none .. worst bin) | Expected | Pearson p |", "|---|---|---|---:|"]
    for _, x in multinomial.sort_values("mn_pearson_p", ascending=False).iterrows():
        lines.append(f"| {x.method} | {x.mn_observed} | {x.mn_expected} | {fmt(x.mn_pearson_p)} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["dev", "test"], required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    if args.stage == "dev":
        stage_dev(cfg)
    else:
        stage_test(cfg)


if __name__ == "__main__":
    main()
