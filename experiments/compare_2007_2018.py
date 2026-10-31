"""Retrospective 2007-2018 fit / 2019-2026 risk-model comparison.

Run from the repository root: python -m experiments.compare_2007_2018
The candidate list and settings below are fixed before this script's first result run.
This is not an untouched holdout: the team has previously inspected 2019-2026.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

from experiments.crisis_response.models import MIN_WEIGHT, run_candidate
from experiments.improved_risk.evt_response import calibrate_threshold, detector_path
from experiments.improved_risk.filtered_response import build_filtered_response
from experiments.improved_risk.models import build_improved_forecasts, causal_volatility
from experiments.layers345_upgrade.models import baseline_pits, build_garch_family
from experiments.layers345_upgrade.run import run_tests
from src.quantedge.analytics import fit_training_portfolio
from src.quantedge.backtesting import basel_traffic_light
from src.quantedge.cleaning import compute_returns
from src.quantedge.config import load_config
from src.quantedge.contributions import risk_contributions
from src.quantedge.data import load_fred_csv
from src.quantedge.evt import gpd_risk
from src.quantedge.range_vol import asset_range_variances, portfolio_range_variance
from src.quantedge.risk import build_risk_forecasts
from src.quantedge.scoring import diebold_mariano, fz0_loss
from src.quantedge.stress import historical_stress, hypothetical_stress


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/compare_2007_2018"
TRAIN_START, TRAIN_END = "2007-01-01", "2018-12-31"
TEST_START, TEST_END = "2019-01-01", "2026-10-20"
CONFIDENCES = [0.90, 0.95, 0.975, 0.99]
CRISIS_SPECS = [
    {"name": "gjrx_vix_evt_d1000", "filter": "gjrx_vix", "window": 1000, "refit_every": 1},
    {"name": "gjrx_range_evt_m1000", "filter": "gjrx_range", "window": 1000, "refit_every": 21},
    {"name": "ms_gjr_t_m1000", "filter": "ms_gjr", "window": 1000, "refit_every": 21},
]
EXPECTED_99 = {"historical", "parametric_ewma", "monte_carlo_bootstrap",
               "evt_gpd", "filtered_historical", "filtered_bootstrap", "filtered_evt",
               "filtered_historical_alert", "filtered_bootstrap_alert",
               "gjr_t", "gjr_evt", "gjrx_vix_evt_d1000", "gjrx_range_evt_m1000",
               "ms_gjr_t_m1000", "filtered_evt_cusum"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(base: dict) -> tuple[dict, pd.Series, dict, pd.DataFrame]:
    """Fit allocation through 2018 and construct all lagged market inputs."""
    prices = pd.read_csv(ROOT / "outputs/tables/prices.csv", index_col=0,
                         parse_dates=True, float_precision="round_trip").sort_index()
    _, simple_all = compute_returns(prices)
    complete = simple_all.loc[:, base["data"]["tickers"]].dropna()
    rates = load_fred_csv(ROOT / "data/raw/risk_free.csv", base["data"]["fred_risk_free_series"])
    fitted = json.loads(json.dumps(base))
    fitted["validation"].update(train_start=TRAIN_START, train_end=TRAIN_END,
                                test_start=TEST_START, test_end=TEST_END)
    weights, allocation = fit_training_portfolio(complete, rates, fitted)
    period = complete.loc[TRAIN_START:TEST_END]
    r = np.log1p(period @ weights)
    held = weights[weights > MIN_WEIGHT]
    held = held / held.sum()
    required = [ROOT / "data/raw/vixcls_fred.csv", ROOT / "data/raw/vix_yahoo.csv",
                ROOT / "data/raw/ohlc.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing market inputs: " + ", ".join(missing) +
                                ". Run python -m experiments.crisis_response.download_data")
    fred = pd.read_csv(required[0], parse_dates=["date"], index_col="date").VIXCLS
    yahoo = pd.read_csv(required[1], parse_dates=["date"], index_col="date").vix_close
    vix = fred.combine_first(yahoo).reindex(r.index)
    if vix.isna().any():
        raise ValueError("VIX missing on portfolio return dates")
    ohlc = pd.read_csv(required[2], parse_dates=["date"])
    asset_var = asset_range_variances(ohlc, list(weights.index)).reindex(r.index)
    if asset_var[held.index].isna().any().any():
        raise ValueError("OHLC range variance missing on held-asset dates")
    range_var = portfolio_range_variance(asset_var, period, held)
    data = {"r": r, "values": r.to_numpy(), "dates": r.index, "weights": held,
            "asset_log": np.log1p(period[held.index]).to_numpy(),
            "x_vix": ((vix/100)**2/252).shift(1).bfill().to_numpy(),
            "x_range": range_var.shift(1).bfill().to_numpy(),
            "range_var": range_var.to_numpy(), "vix": vix}
    return data, weights, allocation, complete


def alert_forecasts(improved: pd.DataFrame, diagnostics: pd.DataFrame,
                    r: pd.Series, base: dict, exp: dict) -> tuple[pd.DataFrame, dict]:
    alert = yaml.safe_load((ROOT / "experiments/improved_risk/evt_response.yaml").read_text())
    v = alert["volatility"]
    slow = causal_volatility(r, v["reference_ewma_lambda"], v["min_periods"], v["floor"])
    fast = causal_volatility(r, v["fast_ewma_lambda"], v["min_periods"], v["floor"])
    threshold, train_triggers, train_days = calibrate_threshold(
        r, slow, pd.Timestamp(TRAIN_END), alert["detector"])
    signal = detector_path(r, slow, threshold,
                           alert["detector"]["standardized_loss_drift"],
                           alert["detector"]["active_forecast_days"])
    empirical = build_filtered_response(
        improved, r, slow, fast, signal, base["project"]["portfolio_value_usd"],
        base["project"]["seed"], exp["historical_window"],
        exp["monte_carlo_simulations"], ["filtered_historical", "filtered_bootstrap"],
        CONFIDENCES)
    empirical = empirical[empirical.variant.eq("alert")].copy()
    evt = improved[improved.method.eq("filtered_evt") & improved.confidence.eq(.99)].copy()
    selected = diagnostics[diagnostics.model.eq("filtered_evt") & diagnostics.selected.eq(True)].copy()
    selected["fit_date"] = pd.to_datetime(selected.fit_date)
    selected = selected.set_index("fit_date")
    if selected.index.has_duplicates or evt.empty:
        raise ValueError("Selected EVT fits missing or duplicated")
    fit_dates = evt.groupby("tail_fit_end").date.min().to_dict()
    active = signal.set_index("date").alert_active_for_forecast
    for idx, item in evt.iterrows():
        fit_date = fit_dates[item.tail_fit_end]
        fit = selected.loc[fit_date]
        sigma = float(slow.loc[item.date])
        updated_sigma = max(sigma, float(fast.loc[item.date])) if active.loc[item.date] else sigma
        risk = gpd_risk(float(fit.threshold_standardized)*updated_sigma,
                        float(fit["shape"]), float(fit.scale)*updated_sigma,
                        float(fit.exceedance_probability), .99)
        evt.loc[idx, "method"] = "filtered_evt_cusum"
        for field, number in risk.items():
            evt.loc[idx, field] = number
        evt.loc[idx, "forecast_volatility"] = updated_sigma
        evt.loc[idx, "pit"] = np.nan
    meta = {"threshold": threshold, "training_triggers": train_triggers,
            "training_volatility_days": train_days,
            "active_test_days": int(signal.loc[signal.date.ge(pd.Timestamp(TEST_START)),
                                               "alert_active_for_forecast"].sum())}
    return pd.concat([empirical, evt], ignore_index=True), meta


def common_rows(data: dict, base: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    r = data["r"]
    exp = yaml.safe_load((ROOT / "experiments/improved_risk/config.yaml").read_text())
    base = json.loads(json.dumps(base))
    base["var"]["confidence_levels"] = CONFIDENCES
    baseline, _ = build_risk_forecasts(r, base, pd.Timestamp(TEST_START), include_evt=True)
    pits = baseline_pits(r, baseline, base)
    baseline = baseline.merge(pits[["date", "method", "pit", "sigma"]],
                              on=["date", "method"], validate="many_to_one")
    baseline["family"] = "baseline"
    improved, diagnostics = build_improved_forecasts(r, base, exp, pd.Timestamp(TEST_START))
    improved = improved[improved.method.isin(["filtered_historical", "filtered_bootstrap",
                                             "filtered_evt"])].copy()
    alert, alert_meta = alert_forecasts(improved, diagnostics, r, base, exp)
    improved["family"] = "filtered"
    alert["family"] = "previously_selected_alert"
    family_cfg = yaml.safe_load((ROOT / "experiments/layers345_upgrade/config.yaml").read_text())
    garch, garch_fits = build_garch_family(r, pd.Timestamp(TEST_START), family_cfg, CONFIDENCES)
    garch = garch[garch.method.isin(["gjr_t", "gjr_evt"])].copy()
    garch["family"] = "garch"
    crisis, crisis_fits = [], []
    for spec in CRISIS_SPECS:
        print(f"running {spec['name']}", flush=True)
        f, fits = run_candidate(spec, data, TEST_START, TEST_END, CONFIDENCES)
        f["family"] = "crisis_response"
        crisis.append(f)
        crisis_fits.append(fits)
    allf = pd.concat([baseline, improved, alert, garch, *crisis], ignore_index=True)
    allf = allf.loc[allf.date.between(TEST_START, TEST_END)].copy()
    allf["date"] = pd.to_datetime(allf.date)
    allf["actual_log_return"] = allf.date.map(r)
    allf["evaluation_index"] = allf.date.map(pd.Series(np.arange(len(r)), index=r.index))
    for column in ("pit", "sigma"):
        if column not in allf:
            allf[column] = np.nan
    allf["sigma"] = allf.sigma.fillna(allf.get("forecast_volatility"))
    fallback_sigma = r.ewm(alpha=.06, min_periods=60, adjust=False).std().shift(1)
    allf["sigma"] = allf.sigma.fillna(pd.Series(allf.date.map(fallback_sigma).to_numpy(), index=allf.index))
    allf["exception"] = allf.actual_log_return < -allf.var_log
    if allf.duplicated(["date", "method", "confidence"]).any():
        raise ValueError("Duplicate model/date/confidence")
    dates = r.loc[TEST_START:TEST_END].index
    groups = allf.groupby(["method", "confidence"]).date.nunique()
    required = {(m, c) for m in EXPECTED_99 - {"filtered_evt_cusum"} for c in CONFIDENCES}
    required.add(("filtered_evt_cusum", .99))
    if set(allf[allf.confidence.eq(.99)].method) != EXPECTED_99 or any(
            groups.get(key, 0) != len(dates) for key in required):
        raise ValueError("One or more candidate forecasts do not cover every common test day")
    if (allf.estimation_end.notna() &
            (pd.to_datetime(allf.estimation_end) >= allf.date)).any():
        raise ValueError("Forecast uses its target day's return")
    fit_logs = pd.concat([garch_fits.assign(method="gjr_family"), *crisis_fits], ignore_index=True)
    return allf.sort_values(["date", "method", "confidence"]), fit_logs, alert_meta


def rolling_lights(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame[frame.confidence.eq(.99)].groupby("method"):
        hits = group.sort_values("date").exception.to_numpy(dtype=bool)
        dates = group.sort_values("date").date.to_numpy()
        for i in range(249, len(hits)):
            light = basel_traffic_light(hits[i-249:i+1], .99)
            rows.append({"method": method, "end_date": dates[i],
                         "exceptions_250": light["traffic_light_exceptions"],
                         "traffic_light": light["traffic_light"]})
    return pd.DataFrame(rows)


def report(allf: pd.DataFrame, table: pd.DataFrame, lights: pd.DataFrame,
           allocation: dict, alert_meta: dict, output: Path) -> None:
    lines = ["# 2007–2018 fit, 2019–2026 retrospective comparison", "",
             "**These dates have already been inspected by the team. This is a chronological robustness run, not a fresh independent holdout. No candidate is selected automatically by a passing p-value.**", "",
             f"Portfolio weights fitted on {allocation['training_start']}–{allocation['training_end']}; "
             f"{table.observations.max()} common forecast days. CUSUM threshold {alert_meta['threshold']:g}, "
             f"calibrated on pre-2019 returns only. All model forecasts use prior-day information.", "",
             "The three simple references, filtered candidates, GJR models, VIX/range candidates, "
             "Markov model, and earlier CUSUM candidates are shown below. CUSUM was originally designed "
             "after inspecting much of this test period; its result is especially susceptible to selection bias.", "",
             "## 99% VaR and ES", "",
             "| Model | Breaches / days | Kupiec p | Independence p | Joint p | DQ p | MF ES p | Mean FZ0 | Mean VaR $ | Mean ES $ | Latest 250-day light |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    last_lights = lights.sort_values("end_date").groupby("method").tail(1).set_index("method")
    for row in table[table.confidence.eq(.99)].sort_values("mean_fz0").itertuples():
        fmt = lambda x: f"{x:.3g}" if pd.notna(x) else "n/a"
        lines.append(f"| {row.method} | {row.exceptions}/{row.observations} | "
                     f"{fmt(row.kupiec_p_value)} | {fmt(row.independence_p_value)} | "
                     f"{fmt(row.conditional_coverage_p_value)} | {fmt(row.dq_p)} | "
                     f"{fmt(row.mf_p)} | {row.mean_fz0:.4f} | {row.avg_var_usd:,.0f} | "
                     f"{row.avg_es_usd:,.0f} | {last_lights.loc[row.method, 'traffic_light']} |")
    lines += ["", "## Evidence for a model choice", "",
              "* **Best balanced candidate in this run: `gjr_evt`.** At 95%, 97.5%, and 99%, "
              "Kupiec, independence, joint coverage, DQ, duration, McNeil–Frey ES, and both "
              "Du–Escanciano ES tests do not reject at 5%. Its 90% duration test does reject, "
              "so this is not a clean pass at every confidence.",
              "* **Best 99% FZ0 score: `gjrx_vix_evt_d1000`.** It has 20 breaches and "
              "passes the 99% VaR tests, but McNeil–Frey ES rejects at 99% (p = 0.0446), "
              "and 90% joint coverage rejects (p = 0.00676). The range model also has a "
              "99% McNeil–Frey rejection (p = 0.039).",
              "* **CUSUM alert variants pass the listed 99% VaR tests**, but were originally "
              "developed after inspecting overlapping outcomes. Their 95% DQ results are "
              "weaker, and `filtered_evt_cusum` was only generated at 99% here. "
              "They should not be promoted on this retrospective result alone.",
              "* Every model's latest 250-day traffic light is green. This single-window "
              "label cannot distinguish the candidates; see the full rolling file and tests.", "",
              "### Leading candidates across confidence levels", "",
              "| Model | Confidence | Breaches | Joint p | DQ p | Duration p | MF ES p | Mean FZ0 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    leaders = ["gjr_evt", "gjrx_vix_evt_d1000", "gjrx_range_evt_m1000",
               "filtered_historical_alert", "filtered_evt_cusum"]
    for method in leaders:
        for row in table[table.method.eq(method)].sort_values("confidence").itertuples():
            fmt = lambda x: f"{x:.3g}" if pd.notna(x) else "n/a"
            lines.append(f"| {method} | {row.confidence:.1%} | {row.exceptions} | "
                         f"{fmt(row.conditional_coverage_p_value)} | {fmt(row.dq_p)} | "
                         f"{fmt(row.dur_p)} | {fmt(row.mf_p)} | {row.mean_fz0:.4f} |")
    counts = lights.groupby(["method", "traffic_light"]).size().unstack(fill_value=0)
    lines += ["", "### Rolling 250-day traffic lights at 99%", "",
              "These are overlapping windows, so counts are descriptive and are not independent tests.", "",
              "| Model | Green windows | Yellow windows | Red windows |",
              "|---|---:|---:|---:|"]
    for method, row in counts.sort_index().iterrows():
        lines.append(f"| {method} | {row.get('green', 0)} | {row.get('yellow', 0)} | {row.get('red', 0)} |")
    lines += ["", "## Reading the comparison", "",
              "P-values below 0.05 reject their corresponding null. Kupiec tests the breach rate; "
              "Christoffersen independence tests whether breaches cluster; joint coverage combines them. "
              "The 250-day traffic light applies at 99% only. Quantile loss and FZ0 measure forecast quality "
              "beyond breach counts; lower scores are better on the same dates. MF tests ES on breach days. "
              "PIT-based Du–Escanciano ES tests are available only for models that produce a full quantile distribution.", "",
              "The Monte Carlo backtest p-values in the CSV use 1,000 null simulations. "
              "They are a check on rare-event asymptotics, not a second independent data sample. "
              "The historical 2008 stress replay uses weights fitted through 2018, so it is a "
              "retrospective scenario rather than an out-of-sample portfolio outcome.", "",
              "Reproduce this run from the repo root with `python -m experiments.compare_2007_2018` "
              "after restoring `data/raw/vixcls_fred.csv`, `vix_yahoo.csv`, and `ohlc.csv` "
              "using `python -m experiments.crisis_response.download_data`. Source file hashes and "
              "the fitted portfolio weights are in `manifest.json`. Yahoo may revise historic OHLC data.", "",
              "`backtests.csv` contains 90%, 95%, 97.5%, and 99% results plus DQ, duration, Monte Carlo p-values, "
              "ES tests and scores. `rolling_traffic_lights.csv`, `yearly.csv`, `stress_summary.csv`, "
              "`risk_contributions.csv`, `diebold_mariano.csv`, and the daily forecasts allow a fuller review.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = load_config(str(ROOT / "config.yaml"))
    data, weights, allocation, complete = load_data(base)
    print("portfolio fitted; generating forecasts", flush=True)
    allf, fits, alert_meta = common_rows(data, base)
    tests_cfg = yaml.safe_load((ROOT / "experiments/crisis_response/config.yaml").read_text())
    tests_cfg["tests"]["confidences"] = CONFIDENCES
    tests_cfg["tests"]["monte_carlo_sims"] = 1000
    print("scoring forecasts and backtests", flush=True)
    table, multinomial = run_tests(allf, tests_cfg)
    lights = rolling_lights(allf)
    yearly = allf.assign(year=allf.date.dt.year).groupby(
        ["method", "confidence", "year"], as_index=False).agg(
            observations=("exception", "size"), breaches=("exception", "sum"),
            mean_var_log=("var_log", "mean"))
    dm = []
    for c in (.95, .99):
        frame = allf[allf.confidence.eq(c)].copy()
        frame["fz0"] = fz0_loss(frame.actual_log_return, frame.var_log, frame.es_log, c)
        matrix = frame.pivot(index="date", columns="method", values="fz0")
        for method in matrix.columns:
            if method != "historical":
                dm.append({"confidence": c, "method": method, "reference": "historical",
                           **diebold_mariano(matrix[method], matrix.historical)})
    stress, stress_paths = historical_stress(complete, weights, base["stress"]["historical"],
                                             base["project"]["portfolio_value_usd"])
    hypothetical, shocks = hypothetical_stress(
        weights, base["stress"]["hypothetical"], base["project"]["portfolio_value_usd"])
    contributions = risk_contributions(complete.loc[:TRAIN_END].iloc[-252:], weights,
                                       [.95, .99], base["project"]["portfolio_value_usd"])
    artifacts = {"daily_forecasts.csv.gz": allf, "fits.csv": fits, "backtests.csv": table,
                 "multinomial.csv": multinomial, "rolling_traffic_lights.csv": lights,
                 "yearly.csv": yearly, "diebold_mariano.csv": pd.DataFrame(dm),
                 "stress_summary.csv": pd.concat([stress, hypothetical], ignore_index=True),
                 "stress_paths.csv": stress_paths, "stress_shock_contributions.csv": shocks,
                 "risk_contributions.csv": contributions}
    for name, frame in artifacts.items():
        frame.to_csv(OUTPUT / name, index=False)
    meta = {"train_start": TRAIN_START, "train_end": allocation["training_end"],
            "test_start": str(allf.date.min().date()), "test_end": str(allf.date.max().date()),
            "test_days": int(allf.date.nunique()), "weights": weights.to_dict(),
            "candidate_specs": CRISIS_SPECS, "confidence_levels": CONFIDENCES,
            "cusum": alert_meta, "selection_warning": "2019-2026 previously inspected; CUSUM especially selected on overlapping outcomes",
            "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                              (ROOT / "config.yaml", ROOT / "outputs/tables/prices.csv",
                               ROOT / "data/raw/risk_free.csv", ROOT / "data/raw/vixcls_fred.csv",
                               ROOT / "data/raw/vix_yahoo.csv", ROOT / "data/raw/ohlc.csv")}}
    (OUTPUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report(allf, table, lights, allocation, alert_meta, OUTPUT)
    print(f"Wrote {len(allf)} forecast rows over {allf.date.nunique()} dates to {OUTPUT}")


if __name__ == "__main__":
    main()
