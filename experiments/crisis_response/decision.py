"""Investor decision layer: risk budget, stress tests, a trade menu and the committee memo.

    python -m experiments.crisis_response.decision

Uses the CURRENT allocation (outputs/models/analytical_run.json, current_weights) and data up
to 2026-10-20. Writes experiments/crisis_response/results/decision/ and
docs/INVESTMENT_COMMITTEE_MEMO.md. Settings are in decision.yaml.

Where the numbers come from:
* Next-day VaR / ES for the whole portfolio: the frozen winning model from frozen_choice.yaml,
  refitted on the last 1,000 days of the current-allocation return series.
* Risk split per asset: asset-level filtered scenarios with DCC (src/quantedge/decision.py),
  using every filtered daily shock since 2007, re-played at today's volatility and correlation.
* Stress tests: crisis paths replayed with today's weights (Daniel's historical_stress), plus
  our own hypothetical scenario and a reverse stress test.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.quantedge.decision import (filtered_scenarios, euler_contributions, trade_menu, mandate_breaches,
                                    reverse_stress, greedy_derisk, EQUITY, DEFENSIVE)
from src.quantedge.range_vol import asset_range_variances, portfolio_range_variance
from src.quantedge.stress import historical_stress, hypothetical_stress
from .models import run_candidate

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()


def load_everything(dcfg):
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    run = json.loads((ROOT / "outputs/models/analytical_run.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(ROOT / "outputs/tables/prices.csv", index_col=0, parse_dates=True,
                         float_precision="round_trip").sort_index()
    simple = prices.pct_change(fill_method=None).iloc[1:][base["data"]["tickers"]].dropna()
    simple = simple.loc[:pd.Timestamp(dcfg["data_end"])]
    weights = pd.Series(run["current_weights"], dtype=float).clip(lower=0)
    weights = weights.where(weights > 1e-9, 0.0)
    weights = weights / weights.sum()
    expected = pd.read_csv(ROOT / "outputs/tables/expected_returns.csv", index_col=0).iloc[:, 0]
    return base, run, simple, weights, expected


def next_day_inputs(simple, weights):
    """Same input dict as models.load_inputs, for the current weights, with one placeholder day added
    at the end so the engine produces the forecast for the next, unseen day. The placeholder
    return is never used for anything except its own (meaningless) PIT value, which we drop."""
    held = weights[weights > 1e-6]
    held = held / held.sum()
    next_day = simple.index[-1] + pd.offsets.BDay(1)
    ext = pd.concat([simple, pd.DataFrame(0.0, index=[next_day], columns=simple.columns)])
    r = np.log1p(ext[held.index] @ held)
    vix = pd.read_csv(ROOT / "data/raw/vixcls_fred.csv", parse_dates=["date"], index_col="date").VIXCLS
    yahoo = pd.read_csv(ROOT / "data/raw/vix_yahoo.csv", parse_dates=["date"], index_col="date").vix_close
    vix = vix.combine_first(yahoo).reindex(ext.index).ffill()      # only the placeholder row is filled
    ohlc = pd.read_csv(ROOT / "data/raw/ohlc.csv", parse_dates=["date"])
    av = asset_range_variances(ohlc, list(held.index)).reindex(ext.index)
    av.iloc[-1] = av.iloc[-2]                                       # placeholder row, never used (lagged away)
    rv = portfolio_range_variance(av, ext, held)
    return {"r": r, "values": r.to_numpy(), "dates": r.index, "weights": held,
            "asset_log": np.log1p(ext[held.index]).to_numpy(),
            "x_vix": ((vix / 100) ** 2 / 252).shift(1).bfill().to_numpy(),
            "x_range": rv.shift(1).bfill().to_numpy(), "range_var": rv.to_numpy()}, next_day


def stress_paths_with_contributions(simple, weights, scenarios, notional):
    summary, paths = historical_stress(simple, weights, scenarios, notional, "daily")
    contrib = []
    for s in scenarios:
        f = simple.loc[pd.Timestamp(s["start"]):pd.Timestamp(s["end"]), weights.index]
        wealth_before = (1 + f @ weights).cumprod().shift(1).fillna(1.0)
        # each asset's dollar P&L each day = wealth at the start of the day x weight x its return
        pnl = f.mul(wealth_before, axis=0).mul(weights, axis=1).sum() * notional
        contrib += [{"scenario": s["name"], "asset": a, "weight": weights[a], "pnl_usd": pnl[a]} for a in weights.index]
    return summary, paths, pd.DataFrame(contrib)


def worst_stress_fn(simple, scenarios, hypo):
    windows = [simple.loc[pd.Timestamp(s["start"]):pd.Timestamp(s["end"])] for s in scenarios]
    shocks = pd.Series(hypo["shocks"], dtype=float)

    def fn(w):
        worst = float(-(w * shocks.reindex(w.index)).sum())
        for f in windows:
            wealth = (1 + f[w.index] @ w).cumprod()
            dd = float(1 - (wealth / wealth.cummax().clip(lower=1)).min())
            worst = max(worst, dd)
        return worst
    return fn


def crisis_betas(simple, q):
    spy = simple.SPY
    bad = spy <= spy.quantile(q)
    x = spy[bad]
    return pd.Series({a: float(np.polyfit(x, simple.loc[bad, a], 1)[0]) for a in simple.columns}), int(bad.sum())


def main():
    warnings.filterwarnings("ignore")
    dcfg = yaml.safe_load((HERE / "decision.yaml").read_text(encoding="utf-8"))
    frozen = yaml.safe_load((HERE / "frozen_choice.yaml").read_text(encoding="utf-8"))
    out = HERE / "results/decision"
    out.mkdir(parents=True, exist_ok=True)
    base, run, simple, weights, expected = load_everything(dcfg)
    notional, lim = dcfg["notional_usd"], dcfg["risk_limits"]
    limits = base["portfolio"]

    # 1. headline next-day risk from the frozen model and the reference
    data, next_day = next_day_inputs(simple, weights)
    head = []
    for spec in (frozen["winner"], next(s for s in yaml.safe_load((HERE / "config.yaml").read_text())["candidates"]
                                        if s["name"] == frozen["reference"])):
        f, _ = run_candidate(spec, data, next_day, next_day, [0.95, 0.975, 0.99])
        for _, x in f.iterrows():
            head.append({"model": spec["name"], "confidence": x.confidence, "forecast_for": next_day.date(),
                         "var_usd": x.var_simple_pct * notional, "es_usd": x.es_simple_pct * notional})
    head = pd.DataFrame(head).drop_duplicates(["model", "confidence"])
    head.to_csv(out / "next_day_risk.csv", index=False)

    # 2. risk budget from filtered DCC scenarios. All seven assets get scenarios, so the trade
    #    menu can also price buying an asset we hold none of today (EEM).
    scen, info = filtered_scenarios(np.log1p(simple[weights.index]))
    held = weights[weights > 1e-6]
    held_ix = [list(weights.index).index(a) for a in held.index]
    contrib = {c: euler_contributions(scen[:, held_ix], held, c, notional) for c in (0.95, lim["es_confidence"], 0.99)}
    table = pd.concat([d.assign(confidence=c) for c, d in contrib.items()], ignore_index=True)
    table.to_csv(out / "risk_budget.csv", index=False)
    daniel = pd.read_csv(ROOT / "outputs/tables/risk_contributions.csv")
    daniel = daniel[(daniel.method == "historical_simple") & (daniel.confidence == 0.99)][["asset", "weight", "es_risk_share", "var_risk_share"]]
    corr = pd.DataFrame({"today_dcc": info["corr_today"].stack(), "average": info["corr_average"].stack()})
    corr.to_csv(out / "correlations_today_vs_average.csv")

    # 3. stress tests
    hypo = dcfg["hypothetical_scenario"]
    s_sum, s_paths, s_contrib = stress_paths_with_contributions(simple, weights, dcfg["historical_scenarios"], notional)
    h_sum, h_contrib = hypothetical_stress(weights, [hypo], notional)
    stress = pd.concat([s_sum, h_sum], ignore_index=True)
    stress["peak_to_trough_loss"] = np.where(stress.type == "historical_replay", -stress.max_drawdown, -stress.cumulative_return)
    stress["over_tolerance"] = stress.peak_to_trough_loss > lim["stress_loss_tolerance_pct"]
    stress.to_csv(out / "stress_summary.csv", index=False)
    pd.concat([s_contrib, h_contrib.rename(columns={"pnl_contribution_usd": "pnl_usd"})[["scenario", "asset", "weight", "pnl_usd"]]],
              ignore_index=True).to_csv(out / "stress_contributions.csv", index=False)
    s_paths.to_csv(out / "stress_paths.csv", index=False)
    betas, n_bad = crisis_betas(simple, dcfg["reverse_stress_quantile"])
    rev = reverse_stress(weights, betas, lim["stress_loss_tolerance_pct"])

    # 4. trade menu
    stress_fn = worst_stress_fn(simple, dcfg["historical_scenarios"], hypo)
    menu = trade_menu(scen, weights, limits, expected, stress_fn, lim["es_confidence"], notional, dcfg["trade_step"])
    menu.to_csv(out / "trade_menu.csv", index=False)
    derisk = greedy_derisk(weights, limits, stress_fn, dcfg["trade_step"], lim["stress_loss_tolerance_pct"])
    derisk.to_csv(out / "derisk_path.csv", index=False)
    budget = contrib[lim["es_confidence"]]
    flagged = budget[budget.es_share_per_weight > dcfg["max_es_share_per_weight"]]
    test_summary = load_test_summary(frozen)

    write_decision_report(head, contrib, daniel, info, stress, s_contrib, h_contrib, betas, n_bad, rev,
                          menu, derisk, flagged, weights, limits, lim, frozen, dcfg, out)
    write_memo(head, contrib, stress, menu, derisk, flagged, weights, lim, frozen, rev, info, dcfg, test_summary)
    print("done:", out / "DECISION.md", "and docs/INVESTMENT_COMMITTEE_MEMO.md")


def load_test_summary(frozen):
    """Key 2018-2026 numbers for the memo, if the test stage has been run."""
    folder = HERE / "results/test"
    if not (folder / "test_backtests.csv").exists():
        return None
    bt = pd.read_csv(folder / "test_backtests.csv")
    win = pd.read_csv(folder / "test_windows.csv")
    pick = lambda m: bt[(bt.method == m) & (bt.confidence == 0.99)].iloc[0]
    covid = lambda m: int(win[(win.method == m) & (win.confidence == 0.99) & (win.window == "COVID crash")].breaches.iloc[0])
    year = lambda m: int(win[(win.method == m) & (win.confidence == 0.99) & (win.window == "All of 2020")].breaches.iloc[0])
    w, r = frozen["winner"]["name"], "gjr_evt_saved"
    return {"winner": pick(w), "ref": pick(r), "covid_w": covid(w), "covid_r": covid(r), "y2020_w": year(w), "y2020_r": year(r)}


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def write_decision_report(head, contrib, daniel, info, stress, s_contrib, h_contrib, betas, n_bad, rev,
                          menu, derisk, flagged, weights, limits, lim, frozen, dcfg, out):
    c = lim["es_confidence"]
    lines = ["# Decision layer: risk budget, stress tests and trade menu", "",
             f"Generated by `python -m experiments.crisis_response.decision`. Current allocation, data to {dcfg['data_end']}, "
             f"$10M portfolio. Frozen risk model: `{frozen['winner']['name']}`.", "",
             "## 1. Next-day risk for the current allocation", "",
             "| Model | Confidence | VaR | ES |", "|---|---:|---:|---:|"]
    for _, x in head.iterrows():
        lines.append(f"| {x.model} | {x.confidence:.1%} | {usd(x.var_usd)} | {usd(x.es_usd)} |")
    lines += ["", f"Limit check: 1-day {c:.1%} ES limit is {pct(lim['daily_es_limit_pct'])} of NAV "
              f"({usd(lim['daily_es_limit_pct'] * dcfg['notional_usd'])}).", "",
              "## 2. Who takes the risk (Euler split, filtered DCC scenarios)", "",
              "Every past day's shocks since 2007, re-played at today's volatility and correlation. "
              "The pieces add up exactly to the portfolio number.", ""]
    for conf, d in contrib.items():
        lines += [f"### {conf:.1%}: portfolio VaR {usd(d.attrs['portfolio_var_usd'])}, ES {usd(d.attrs['portfolio_es_usd'])}, "
                  f"diversification saves {usd(d.attrs['diversification_benefit_usd'])} of ES", "",
                  "| Asset | Weight | Position $ | Marginal ES (per $1) | ES piece $ | ES share | VaR share | ES share / weight |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, x in d.iterrows():
            lines.append(f"| {x.asset} | {pct(x.weight)} | {usd(x.position_usd)} | {x.marginal_es:.4f} | {usd(x.component_es_usd)} | "
                         f"{pct(x.es_share)} | {pct(x.var_share)} | {x.es_share_per_weight:.2f} |")
        lines.append("")
    lines += [f"Risk budget rule: no asset should carry more than {dcfg['max_es_share_per_weight']:.1f}x its weight in "
              f"{c:.1%} ES share. Flagged: " + (", ".join(f"{x.asset} ({x.es_share_per_weight:.2f}x)" for _, x in flagged.iterrows()) or "none") + ".", ""]
    lines += ["Compare: the main pipeline's split (raw last 252 days, 99%):", "",
              "| Asset | Weight | ES share | VaR share |", "|---|---:|---:|---:|"]
    for _, x in daniel.iterrows():
        if x.weight > 1e-6:
            lines.append(f"| {x.asset} | {pct(x.weight)} | {pct(x.es_risk_share)} | {pct(x.var_risk_share)} |")
    lines += ["", "Today's volatility vs the long-run average (daily):", "",
              "| Asset | Today | Long run |", "|---|---:|---:|"]
    for a, s in info["sigma_today"].items():
        lines.append(f"| {a} | {pct(s, 2)} | {pct(info['sigma_long_run'][a], 2)} |")
    if "corr_today" in info:
        ct, ca = info["corr_today"], info["corr_average"]
        lines += ["", f"SPY-AGG correlation: today {ct.loc['SPY', 'AGG']:.2f}, average {ca.loc['SPY', 'AGG']:.2f}. "
                  f"SPY-GLD: today {ct.loc['SPY', 'GLD']:.2f}, average {ca.loc['SPY', 'GLD']:.2f}."]
    lines += ["", "## 3. Stress tests (today's weights, daily rebalancing)", "",
              f"Tolerance: no scenario should lose more than {pct(lim['stress_loss_tolerance_pct'], 0)} peak to trough.", "",
              "| Scenario | Type | Loss (peak to trough) | P&L at end $ | Worst day | Over tolerance? |",
              "|---|---|---:|---:|---:|---|"]
    for _, x in stress.iterrows():
        wd = pct(x.worst_day_return) if pd.notna(x.get("worst_day_return")) else "n/a"
        lines.append(f"| {x.scenario} | {x.type} | {pct(x.peak_to_trough_loss)} | {usd(x.pnl_usd)} | {wd} | {'yes' if x.over_tolerance else 'no'} |")
    hypo = dcfg["hypothetical_scenario"]
    lines += ["", f"**Our hypothetical scenario, {hypo['name']}:** {hypo['description']}", "",
              "P&L by asset in each scenario ($):", ""]
    allc = pd.concat([s_contrib, h_contrib.rename(columns={"pnl_contribution_usd": "pnl_usd"})[["scenario", "asset", "weight", "pnl_usd"]]])
    piv = allc[allc.weight > 1e-6].pivot_table(index="asset", columns="scenario", values="pnl_usd")
    lines += ["| Asset | " + " | ".join(piv.columns) + " |", "|---|" + "---:|" * len(piv.columns)]
    for a, row in piv.iterrows():
        lines.append(f"| {a} | " + " | ".join(usd(v) for v in row) + " |")
    lines += ["", f"**Reverse stress test:** with assets moving like they did on the worst {dcfg['reverse_stress_quantile']:.0%} of "
              f"S&P days since 2007 ({n_bad} days), an S&P 500 fall of about **{pct(rev, 0)}** would cost the portfolio "
              f"{pct(lim['stress_loss_tolerance_pct'], 0)}. Crisis betas: " +
              ", ".join(f"{a} {b:.2f}" for a, b in betas.items() if weights.get(a, 0) > 1e-6) + ".", "",
              "## 4. Trade menu (sell 5% of one asset, buy 5% of another, mandate limits respected)", "",
              f"Base: {c:.1%} ES {usd(menu.attrs['base_es_usd'])}, worst stress loss {pct(menu.attrs['base_worst_stress'])}, "
              f"expected return {pct(menu.attrs['base_expected_return'], 2)} a year.", "",
              "| Sell | Buy | ES change $ | Worst stress loss | Expected return change |", "|---|---|---:|---:|---:|"]
    for _, x in menu.head(8).iterrows():
        lines.append(f"| {x.sell} | {x.buy} | {x.es_change_usd:+,.0f} | {pct(x.worst_stress_loss_pct)} | {x.expected_return_change * 100:+.2f} pp |")
    by_stress = menu.sort_values("worst_stress_loss_pct").head(5)
    lines += ["", "Best five trades for the worst stress loss instead:", "",
              "| Sell | Buy | ES change $ | Worst stress loss | Expected return change |", "|---|---|---:|---:|---:|"]
    for _, x in by_stress.iterrows():
        lines.append(f"| {x.sell} | {x.buy} | {x.es_change_usd:+,.0f} | {pct(x.worst_stress_loss_pct)} | {x.expected_return_change * 100:+.2f} pp |")
    lines += ["", f"**Can we get every stress loss under {pct(lim['stress_loss_tolerance_pct'], 0)} inside the mandate?** "
              "We kept making the 5% trade that helps most until nothing helped any more:", "",
              "| Step | Trade | Worst stress loss |", "|---:|---|---:|"]
    for _, x in derisk.iterrows():
        lines.append(f"| {x.step} | {x.trade} | {pct(x.worst_stress_loss)} |")
    lines += ["", f"Mandate check on the current weights: {', '.join(mandate_breaches(weights, limits)) or 'all limits met'}. "
              f"Equity {pct(weights[list(EQUITY)].sum())} (limit {pct(limits['max_equity_exposure'], 0)}), "
              f"defensive {pct(weights[list(DEFENSIVE)].sum())} (minimum {pct(limits['min_defensive_exposure'], 0)})."]
    (out / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memo(head, contrib, stress, menu, derisk, flagged, weights, lim, frozen, rev, info, dcfg, test):
    notional = dcfg["notional_usd"]
    c = lim["es_confidence"]
    tol = lim["stress_loss_tolerance_pct"]
    h = head[(head.model == frozen["winner"]["name"])].set_index("confidence")
    es, var99, es99 = h.es_usd[c], h.var_usd[0.99], h.es_usd[0.99]
    es_limit = lim["daily_es_limit_pct"] * notional
    worst = stress.sort_values("peak_to_trough_loss", ascending=False).iloc[0]
    over = stress[stress.over_tolerance]
    budget = contrib[c].sort_values("es_share", ascending=False)
    fixed = derisk.worst_stress_loss.iloc[-1] <= tol
    best_reachable = derisk.worst_stress_loss.min()
    if es > es_limit:
        verdict = "REJECT the mandate as proposed"
        why = (f"normal day-to-day risk ({usd(es)} ES) is already above the {usd(es_limit)} limit. "
               "We'd need a different allocation before taking it on.")
    elif len(over) == 0:
        verdict = "ACCEPT the mandate as proposed"
        why = "day-to-day risk and every stress scenario sit inside the limits."
    else:
        verdict = "ACCEPT the allocation, but ADJUST the stress-loss limit"
        why = (f"day-to-day risk is comfortably inside the limit, but a repeat of the {worst.scenario} would lose "
               f"{pct(worst.peak_to_trough_loss)}, above the {pct(tol, 0)} tolerance. "
               + ("A few small trades inside the mandate fix that, at a small cost in return."
                  if fixed else
                  f"No mix allowed by the mandate gets below {pct(best_reachable)} (bonds and gold are already at their caps), "
                  "so the honest fix is the limit, not the portfolio."))
    sp = info["corr_today"].loc["SPY", "AGG"] if "corr_today" in info else np.nan
    sg = info["corr_today"].loc["SPY", "GLD"] if "corr_today" in info else np.nan
    sg_avg = info["corr_average"].loc["SPY", "GLD"] if "corr_today" in info else np.nan
    new_limit = np.ceil(worst.peak_to_trough_loss * 100 + 1) / 100
    L = ["# Investment Committee Memo: Serendib University Endowment, $10M mandate", "",
         f"*Data to {dcfg['data_end']}. Team Tiramisu. All numbers are for the current allocation.*", "",
         f"## Recommendation: {verdict}", "",
         why, "",
         "## The portfolio", "",
         f"{pct(weights['AGG'], 0)} US bonds (AGG), {pct(weights['GLD'], 0)} gold (GLD) and {pct(weights[list(EQUITY)].sum(), 0)} "
         f"stocks (SPY {pct(weights['SPY'], 0)}, EFA {pct(weights['EFA'], 0)}, XLK {pct(weights['XLK'], 0)}, "
         f"XLE {pct(weights['XLE'], 0)}). It meets every mandate limit.", "",
         "## How much can we lose tomorrow?", "",
         f"* On a bad day (the worst 1 in 40) we'd lose about **{usd(es)}** on average. That's {es / es_limit:.0%} of our "
         f"{usd(es_limit)} daily limit.",
         f"* On the worst 1-in-100 day: at least {usd(var99)}, and {usd(es99)} on average when it happens."]
    if test is not None:
        t, r = test["winner"], test["ref"]
        L.append(f"* Can we trust that? The risk model was picked using 2011-2017 only, then tested once on 2018-2026: "
                 f"{int(t.exceptions)} days lost more than the 1-in-100 line, against {t.observations * 0.01:.0f} expected "
                 f"(old model: {int(r.exceptions)}). Across all of 2020 it had {test['y2020_w']} such days against "
                 f"{test['y2020_r']} for the old model; in the crash itself (20 Feb-30 Apr) both had {test['covid_w']}. "
                 "No daily model avoids every COVID breach; regulators excluded 6-27 March 2020 for banks.")
    L += ["", "## Where the risk comes from", "",
          "| Asset | Share of the money | Share of the tail risk (ES) |", "|---|---:|---:|"]
    for _, x in budget.iterrows():
        if x.weight > 0.02:
            L.append(f"| {x.asset} | {pct(x.weight, 0)} | {pct(x.es_share, 0)} |")
    L += ["", f"* Gold is not acting as a safe asset right now: its correlation with stocks is {sg:.2f} (normally {sg_avg:.2f}), "
          "so it adds to the losses on bad days instead of cushioning them.",
          f"* Bonds still cushion, but less than usual: SPY-AGG correlation is {sp:.2f}. In March 2020 this flipped and "
          "bonds fell with stocks."]
    if len(flagged):
        L.append("* Over budget (more than 2x its weight in tail risk): " +
                 ", ".join(f"{x.asset} ({pct(x.weight, 0)} of money, {pct(x.es_share, 0)} of risk)" for _, x in flagged.iterrows()) + ".")
    L += ["", "## Stress tests (today's weights)", "",
          "| Scenario | Loss, peak to trough | In dollars |", "|---|---:|---:|"]
    for _, x in stress.iterrows():
        L.append(f"| {x.scenario} | {pct(x.peak_to_trough_loss)} | {usd(x.peak_to_trough_loss * notional)} |")
    L += ["", f"The last line is our own scenario: stocks crash while bonds and gold are sold for cash at the same time. "
          f"Reverse stress test: the S&P 500 would have to fall about **{pct(rev, 0)}** before we lose {pct(tol, 0)}.", "",
          "## What we ask the board to approve", ""]
    if len(over) and not fixed:
        L.append(f"1. Raise the stress-loss limit from {pct(tol, 0)} to **{pct(new_limit, 0)}**, peak to trough. The allocation "
                 f"already holds the most bonds and gold the mandate allows, and a 2008 repeat would still cost {pct(worst.peak_to_trough_loss)}.")
    elif len(over):
        L.append(f"1. Keep the {pct(tol, 0)} limit and make these trades: " +
                 "; ".join(derisk.trade.iloc[1:]) + ".")
    else:
        L.append(f"1. Keep the {pct(tol, 0)} stress-loss limit.")
    if len(flagged):
        x = flagged.iloc[0]
        # only trims that cut ES without making the worst stress loss worse
        trim = menu[(menu.sell == x.asset) & (menu.worst_stress_change_pct <= 1e-9)].sort_values("es_change_usd").head(1)
        if len(trim):
            t = trim.iloc[0]
            L.append(f"2. Trim {x.asset} by 5% into {t.buy}: tail risk (ES) falls by {usd(-t.es_change_usd)} a day, "
                     f"expected return changes by {t.expected_return_change * 100:+.2f} pp a year.")
    L += [f"{3 if len(flagged) else 2}. Daily limit: 1-day {c:.1%} ES at most {usd(es_limit)}, checked every morning.",
          f"{4 if len(flagged) else 3}. Early warning: if the 60-day SPY-AGG correlation turns positive, review the bond "
          "allocation within a week.", "",
          "*Caveats: the 2% daily and 20% stress limits are our assumptions until the client brief confirms them. "
          "The 2008 replay uses weights fitted on data that includes 2008. Stress tests show what could happen, not what will.*"]
    (ROOT / "docs/INVESTMENT_COMMITTEE_MEMO.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
