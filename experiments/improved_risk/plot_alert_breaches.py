"""Plot saved 99% VaR breaches for three alert candidates and their references."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAIRS = (
    ("Filtered EVT", "evt_response", "filtered_evt_reference",
     "evt_response", "filtered_evt_cusum"),
    ("Filtered historical", "filtered_response", "filtered_historical_reference",
     "filtered_response", "filtered_historical_alert"),
    ("Monte Carlo, filtered", "filtered_response", "filtered_bootstrap_reference",
     "filtered_response", "filtered_bootstrap_alert"),
    ("Monte Carlo, raw to alert", None, "monte_carlo_bootstrap",
     "filtered_response", "filtered_bootstrap_alert"),
)


def load_comparison(results: Path) -> tuple[list[tuple], pd.DataFrame]:
    groups = []
    detail = []
    shared_dates = None
    shared_losses = None
    for label, baseline_folder, baseline_name, candidate_folder, candidate_name in PAIRS:
        baseline_source = results / baseline_folder if baseline_folder else results
        candidate_source = results / candidate_folder
        baseline_file = (baseline_source / "daily_forecasts.csv.gz" if baseline_folder
                         else baseline_source / "forecasts.csv.gz")
        baseline_rows = pd.read_csv(baseline_file, parse_dates=["date"])
        candidate_rows = pd.read_csv(candidate_source / "daily_forecasts.csv.gz", parse_dates=["date"])
        baseline = baseline_rows[baseline_rows.confidence.eq(.99) &
                                 baseline_rows.method.eq(baseline_name)].sort_values(
                                     "date").reset_index(drop=True)
        candidate = candidate_rows[candidate_rows.confidence.eq(.99) &
                                   candidate_rows.method.eq(candidate_name)].sort_values(
                                       "date").reset_index(drop=True)
        baseline_tests = pd.read_csv(baseline_source / "backtests.csv")
        candidate_tests = pd.read_csv(candidate_source / "backtests.csv")
        if baseline.date.duplicated().any() or candidate.date.duplicated().any():
            raise ValueError(f"Duplicate 99% forecast for {label}")
        if len(baseline) != 2212 or not baseline.date.equals(candidate.date):
            raise ValueError(f"Different evaluation dates for {label}")
        if not np.allclose(baseline.actual_loss_usd, candidate.actual_loss_usd,
                           atol=.01, rtol=1e-10):
            raise ValueError(f"Different realized losses for {label}")
        if shared_dates is None:
            shared_dates = baseline.date
            shared_losses = baseline.actual_loss_usd
        elif not shared_dates.equals(baseline.date) or not np.allclose(
                shared_losses, baseline.actual_loss_usd, atol=.01, rtol=1e-10):
            raise ValueError("Methods do not share the same dates and losses")
        for name, frame, tests in ((baseline_name, baseline, baseline_tests),
                                   (candidate_name, candidate, candidate_tests)):
            row = tests[tests.method.eq(name) & tests.confidence.eq(.99)]
            if len(row) != 1 or int(row.iloc[0].exceptions) != int(frame.exception.sum()):
                raise ValueError(f"Saved backtest count differs from plotted breaches for {name}")
        avoided = baseline.exception.to_numpy() & ~candidate.exception.to_numpy()
        groups.append((label, baseline, candidate, avoided))
        detail.append({"family": label, "reference_model": baseline_name,
                       "alert_model": candidate_name, "observations": len(baseline),
                       "reference_breaches": int(baseline.exception.sum()),
                       "alert_breaches": int(candidate.exception.sum()),
                       "avoided_breaches": int(avoided.sum()),
                       "new_breaches": int((~baseline.exception.to_numpy() &
                                            candidate.exception.to_numpy()).sum()),
                       "alert_kupiec_p": float(candidate_tests.loc[
                           candidate_tests.method.eq(candidate_name) & candidate_tests.confidence.eq(.99),
                           "kupiec_p_value"].iloc[0]),
                       "alert_independence_p": float(candidate_tests.loc[
                           candidate_tests.method.eq(candidate_name) & candidate_tests.confidence.eq(.99),
                           "independence_p_value"].iloc[0]),
                       "alert_combined_p": float(candidate_tests.loc[
                           candidate_tests.method.eq(candidate_name) & candidate_tests.confidence.eq(.99),
                           "conditional_coverage_p_value"].iloc[0])})
    return groups, pd.DataFrame(detail)


def plot(groups: list[tuple], output: Path, *, zoom: bool) -> None:
    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 12 if zoom else 15),
                             sharex=True, sharey=True, layout="constrained")
    if zoom:
        start, end = pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-20")
    for row_index, (label, baseline, candidate, avoided) in enumerate(groups):
        for col_index, (frame, subtitle) in enumerate(((baseline, "Original"),
                                                       (candidate, "Next-day alert"))):
            ax = axes[row_index, col_index]
            active = frame.date.between(start, end) if zoom else np.ones(len(frame), dtype=bool)
            visible = frame.loc[active]
            ax.plot(visible.date, visible.actual_loss_usd / 1e6,
                    color="#666666", lw=1.0 if zoom else .55, label="Realized loss")
            ax.plot(visible.date, visible.var_usd / 1e6,
                    color="#1769aa", lw=1.5 if zoom else .8, label="99% VaR")
            breaches = visible[visible.exception]
            ax.scatter(breaches.date, breaches.actual_loss_usd / 1e6,
                       color="#d43828", s=34 if zoom else 16, zorder=4,
                       label="Breach")
            if col_index == 1:
                kept = frame.loc[active & avoided]
                ax.scatter(kept.date, kept.actual_loss_usd / 1e6,
                           facecolors="none", edgecolors="#15803d", linewidths=1.7,
                           s=58 if zoom else 34, zorder=5,
                           label="Original breach avoided")
            ax.set_title(f"{label} — {subtitle}: {int(frame.exception.sum())} total breaches")
            ax.set_ylabel("USD millions")
            ax.grid(alpha=.2)
    handles = [Line2D([0], [0], color="#666666", label="Realized loss"),
               Line2D([0], [0], color="#1769aa", label="99% VaR"),
               Line2D([0], [0], marker="o", linestyle="", color="#d43828",
                      label="Breach"),
               Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                      markeredgecolor="#15803d", color="#15803d",
                      label="Original breach avoided")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.suptitle("99% VaR: original models and next-day alert variants" +
                 (" — February–March 2020" if zoom else " — 2018–2026"))
    for ax in axes[-1]:
        ax.set_xlabel("Evaluation date")
    if zoom:
        for ax in axes[-1]:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    groups, summary = load_comparison(args.results)
    output = args.results / "alert_breach_comparison"
    output.mkdir(parents=True, exist_ok=True)
    plot(groups, output / "99pct_breaches_all_dates.png", zoom=False)
    plot(groups, output / "99pct_breaches_march_2020.png", zoom=True)
    summary.to_csv(output / "summary.csv", index=False)
    lines = ["# Three alert candidates and their 99% VaR breaches", "",
             "[Full evaluation chart](99pct_breaches_all_dates.png) | [February-March 2020 detail](99pct_breaches_march_2020.png)", "",
             "The first three rows compare one candidate with its unchanged reference. The fourth shows the original raw Monte Carlo bootstrap beside the same filtered-bootstrap alert candidate (shown again for clarity). There are three distinct new candidates, not four. All panels use the same 2,212 dates and realized losses. Red dots are VaR breaches. Hollow green circles mark reference breaches that the alert variant avoided. The plots use the same vertical scale so forecast size is directly comparable.", "",
             "| Family | Original breaches | Alert breaches | Avoided | New | Alert Kupiec p | Alert independence p | Alert combined p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.family} | {r.reference_breaches} | {r.alert_breaches} | "
                     f"{r.avoided_breaches} | {r.new_breaches} | "
                     f"{r.alert_kupiec_p:.4f} | {r.alert_independence_p:.4f} | "
                     f"{r.alert_combined_p:.4f} |")
    lines += ["", "The three distinct alert candidates are not rejected by these three **99% VaR** tests at 5% on the previously inspected development period. This is not independent validation. Raw EVT with daily scaling plus an alert is excluded because it still rejects independence (p about 0.018). ES is not assessed by these VaR tests.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved side-by-side breach images to {output}")


if __name__ == "__main__":
    main()
