"""Historical return-path replay and explicitly assumed hypothetical shocks."""
import numpy as np
import pandas as pd
from .kpis import max_drawdown


def historical_stress(simple_returns: pd.DataFrame, weights: pd.Series, scenarios: list[dict],
                      portfolio_value: float, rebalance: str = "daily") -> tuple[pd.DataFrame, pd.DataFrame]:
    if rebalance not in ("daily", "buy_and_hold"):
        raise ValueError("Stress rebalance must be daily or buy_and_hold")
    if (not np.isfinite(portfolio_value) or portfolio_value <= 0
            or not np.isclose(weights.sum(), 1) or not np.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError("Stress testing requires a positive value and long-only fully invested weights")
    data = simple_returns.loc[:, weights.index]
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Stress returns require a sorted unique index")
    summaries, paths = [], []
    for scenario in scenarios:
        start, end = pd.Timestamp(scenario["start"]), pd.Timestamp(scenario["end"])
        if start > end:
            raise ValueError("Stress scenario start must precede end")
        source = data.loc[start:end]
        frame = source.dropna()
        row = {"scenario": scenario["name"], "type": "historical_replay",
               "requested_start": start, "requested_end": end, "rebalance": rebalance,
               "allocation_basis": "current_allocation_retrospective_replay"}
        if frame.empty:
            summaries.append({**row, "status": "missing_data", "observations": 0})
            continue
        if not np.isfinite(frame.to_numpy()).all() or (frame <= -1).any().any():
            raise ValueError("Invalid historical simple returns")
        if rebalance == "daily":
            returns = frame @ weights
            wealth = (1+returns).cumprod()
        else:
            wealth = (1+frame).cumprod() @ weights
            returns = wealth.pct_change(fill_method=None)
            returns.iloc[0] = wealth.iloc[0]-1
        # Seven calendar days tolerate weekend/holiday endpoints, not missing crisis months.
        partial = (frame.index.min() > start+pd.Timedelta(days=7) or
                   frame.index.max() < end-pd.Timedelta(days=7) or len(frame) != len(source))
        total = float(wealth.iloc[-1]-1)
        summaries.append({**row, "status": "partial_data" if partial else "ok",
                          "actual_start": frame.index.min(), "actual_end": frame.index.max(),
                          "observations": len(frame), "cumulative_return": total,
                          "pnl_usd": total*portfolio_value,
                          "max_drawdown": max_drawdown(returns), "worst_day_return": returns.min(),
                          "ending_value_usd": float(wealth.iloc[-1]*portfolio_value)})
        peak = wealth.cummax().clip(lower=1)
        paths.extend({"scenario": scenario["name"], "date": date, "portfolio_return": returns.loc[date],
                      "wealth": wealth.loc[date], "value_usd": wealth.loc[date]*portfolio_value,
                      "drawdown": wealth.loc[date]/peak.loc[date]-1} for date in frame.index)
    return pd.DataFrame(summaries), pd.DataFrame(paths)


def hypothetical_stress(weights: pd.Series, scenarios: list[dict], portfolio_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, contributions = [], []
    for scenario in scenarios:
        shocks = pd.Series(scenario["shocks"], dtype=float).reindex(weights.index)
        if shocks.isna().any() or not np.isfinite(shocks).all() or (shocks < -1).any():
            raise ValueError("Every asset needs a finite hypothetical simple-return shock >= -100%")
        pnl = weights*shocks
        total = float(pnl.sum())
        summaries.append({"scenario": scenario["name"], "type": "hypothetical_one_period",
                          "status": "assumed_shocks", "description": scenario.get("description", ""),
                          "cumulative_return": total, "pnl_usd": total*portfolio_value,
                          "ending_value_usd": (1+total)*portfolio_value})
        contributions.extend({"scenario": scenario["name"], "asset": asset, "weight": weights[asset],
                              "shock": shocks[asset], "return_contribution": pnl[asset],
                              "pnl_contribution_usd": pnl[asset]*portfolio_value} for asset in weights.index)
    return pd.DataFrame(summaries), pd.DataFrame(contributions)
