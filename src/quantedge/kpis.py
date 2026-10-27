import numpy as np
import pandas as pd
from .utils import annualized_return


def cumulative_wealth(returns: pd.Series) -> pd.Series:
    return (1 + returns.dropna()).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    wealth = cumulative_wealth(returns)
    return float((wealth / wealth.cummax().clip(lower=1) - 1).min()) if not wealth.empty else float("nan")


def compute_kpis(portfolio_returns: pd.Series, benchmark_returns: pd.Series, market_returns: pd.Series, risk_free_annual: float, periods_per_year: int = 252) -> dict:
    frame = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark"), market_returns.rename("market")], axis=1).dropna()
    if frame.empty:
        raise ValueError("No aligned observations for KPI calculation")
    p, b, m = frame["portfolio"], frame["benchmark"], frame["market"]
    ann = annualized_return(p, periods_per_year)
    vol = float(p.std(ddof=1) * np.sqrt(periods_per_year))
    downside = float(np.sqrt(np.mean(np.minimum(p - ((1 + risk_free_annual) ** (1 / periods_per_year) - 1), 0) ** 2) * periods_per_year))
    beta = float(p.cov(m) / m.var()) if m.var() > 0 else np.nan
    market_ann = annualized_return(m, periods_per_year)
    mdd = max_drawdown(p)
    active = p - b
    tracking = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    return {"annualized_return": ann, "annualized_volatility": vol, "sharpe_ratio": (ann-risk_free_annual)/vol if vol>0 else np.nan, "sortino_ratio": (ann-risk_free_annual)/downside if downside>0 else np.nan, "treynor_ratio": (ann-risk_free_annual)/beta if beta and np.isfinite(beta) else np.nan, "jensens_alpha": ann-(risk_free_annual+beta*(market_ann-risk_free_annual)) if np.isfinite(beta) else np.nan, "information_ratio": float(active.mean()*periods_per_year/tracking) if tracking>0 else np.nan, "max_drawdown": mdd, "calmar_ratio": ann/abs(mdd) if mdd else np.nan, "beta": beta, "observations": len(frame)}
