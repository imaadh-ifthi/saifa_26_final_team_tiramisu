import numpy as np
import pandas as pd
from .utils import annualized_return


def cumulative_wealth(returns: pd.Series) -> pd.Series:
    return (1 + returns.dropna()).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    wealth = cumulative_wealth(returns)
    return float((wealth / wealth.cummax().clip(lower=1) - 1).min()) if not wealth.empty else float("nan")


def compute_kpis(portfolio_returns: pd.Series, benchmark_returns: pd.Series, market_returns: pd.Series, risk_free_annual: float | pd.Series, periods_per_year: int = 252) -> dict:
    frame = pd.concat([portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark"), market_returns.rename("market")], axis=1).dropna()
    if frame.empty:
        raise ValueError("No aligned observations for KPI calculation")
    p, b, m = frame["portfolio"], frame["benchmark"], frame["market"]
    if isinstance(risk_free_annual, pd.Series):
        rf = risk_free_annual.reindex(frame.index)
        if rf.isna().any() or not np.isfinite(rf).all() or (rf <= -1).any():
            raise ValueError("Risk-free series must cover every KPI observation")
        daily_rf = (1+rf)**(1/periods_per_year)-1
        annual_rf = annualized_return(daily_rf, periods_per_year)
    else:
        annual_rf = float(risk_free_annual)
        daily_rf = (1+annual_rf)**(1/periods_per_year)-1
    ann = annualized_return(p, periods_per_year)
    vol = float(p.std(ddof=1) * np.sqrt(periods_per_year))
    downside = float(np.sqrt(np.mean(np.minimum(p - daily_rf, 0) ** 2) * periods_per_year))
    beta = float(p.cov(m) / m.var()) if m.var() > 0 else np.nan
    market_ann = annualized_return(m, periods_per_year)
    mdd = max_drawdown(p)
    active = p - b
    tracking = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    return {"annualized_return": ann, "annualized_volatility": vol, "sharpe_ratio": (ann-annual_rf)/vol if vol>0 else np.nan, "sortino_ratio": (ann-annual_rf)/downside if downside>0 else np.nan, "treynor_ratio": (ann-annual_rf)/beta if beta and np.isfinite(beta) else np.nan, "jensens_alpha": ann-(annual_rf+beta*(market_ann-annual_rf)) if np.isfinite(beta) else np.nan, "information_ratio": float(active.mean()*periods_per_year/tracking) if tracking>0 else np.nan, "max_drawdown": mdd, "calmar_ratio": ann/abs(mdd) if mdd else np.nan, "beta": beta, "observations": len(frame), "annualized_risk_free": annual_rf}
