import numpy as np
import pandas as pd


def annualized_historical_returns(simple_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    return (1 + simple_returns.mean()) ** periods_per_year - 1


def market_proxy_returns(simple_returns: pd.DataFrame, market_weights: dict[str, float]) -> pd.Series:
    weights = pd.Series(market_weights, dtype=float).reindex(simple_returns.columns).fillna(0)
    if weights.sum() <= 0:
        raise ValueError("Market proxy weights must sum to a positive value")
    active = weights.ne(0)
    return simple_returns.mul(weights / weights.sum()).sum(axis=1, min_count=int(active.sum()))


def capm_expected_returns(simple_returns: pd.DataFrame, market_returns: pd.Series, risk_free_annual: float, periods_per_year: int = 252) -> pd.Series:
    aligned = simple_returns.join(market_returns.rename("market"), how="inner").dropna()
    market = aligned["market"]
    market_var = market.var(ddof=1) * periods_per_year
    if market_var <= 0:
        raise ValueError("Market proxy variance must be positive")
    market_ann = (1 + market.mean()) ** periods_per_year - 1
    return pd.Series({asset: risk_free_annual + (aligned[asset].cov(market) * periods_per_year / market_var) * (market_ann - risk_free_annual) for asset in simple_returns.columns})


def shrink_expected_returns(historical_mu: pd.Series, prior_mu: pd.Series, alpha: float = .5) -> pd.Series:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * historical_mu + (1 - alpha) * prior_mu.reindex(historical_mu.index)


def annualized_covariance(log_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return log_returns.cov() * periods_per_year


def shrink_covariance_toward_diagonal(cov: pd.DataFrame, shrinkage: float = .25) -> pd.DataFrame:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    diagonal = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
    shrunk = (1 - shrinkage) * cov + shrinkage * diagonal
    return (shrunk + shrunk.T) / 2
