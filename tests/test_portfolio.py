import numpy as np
import pandas as pd
from src.quantedge.portfolio import optimize_target_return_min_variance


def test_optimizer_respects_basic_constraints():
    tickers = ["A", "B", "C"]
    mu = pd.Series({"A": .04, "B": .08, "C": .12})
    covariance = pd.DataFrame(np.diag([.01, .02, .04]), index=tickers, columns=tickers)
    weights = optimize_target_return_min_variance(mu, covariance, .065, tickers, [(0, .8)] * 3)
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()
    assert float(mu @ weights) >= .065 - 1e-6
    assert np.allclose(covariance, covariance.T)
    assert (np.diag(covariance) >= 0).all()
