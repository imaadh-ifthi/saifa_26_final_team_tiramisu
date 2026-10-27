import numpy as np


import pandas as pd


import pytest


from scipy.stats import norm, chi2


from src.quantedge.tail import empirical_risk, normal_risk, tail_weights


from src.quantedge.risk import build_risk_forecasts, common_evaluation_sample


from src.quantedge.kpis import max_drawdown


from src.quantedge.config import load_config


from src.quantedge.var import build_var_suite


def test_initial_drawdown_includes_starting_wealth():
    assert max_drawdown(pd.Series([-.1, 0.])) == pytest.approx(-.1)


def test_empirical_es_fractional_tail_and_correct_dollar_transform():
    returns = np.log1p(np.array([-.4, -.2, .1, .2]))
    result = empirical_risk(returns, .625)  # Tail mass = 1.5 observations.
    assert result["es_simple_pct"] == pytest.approx((.4+.5*.2)/1.5)
    assert result["es_log"] == pytest.approx((-returns[0]-.5*returns[1])/1.5)
    assert result["es_simple_pct"] != pytest.approx(1-np.exp(-result["es_log"]))
    assert result["es_log"] >= result["var_log"]
    assert tail_weights(-returns, .625).sum() == pytest.approx(1)


def test_normal_es_against_conditional_integral():
    sigma, c = .2, .95
    risk = normal_risk(sigma, c)
    z = norm.ppf(c)
    expected = norm.expect(lambda r: -np.expm1(r), loc=0, scale=sigma,
                           lb=-10, ub=-z*sigma)/(1-c)
    assert risk["es_log"] == pytest.approx(sigma*norm.pdf(z)/(1-c))
    assert risk["es_simple_pct"] == pytest.approx(expected)
    assert risk["es_simple_pct"] > risk["var_simple_pct"]
    assert normal_risk(0, c)["es_simple_pct"] == pytest.approx(0)
