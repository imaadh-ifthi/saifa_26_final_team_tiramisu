import numpy as np


import pandas as pd


import pytest


from scipy.stats import norm, chi2


from src.quantedge.tail import empirical_risk, normal_risk, tail_weights


from src.quantedge.backtesting import kupiec_test, christoffersen_test, basel_traffic_light, backtest_forecasts


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


def test_kupiec_boundaries_and_reference_values():
    zero = kupiec_test(np.zeros(250), .99)
    assert zero["kupiec_lr"] == pytest.approx(-2*250*np.log(.99))
    assert zero["kupiec_p_value"] == pytest.approx(chi2.sf(zero["kupiec_lr"], 1))
    assert np.isfinite(kupiec_test(np.ones(250), .99)["kupiec_lr"])
    # MATLAB published example: 1043 observations, 57 failures at 95%.
    hits = np.r_[np.ones(57), np.zeros(986)]
    assert kupiec_test(hits, .95)["kupiec_lr"] == pytest.approx(.46147, abs=1e-5)


def test_christoffersen_reference_counts_and_cluster_detection():
    # Published n00=932,n01=53,n10=53,n11=4, generated as 53 closed failure runs.
    hits = [0]*933
    for j in range(53):
        hits.extend([1]*(2 if j < 4 else 1))
        hits.append(0)
    result = christoffersen_test(hits)
    assert [result[k] for k in ("n00", "n01", "n10", "n11")] == [932, 53, 53, 4]
    assert result["independence_lr"] == pytest.approx(.25866, abs=1e-5)
    assert christoffersen_test(np.r_[np.zeros(125), np.ones(125)])["independence_reject"]
    assert christoffersen_test(np.zeros(250))["independence_status"] == "insufficient_transition_states"


def test_basel_uses_last_250_observations_and_requires_full_window():
    assert basel_traffic_light(np.r_[np.ones(100), np.zeros(250)], .99)["traffic_light"] == "green"
    assert basel_traffic_light(np.r_[np.ones(5), np.zeros(245)], .99)["traffic_light"] == "yellow"
    assert basel_traffic_light(np.r_[np.ones(10), np.zeros(240)], .99)["traffic_light"] == "red"
    assert basel_traffic_light(np.zeros(249), .99)["traffic_light"] == "insufficient_data"
    assert basel_traffic_light(np.zeros(250), .95)["traffic_light"] == "not_applicable"
