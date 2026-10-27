import numpy as np


import pandas as pd


import pytest


from scipy.stats import norm, chi2


from src.quantedge.tail import empirical_risk, normal_risk, tail_weights


from src.quantedge.backtesting import kupiec_test, christoffersen_test, basel_traffic_light, backtest_forecasts


from src.quantedge.evt import fit_thresholds, select_threshold, gpd_risk


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


def _small_config():
    cfg = load_config()
    cfg["var"].update(historical_window=40, monte_carlo_window=40,
                      monte_carlo_simulations=500, ewma_min_periods=20)
    cfg["evt"].update(window=200, refit_every=30, min_exceedances=15,
                      threshold_quantiles=[.8, .85, .9], shape_tolerance=1, scale_tolerance=1)
    return cfg


def test_forecasts_do_not_change_when_today_and_future_change():
    cfg = _small_config()
    dates = pd.bdate_range("2010-01-01", periods=250)
    returns = pd.Series(np.random.default_rng(12).standard_t(6, 250)*.01, index=dates)
    before, diag = build_risk_forecasts(returns, cfg, start_date=dates[200])
    changed = returns.copy()
    changed.iloc[220:] = -.4
    after, _ = build_risk_forecasts(changed, cfg, start_date=dates[200])
    cols = ["method", "confidence", "date", "var_log", "es_log", "es_simple_pct"]
    pd.testing.assert_frame_equal(before.loc[before.date <= dates[220], cols].reset_index(drop=True),
                                  after.loc[after.date <= dates[220], cols].reset_index(drop=True))
    assert (before.estimation_end < before.date).all()
    assert (diag.estimation_end < diag.fit_date).all()
    assert (before.es_simple_pct+1e-12 >= before.var_simple_pct).all()
    assert set(before.method) == {"historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd"}
    common = common_evaluation_sample(before, list(before.method.unique()))
    assert common.groupby(["method", "confidence"]).size().nunique() == 1
    legacy, summary = build_var_suite(returns, cfg)
    assert {"es_log", "es_usd", "evaluation_basis"}.issubset(legacy)
    assert "latest_es_usd" in summary


def test_gpd_exponential_limit_and_infinite_log_es():
    result = gpd_risk(.02, 0, .01, .1, .99)
    v = .02+.01*np.log(10)
    assert result["var_log"] == pytest.approx(v)
    assert result["es_log"] == pytest.approx(v+.01)
    assert result["es_simple_pct"] == pytest.approx(1-np.exp(-v)/1.01)
    assert gpd_risk(.02, 1.1, .01, .1, .99)["es_log"] == np.inf
    assert gpd_risk(.02, 1.1, .01, .1, .99)["es_simple_pct"] < 1
    with pytest.raises(ValueError, match="below"):
        gpd_risk(.02, .1, .01, .05, .90)


def test_evt_diagnostics_recover_exponential_shape_and_fail_insufficient_tail():
    rng = np.random.default_rng(42)
    losses = rng.exponential(.01, 5000)
    report = fit_thresholds(pd.Series(-losses), [.85, .9, .95], 100)
    chosen, status = select_threshold(report, .9)
    assert abs(chosen["shape"]) < .15
    assert status == "stable_adjacent_thresholds"
    assert {"mean_excess", "modified_scale", "ks_statistic", "stable_with_next"}.issubset(report)
    insufficient = fit_thresholds(pd.Series(-losses[:20]), [.9], 50)
    assert select_threshold(insufficient, .99)[0] is None


def test_unavailable_evt_and_internal_gaps_are_not_silently_validated():
    config = _small_config()
    dates = pd.bdate_range("2020-01-01", periods=240)
    flat = pd.Series(0., index=dates)
    forecasts, diagnostics = build_risk_forecasts(flat, config, dates[200])
    assert diagnostics.status.eq("insufficient_exceedances").all()
    with pytest.raises(ValueError, match="no forecasts"):
        common_evaluation_sample(forecasts, ["historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd"])
