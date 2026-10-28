import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, chi2, genpareto
from scipy.integrate import quad
from src.quantedge.tail import empirical_risk, normal_risk, tail_weights
from src.quantedge.backtesting import kupiec_test, christoffersen_test, basel_traffic_light, backtest_forecasts
from src.quantedge.evt import fit_thresholds, select_threshold, gpd_risk
from src.quantedge.risk import build_risk_forecasts, common_evaluation_sample
from src.quantedge.contributions import risk_contributions
from src.quantedge.stress import historical_stress, hypothetical_stress
from src.quantedge.kpis import max_drawdown
from src.quantedge.analytics import fit_training_portfolio, run_analytical_layers, export_analytical_layers
from src.quantedge.config import load_config
from src.quantedge.var import build_var_suite


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


@pytest.mark.parametrize("shape", [-.6, -.01, 0., 1e-9, .2, 1.2])
def test_scalar_gpd_survival_matches_scipy_quadrature(shape):
    threshold, scale, confidence = .005, .003, .99
    risk = gpd_risk(threshold, shape, scale, .15, confidence)
    v = risk["var_log"]
    conditional_scale = scale+shape*(v-threshold)
    upper = min(-conditional_scale/shape, 40.) if shape < 0 else 40.
    integral = quad(lambda y: np.exp(-y)*genpareto.sf(y, shape, scale=conditional_scale),
                    0, upper, epsabs=1e-9)[0]
    expected = -np.expm1(-v)+np.exp(-v)*integral
    assert risk["es_simple_pct"] == pytest.approx(expected, abs=1e-12)


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


def test_risk_components_add_up_and_match_finite_differences():
    rng = np.random.default_rng(9)
    data = pd.DataFrame(rng.normal(.0001, .01, (300, 3)), columns=list("ABC"),
                        index=pd.bdate_range("2020-01-01", periods=300))
    weights = pd.Series([.4, .35, .25], index=data.columns)
    result = risk_contributions(data, weights, [.95, .99], 1e7)
    for (method, c), group in result.groupby(["method", "confidence"]):
        assert group.component_var.sum() == pytest.approx(group.portfolio_var.iloc[0])
        assert group.component_es.sum() == pytest.approx(group.portfolio_es.iloc[0])
        assert group.component_es_usd.sum() == pytest.approx(group.portfolio_es.iloc[0]*1e7)
        if method == "historical_simple":
            losses = -data.to_numpy() @ weights.to_numpy()
            epsilon = 1e-7
            for j, (_, row) in enumerate(group.iterrows()):
                bumped = losses-epsilon*data.iloc[:, j].to_numpy()
                derivative = (np.quantile(bumped, c)-np.quantile(losses, c))/epsilon
                assert row.marginal_var == pytest.approx(derivative, abs=1e-7)


def test_stress_paths_rebalance_policy_shocks_and_initial_drawdown():
    dates = pd.bdate_range("2020-01-01", periods=2)
    data = pd.DataFrame({"A": [-.1, .1], "B": [0, 0]}, index=dates)
    weights = pd.Series({"A": .5, "B": .5})
    scenarios = [{"name": "test", "start": str(dates[0].date()), "end": str(dates[-1].date())}]
    daily, paths = historical_stress(data, weights, scenarios, 100)
    held, _ = historical_stress(data, weights, scenarios, 100, "buy_and_hold")
    assert daily.cumulative_return.iloc[0] == pytest.approx(.95*1.05-1)
    assert held.cumulative_return.iloc[0] == pytest.approx(.5*.9*1.1+.5-1)
    assert paths.drawdown.iloc[0] == pytest.approx(-.05)
    assert max_drawdown(pd.Series([-.1, 0])) == pytest.approx(-.1)
    shocks, components = hypothetical_stress(weights, [{"name": "shock", "shocks": {"A": -.2, "B": .1}}], 100)
    assert shocks.pnl_usd.iloc[0] == pytest.approx(-5)
    assert components.pnl_contribution_usd.sum() == pytest.approx(-5)
    with pytest.raises(ValueError, match="Every asset"):
        hypothetical_stress(weights, [{"name": "bad", "shocks": {"A": -.2}}], 100)


def test_training_weights_are_invariant_to_future_returns_and_rates():
    cfg, data, rates, weights = _integration_fixture()
    first, metadata = fit_training_portfolio(data, rates, cfg)
    modified = data.copy()
    modified.loc[cfg["validation"]["test_start"]:] *= -10
    future_rates = rates.copy()
    future_rates.loc[cfg["validation"]["test_start"]:] = 100
    second, metadata2 = fit_training_portfolio(modified, future_rates, cfg)
    pd.testing.assert_series_equal(first, second)
    assert metadata == metadata2
    assert pd.Timestamp(metadata["training_end"]) < pd.Timestamp(cfg["validation"]["test_start"])
    cfg["validation"]["train_end"] = cfg["validation"]["test_start"]
    with pytest.raises(ValueError, match="before"):
        fit_training_portfolio(data, rates, cfg)


def test_training_can_include_2008_and_responds_to_crisis_period_returns():
    cfg, data, rates, _ = _integration_fixture()
    dates = pd.bdate_range("2007-01-01", periods=len(data))
    data.index = dates
    rates.index = dates
    cfg["validation"].update(train_start="2007-01-01", train_end=str(dates[499].date()),
                              test_start=str(dates[500].date()))
    before, metadata = fit_training_portfolio(data, rates, cfg)
    changed = data.copy()
    crisis_dates = dates[dates.year == 2008][:5]
    changed.loc[crisis_dates, "AGG"] = -.10
    after, _ = fit_training_portfolio(changed, rates, cfg)
    assert metadata["training_start"] == "2007-01-01"
    assert metadata["training_observations"] == 500
    assert len(crisis_dates) == 5
    assert not np.allclose(before, after)


def _integration_fixture():
    cfg = _small_config()
    dates = pd.bdate_range("2010-01-01", periods=650)
    rng = np.random.default_rng(22)
    data = pd.DataFrame(rng.standard_t(8, (650, 7))*.008+.0004,
                        index=dates, columns=cfg["data"]["tickers"])
    cfg["portfolio"]["target_return_annual"] = .01
    cfg["validation"].update(train_start=str(dates[0].date()), train_end=str(dates[499].date()),
                              test_start=str(dates[500].date()), min_train_observations=300)
    cfg["stress"]["historical"] = [{"name": "fixture", "start": str(dates[0].date()), "end": str(dates[50].date())}]
    rates = pd.Series(2., index=dates)
    weights = pd.Series([.2, .15, .05, .4, .1, .05, .05], index=data.columns)
    return cfg, data, rates, weights


def test_end_to_end_exports_and_evaluation_alignment(tmp_path):
    cfg, data, rates, weights = _integration_fixture()
    tables, metadata = run_analytical_layers(data, rates, weights, cfg)
    export_analytical_layers(tables, metadata, cfg, tmp_path)
    forecasts = tables["risk_forecasts_oos"]
    assert len(tables["backtest_results"]) == 12
    assert forecasts.date.min() == data.index[500]
    assert forecasts.groupby(["method", "confidence"]).size().eq(150).all()
    assert (forecasts.allocation_training_end < forecasts.date).all()
    assert tables["oos_kpi_summary"].observations.iloc[0] == 150
    assert pd.read_csv(tmp_path/"outputs/tables/oos_portfolio_weights.csv").asset.tolist() == list(weights.index)
    assert (tmp_path/"outputs/models/analytical_run.json").exists()
    with pytest.raises(ValueError, match="gaps"):
        # Remove an interior point instead of the first evaluation date.
        backtest_forecasts(forecasts.drop(forecasts.index[12]))


def test_full_pipeline_exports_all_layers_from_raw_prices(tmp_path, monkeypatch):
    from src.quantedge import main as pipeline
    cfg, returns, rates, _ = _integration_fixture()
    prices = 100*(1+returns).cumprod()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "load_config", lambda _: cfg)
    prices.to_csv(tmp_path/"prices.csv")
    rates.rename("DGS3MO").to_csv(tmp_path/"rates.csv")
    pipeline.run(prices_path=str(tmp_path/"prices.csv"), risk_free_path=str(tmp_path/"rates.csv"))
    tables = tmp_path/"outputs/tables"
    assert pd.read_csv(tables/"portfolio_weights.csv").asset.tolist() == list(returns.columns)
    assert pd.read_csv(tables/"expected_returns.csv").Ticker.tolist() == list(returns.columns)
    assert pd.read_csv(tables/"covariance_matrix.csv").asset.tolist() == list(returns.columns)
    assert {"es_usd", "var_usd", "evaluation_basis"}.issubset(pd.read_csv(tables/"var_forecasts.csv"))
    assert len(pd.read_csv(tables/"backtest_results.csv")) == 12
    assert (tmp_path/"outputs/models/analytical_run.json").exists()


def test_unavailable_evt_and_internal_gaps_are_not_silently_validated():
    config = _small_config()
    dates = pd.bdate_range("2020-01-01", periods=240)
    flat = pd.Series(0., index=dates)
    forecasts, diagnostics = build_risk_forecasts(flat, config, dates[200])
    assert diagnostics.status.eq("insufficient_exceedances").all()
    with pytest.raises(ValueError, match="no forecasts"):
        common_evaluation_sample(forecasts, ["historical", "parametric_ewma", "monte_carlo_bootstrap", "evt_gpd"])


def test_stress_marks_truncated_crisis_data():
    dates = pd.bdate_range("2020-03-01", periods=10)
    data = pd.DataFrame({"A": -.01}, index=dates)
    weights = pd.Series({"A": 1.})
    report, _ = historical_stress(data, weights,
        [{"name": "partial", "start": "2020-01-01", "end": "2020-03-31"}], 100)
    assert report.status.iloc[0] == "partial_data"
