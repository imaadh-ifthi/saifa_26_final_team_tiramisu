from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.integrate import quad
from scipy.stats import t
from experiments.improved_risk.models import (causal_volatility, fit_student_df, student_risk,
                                               filtered_gpd_risk, build_improved_forecasts, empirical_variant)
from experiments.improved_risk.run import compare_models, export_comparison
from experiments.improved_risk.volatility_probe import forecasts_for_lambda
from src.quantedge.config import load_config
from src.quantedge.evt import gpd_risk
from src.quantedge.tail import normal_risk


def settings():
    base = load_config()
    base["var"].update(historical_window=40, monte_carlo_window=40, monte_carlo_simulations=200,
                       ewma_min_periods=20)
    base["evt"].update(window=100, refit_every=30, min_exceedances=10,
                       threshold_quantiles=[.8, .85, .9], shape_tolerance=1, scale_tolerance=1)
    dates = pd.bdate_range("2010-01-01", periods=240)
    base["validation"].update(test_start=str(dates[170].date()))
    config_path = Path(__file__).resolve().parents[1]/"config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["filter"]["min_periods"] = 20
    config["historical_window"] = 80
    config["monte_carlo_simulations"] = 200
    config["student_t"].update(window=100, refit_every=30)
    config["evt"] = base["evt"].copy()
    config["sensitivity"] = {"raw_historical_windows": [60, 100],
                              "empirical_quantile_methods": ["inverted_cdf"]}
    returns = pd.Series(np.random.default_rng(31).standard_t(6, 240)*.008, index=dates)
    return returns, base, config


def test_causal_filter_and_standardization_use_prior_observations():
    dates = pd.bdate_range("2020-01-01", periods=100)
    returns = pd.Series(np.sin(np.arange(100))*.01, index=dates)
    before = causal_volatility(returns, min_periods=20)
    changed = returns.copy()
    changed.iloc[70:] = -.5
    after = causal_volatility(changed, min_periods=20)
    pd.testing.assert_series_equal(before.iloc[:71], after.iloc[:71])
    expected = returns.iloc[:70].ewm(alpha=.06, min_periods=20, adjust=False).std().iloc[-1]
    assert before.iloc[70] == pytest.approx(expected)
    assert after.iloc[71] > before.iloc[71]
    flat = causal_volatility(pd.Series(0., index=dates), min_periods=20)
    assert flat.dropna().gt(0).all()


@pytest.mark.parametrize("df,sigma", [(2.1, .01), (3., .01), (6., .2), (30., .02)])
def test_student_es_against_numerical_density_integral(df, sigma):
    c = .99
    risk = student_risk(sigma, df, c)
    scale = np.sqrt((df-2)/df)
    x = t.ppf(c, df)
    expected_log = quad(lambda z: sigma*scale*z*t.pdf(z, df), x, np.inf)[0]/(1-c)
    expected_simple = quad(lambda z: -np.expm1(-sigma*scale*z)*t.pdf(z, df), x, np.inf,
                            epsabs=1e-11)[0]/(1-c)
    assert risk["var_log"] == pytest.approx(sigma*scale*x)
    assert risk["es_log"] == pytest.approx(expected_log, rel=1e-6)
    assert risk["es_simple_pct"] == pytest.approx(expected_simple, abs=2e-6)
    assert risk["es_simple_pct"] >= risk["var_simple_pct"]


def test_student_variance_scaling_gaussian_limit_and_fit():
    assert student_risk(.01, 100000., .99)["var_log"] == pytest.approx(normal_risk(.01, .99)["var_log"], rel=1e-4)
    rng = np.random.default_rng(8)
    residuals = rng.standard_t(6, 20000)*np.sqrt(4/6)
    df, status = fit_student_df(residuals)
    assert 5 < df < 7
    assert status == "ok"
    with pytest.raises(ValueError, match="above 2"):
        fit_student_df(residuals, (1., 5.))
    assert student_risk(0, 6, .99)["es_simple_pct"] == 0


def test_filtered_gpd_scales_tail_and_updates_without_refitting():
    fit = pd.Series({"threshold_log": 1.5, "shape": .1, "scale": .5, "exceedance_probability": .1})
    first = filtered_gpd_risk(fit, .01, .99)
    second = filtered_gpd_risk(fit, .02, .99)
    expected = gpd_risk(.015, .1, .005, .1, .99)
    assert first == expected
    assert second["var_log"] == pytest.approx(2*first["var_log"])
    assert second["es_log"] == pytest.approx(2*first["es_log"])
    assert second["es_simple_pct"] > first["es_simple_pct"]


def test_all_candidates_are_invariant_to_today_and_future_returns():
    returns, base, config = settings()
    start = base["validation"]["test_start"]
    before, diag = build_improved_forecasts(returns, base, config, start)
    changed = returns.copy()
    changed.iloc[200:] = -.3
    after, _ = build_improved_forecasts(changed, base, config, start)
    cols = ["date", "method", "confidence", "var_log", "es_log", "es_simple_pct", "forecast_volatility"]
    pd.testing.assert_frame_equal(before.loc[before.date <= returns.index[200], cols].reset_index(drop=True),
                                  after.loc[after.date <= returns.index[200], cols].reset_index(drop=True))
    assert (before.estimation_end < before.date).all()
    assert (before.tail_fit_end < before.date).all()
    assert (diag.estimation_end < diag.fit_date).all()
    assert (before.es_simple_pct+1e-12 >= before.var_simple_pct).all()
    assert before.groupby(["method", "confidence"]).size().nunique() == 1
    repeat, _ = build_improved_forecasts(returns, base, config, start)
    pd.testing.assert_frame_equal(before, repeat)


def test_comparison_exports_all_baselines_candidates_and_sensitivities(tmp_path):
    returns, base, config = settings()
    tables = compare_models(returns, base, config)
    forecasts = tables["forecasts"]
    assert forecasts.method.nunique() == 11
    assert len(tables["backtests"]) == 33
    assert forecasts.groupby(["method", "confidence"]).size().eq(70).all()
    assert set(forecasts.category) == {"baseline", "candidate", "sensitivity"}
    assert forecasts.quantile_loss.ge(0).all()
    assert tables["scores"].mean_quantile_loss.ge(0).all()
    export_comparison(tables, tmp_path, {"evaluation_label": "development"})
    assert pd.read_csv(tmp_path/"forecasts.csv.gz").shape == forecasts.shape
    assert "not an untouched holdout" in (tmp_path/"comparison.md").read_text()
    assert len(pd.read_csv(tmp_path/"backtests.csv")) == 33


def test_quantile_sensitivity_is_explicit_and_not_a_confidence_change():
    returns = np.array([-.04, -.03, -.02, .01, .02])
    linear = empirical_variant(returns, .9)
    discrete = empirical_variant(returns, .9, "inverted_cdf")
    assert linear["var_log"] == pytest.approx(.036)
    assert discrete["var_log"] == pytest.approx(.04)
    assert linear["es_log"] == discrete["es_log"]


def test_inverse_cdf_quantile_uses_loss_order_statistic_at_exact_boundary():
    returns = -np.arange(1, 11)/100
    risk = empirical_variant(returns, .9, "inverted_cdf")
    assert risk["var_log"] == pytest.approx(.09)


def test_comparison_rejects_missing_warmup_instead_of_shortening_test():
    returns, base, config = settings()
    base["validation"]["test_start"] = str(returns.index[110].date())
    with pytest.raises(ValueError, match="every declared evaluation date"):
        compare_models(returns, base, config)


def test_degenerate_tail_sample_fails_explicitly():
    returns, base, config = settings()
    returns[:] = 0
    with pytest.raises(ValueError, match="varying finite"):
        build_improved_forecasts(returns, base, config, base["validation"]["test_start"])


def test_fixed_volatility_probe_uses_prior_returns_and_same_dates():
    returns, base, experiment = settings()
    start = pd.Timestamp(base["validation"]["test_start"])
    reference = forecasts_for_lambda(returns, start, base, experiment, .94)
    faster = forecasts_for_lambda(returns, start, base, experiment, .90)
    assert len(reference) == len(faster) == 140
    assert reference[["date", "method"]].equals(faster[["date", "method"]])
    assert (reference.estimation_end < reference.date).all()
    changed = returns.copy()
    changed.iloc[200:] = -.3
    revised = forecasts_for_lambda(changed, start, base, experiment, .90)
    cols = ["date", "method", "var_log", "var_usd", "forecast_volatility"]
    pd.testing.assert_frame_equal(
        faster[faster.date.le(returns.index[200])][cols].reset_index(drop=True),
        revised[revised.date.le(returns.index[200])][cols].reset_index(drop=True))
