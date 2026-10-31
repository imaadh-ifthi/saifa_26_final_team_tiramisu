import numpy as np
import pandas as pd
import pytest
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from experiments.improved_risk.candidate_models.gjr_filtered_historical import GJRGARCHFilteredHistorical
from experiments.improved_risk.candidate_models.gjr_filtered_bootstrap import GJRGARCHFilteredBootstrap


def sample_returns(n=1200, seed=123):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0.0003, 0.008, n), index=dates)


def test_gjr_model_one_step_ahead_property():
    """Verify one-step-ahead property (no lookahead bias)."""
    returns = sample_returns(1200)
    model1 = GJRGARCHFilteredHistorical(window=1000, refit_every=21)
    model2 = GJRGARCHFilteredHistorical(window=1000, refit_every=21)

    eval_idx = 1050
    est1, _ = model1.fit_and_forecast(returns, eval_idx, [0.95, 0.99])

    # Alter future return at eval_idx and after
    changed = returns.copy()
    changed.iloc[eval_idx:] = -0.5
    est2, _ = model2.fit_and_forecast(changed, eval_idx, [0.95, 0.99])

    # Estimates for eval_idx must be strictly identical
    for c in [0.95, 0.99]:
        assert est1[c]["var_log"] == pytest.approx(est2[c]["var_log"])
        assert est1[c]["es_log"] == pytest.approx(est2[c]["es_log"])
        assert est1[c]["forecast_volatility"] == pytest.approx(est2[c]["forecast_volatility"])

    bootstrap1 = GJRGARCHFilteredBootstrap(window=1000, refit_every=21, n_simulations=1000, seed=42)
    bootstrap2 = GJRGARCHFilteredBootstrap(window=1000, refit_every=21, n_simulations=1000, seed=42)
    boot1, _ = bootstrap1.fit_and_forecast(returns, eval_idx, [0.95, 0.99], seed_offset=eval_idx)
    boot2, _ = bootstrap2.fit_and_forecast(changed, eval_idx, [0.95, 0.99], seed_offset=eval_idx)
    for c in [0.95, 0.99]:
        assert boot1[c]["var_log"] == pytest.approx(boot2[c]["var_log"])
        assert boot1[c]["es_log"] == pytest.approx(boot2[c]["es_log"])


def test_gjr_volatility_positive_and_finite():
    """Verify forecast volatility is positive and finite."""
    returns = sample_returns(1100)
    model = GJRGARCHFilteredHistorical(window=1000, refit_every=21)
    estimates, diag = model.fit_and_forecast(returns, 1000, [0.99])

    vol = estimates[0.99]["forecast_volatility"]
    assert np.isfinite(vol)
    assert vol > 0


def test_gjr_var_is_positive():
    """Verify VaR is positive."""
    returns = sample_returns(1100)
    hist_model = GJRGARCHFilteredHistorical(window=1000, refit_every=21)
    boot_model = GJRGARCHFilteredBootstrap(window=1000, refit_every=21, n_simulations=500, seed=42)

    hist_est, _ = hist_model.fit_and_forecast(returns, 1000, [0.90, 0.95, 0.99])
    boot_est, _ = boot_model.fit_and_forecast(returns, 1000, [0.90, 0.95, 0.99])

    for c in [0.90, 0.95, 0.99]:
        assert hist_est[c]["var_log"] > 0
        assert hist_est[c]["var_simple_pct"] > 0
        assert hist_est[c]["es_log"] >= hist_est[c]["var_log"]

        assert boot_est[c]["var_log"] > 0
        assert boot_est[c]["var_simple_pct"] > 0
        assert boot_est[c]["es_log"] >= boot_est[c]["var_log"]


def test_gjr_exception_flags_are_boolean():
    """Verify exception flags are boolean."""
    returns = sample_returns(1100)
    model = GJRGARCHFilteredHistorical(window=1000, refit_every=21)
    estimates, _ = model.fit_and_forecast(returns, 1000, [0.99])
    actual_log_return = float(returns.iloc[1000])

    exception = actual_log_return < -estimates[0.99]["var_log"]
    assert isinstance(exception, (bool, np.bool_))


def test_gjr_garch_parameters_satisfy_stationarity():
    """Verify GJR-GARCH parameters satisfy stationarity: alpha + gamma / 2 + beta < 1."""
    returns = sample_returns(1100)
    model = GJRGARCHFilteredHistorical(window=1000, refit_every=21)
    _, diag = model.fit_and_forecast(returns, 1000, [0.99])

    assert diag is not None
    persistence = diag["persistence"]
    alpha = diag["alpha"]
    gamma = diag["gamma"]
    beta = diag["beta"]

    assert pytest.approx(alpha + gamma / 2 + beta) == persistence
    assert persistence < 1.0
    assert diag["converged"] is True
    assert diag["omega"] > 0
    assert diag["nu"] > 2.0


def test_bootstrap_is_seeded_and_diagnostics_are_at_refit_cutoff():
    returns = sample_returns(1100)
    forecasts = []
    for _ in range(2):
        model = GJRGARCHFilteredBootstrap(window=1000, refit_every=21,
                                          n_simulations=1000, seed=42)
        estimates, diagnostic = model.fit_and_forecast(returns, 1000, [.90, .95, .99], seed_offset=1000)
        forecasts.append(estimates)
        assert diagnostic["fit_date"] == returns.index[1000]
        assert diagnostic["estimation_end"] == returns.index[999]
        assert diagnostic["estimation_end"] < diagnostic["fit_date"]
        assert diagnostic["persistence"] < 1
        assert all(isinstance(estimates[c]["var_log"], float) for c in [.90, .95, .99])
        assert all(estimates[c]["forecast_volatility"] > 0 for c in [.90, .95, .99])
    for confidence in [.90, .95, .99]:
        assert forecasts[0][confidence]["var_log"] == pytest.approx(forecasts[1][confidence]["var_log"])
        assert forecasts[0][confidence]["es_log"] == pytest.approx(forecasts[1][confidence]["es_log"])
