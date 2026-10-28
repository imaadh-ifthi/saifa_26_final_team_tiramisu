"""Checks for the Layer 3-5 upgrade modules (GJR-GARCH, ES backtests, scoring, EVT extras)."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from src.quantedge.tail_grid import probability_grid, risk_from_quantiles, pit_from_quantiles, gpd_tail_quantiles
from src.quantedge.es_backtests import du_escanciano, kratz_multinomial, exceedance_residual_test
from src.quantedge.backtests_extra import dynamic_quantile, duration_test, monte_carlo_pvalues
from src.quantedge.scoring import fz0_loss, diebold_mariano, model_confidence_set
from src.quantedge.evt_extra import extremal_index, eqd_threshold, gpd_quantile, bootstrap_evt_interval


def simulate_gjr(n=3000, seed=1, omega=2e-6, alpha=0.02, gamma=0.12, beta=0.9):
    rng = np.random.default_rng(seed)
    r, var = np.zeros(n), omega / (1 - alpha - gamma / 2 - beta)
    for i in range(n):
        r[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + (alpha + gamma * (r[i] < 0)) * r[i] ** 2 + beta * var
    return r


def test_gjr_fit_recovers_leverage_and_persistence():
    fit = fit_gjr_garch(simulate_gjr())
    assert fit["gamma"] > fit["alpha"]
    assert 0.9 < fit["persistence"] < 0.999
    assert abs(np.std(fit["residuals"]) - 1) < 0.1


def test_filter_uses_only_past_returns():
    r = simulate_gjr(1200)
    params = fit_gjr_garch(r[:1000])
    a = filter_volatility(r[:1100], params)
    changed = r[:1100].copy()
    changed[1050] = -0.2                               # a crash on day 1050
    b = filter_volatility(changed, params)
    assert np.allclose(a[:1051], b[:1051])             # sigma up to day 1050 cannot know about it
    assert b[1051] > a[1051]                           # the day after reacts


def test_grid_matches_normal_closed_form():
    grid = probability_grid()
    q = 0.01 * norm.ppf(grid)
    risk = risk_from_quantiles(q, grid, 0.99)
    assert np.isclose(risk["var_log"], 0.01 * norm.ppf(0.99), rtol=1e-6)
    assert np.isclose(risk["es_log"], 0.01 * norm.pdf(norm.ppf(0.99)) / 0.01, rtol=2e-3)
    assert np.isclose(pit_from_quantiles(-0.01 * norm.ppf(0.99), q, grid), 0.01, rtol=1e-4)


def test_gpd_tail_quantiles_reproduce_gpd_formula():
    grid = probability_grid()
    sample = np.random.default_rng(0).standard_normal(1000) * 0.01
    q = gpd_tail_quantiles(sample, grid, 0.012, 0.2, 0.005, 0.1)
    p = 0.01
    expected = -(0.012 + 0.005 / 0.2 * ((0.1 / p) ** 0.2 - 1))
    assert np.isclose(np.interp(p, grid, q), expected, rtol=1e-6)
    assert np.all(np.diff(q) >= 0)


def test_du_escanciano_accepts_correct_and_rejects_thin_tails():
    rng = np.random.default_rng(3)
    good = rng.random(5000)
    assert du_escanciano(good, 0.025)["de_ues_p"] > 0.01
    # a model with too-thin tails maps bad days to very small PITs too often
    bad = np.where(rng.random(5000) < 0.03, rng.random(5000) * 0.003, good)
    result = du_escanciano(bad, 0.025)
    assert result["de_ues_p"] < 0.01 and result["de_ues_z"] > 0


def test_multinomial_counts_add_up_and_detect_heavy_tail():
    rng = np.random.default_rng(4)
    good = kratz_multinomial(rng.random(4000))
    assert sum(map(int, good["mn_observed"].split(";"))) == 4000
    bad = kratz_multinomial(np.concatenate([rng.random(3900), rng.random(100) * 0.005]))
    assert bad["mn_pearson_p"] < good["mn_pearson_p"] and bad["mn_pearson_p"] < 0.01


def test_exceedance_residual_flags_understated_es():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(5000) * 0.01
    var = np.full(5000, 0.01 * norm.ppf(0.99))
    true_es = np.full(5000, 0.01 * norm.pdf(norm.ppf(0.99)) / 0.01)
    sigma = np.full(5000, 0.01)
    assert exceedance_residual_test(r, var, true_es, sigma)["mf_p"] > 0.05
    assert exceedance_residual_test(r, var, true_es * 0.8, sigma)["mf_p"] < 0.05


def test_dq_and_duration_detect_clustered_breaches():
    rng = np.random.default_rng(6)
    iid = rng.random(3000) < 0.01
    clustered = np.zeros(3000, dtype=bool)
    for start in rng.choice(2990, 8, replace=False):
        clustered[start:start + 4] = True
    var = np.full(3000, 0.02)
    assert dynamic_quantile(clustered, var, 0.01)["dq_p"] < 0.01
    assert duration_test(clustered)["dur_weibull_b"] < 1
    assert monte_carlo_pvalues(iid, 0.01, 999)["mc_kupiec_p"] > 0.01


def test_fz0_prefers_true_risk_and_mcs_keeps_it():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(4000) * 0.01
    v, e = 0.01 * norm.ppf(0.975), 0.01 * norm.pdf(norm.ppf(0.975)) / 0.025
    losses = pd.DataFrame({"true": fz0_loss(y, np.full(4000, v), np.full(4000, e), 0.975),
                           "too_low": fz0_loss(y, np.full(4000, v * 0.6), np.full(4000, e * 0.6), 0.975)})
    assert losses.true.mean() < losses.too_low.mean()
    assert diebold_mariano(losses.true, losses.too_low)["dm_stat"] < 0
    mcs = model_confidence_set(losses, n_boot=500)
    assert bool(mcs.set_index("model").loc["true", "in_mcs"])


def test_extremal_index_sees_clusters():
    rng = np.random.default_rng(9)
    iid = rng.standard_normal(5000)
    clustered = np.repeat(rng.standard_normal(2500), 2)   # every value appears twice in a row
    assert extremal_index(iid, np.quantile(iid, 0.95)) > 0.8
    assert extremal_index(clustered, np.quantile(clustered, 0.95)) < 0.7


def test_eqd_picks_a_candidate_and_interval_brackets_point():
    rng = np.random.default_rng(10)
    losses = np.abs(rng.standard_t(4, 1000)) * 0.01
    result = eqd_threshold(losses, [0.8, 0.85, 0.9, 0.95], n_boot=20)
    assert result["best"]["threshold_quantile"] in (0.8, 0.85, 0.9, 0.95)
    ci = bootstrap_evt_interval(losses, 0.9, 0.99, n_boot=100)
    assert ci["var_log_5"] <= ci["var_log"] <= ci["var_log_95"]
    assert np.isclose(gpd_quantile(0.5, 0.0, 1.0), np.log(2))
