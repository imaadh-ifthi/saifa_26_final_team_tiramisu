"""Checks for the crisis-response models and the decision layer.

Most tests use small simulated data so they run in seconds. The key ones check timing:
a forecast for day t must not change when we change anything from day t onwards.
"""
import numpy as np
import pandas as pd
import pytest
from src.quantedge.dcc import correlation_path, fit_dcc
from src.quantedge.decision import euler_contributions, mandate_breaches, trade_menu, reverse_stress
from src.quantedge.garch import fit_gjr_garch, filter_volatility
from src.quantedge.garch_x import fit_gjr_garch_x, filter_volatility_x
from src.quantedge.model_selection import apply_selection_rule
from src.quantedge.ms_garch import mixture_quantiles
from src.quantedge.range_vol import garman_klass_gap, fit_har, har_variance_path, har_design
from src.quantedge.tail_grid import probability_grid

LIMITS = {"max_equity_exposure": 0.60, "min_defensive_exposure": 0.35, "max_eem": 0.10, "max_xlk": 0.15,
          "max_xle": 0.10, "gld_bounds": [0.05, 0.15], "max_single_position": 0.45}


def simulate_garch_x(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(1e-4, 5e-5, n))
    r, s2 = np.zeros(n), np.full(n, 1e-4)
    for t in range(1, n):
        s2[t] = 2e-6 + (0.02 + 0.10 * (r[t - 1] < 0)) * r[t - 1] ** 2 + 0.80 * s2[t - 1] + 0.10 * x[t]
        r[t] = np.sqrt(s2[t]) * rng.standard_t(8) / np.sqrt(8 / 6)
    return r, x


def test_garch_x_without_x_matches_plain_gjr():
    r, _ = simulate_garch_x()
    a, b = fit_gjr_garch(r), fit_gjr_garch_x(r)
    assert b["delta"] == 0.0
    assert abs(a["loglik"] - b["loglik"]) < 1e-3
    assert np.allclose(filter_volatility(r, a), filter_volatility_x(r, None, b), rtol=1e-3)


def test_garch_x_finds_the_regressor():
    r, x = simulate_garch_x()
    fit = fit_gjr_garch_x(r, x)
    assert fit["delta"] > 0.02
    assert fit["loglik"] > fit_gjr_garch_x(r)["loglik"]


def test_garch_x_sigma_does_not_see_the_future():
    r, x = simulate_garch_x()
    p = fit_gjr_garch_x(r[:1000], x[:1000])
    base = filter_volatility_x(r[:1100], x[:1100], p, x_next=x[1100])
    r2, x2 = r.copy(), x.copy()
    r2[1050:] *= 5          # change returns from day 1050 on
    x2[1051:] *= 5          # and x from day 1051 on (x[1050] is yesterday's data for day 1050)
    moved = filter_volatility_x(r2[:1100], x2[:1100], p, x_next=x2[1100])
    assert np.allclose(base[:1051], moved[:1051])
    assert not np.allclose(base[1051:], moved[1051:])


def test_garman_klass_with_gap_basic_cases():
    idx = pd.date_range("2020-01-01", periods=3)
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "adj_close": 100.0}, index=idx)
    assert np.allclose(garman_klass_gap(flat).iloc[1:], 0.0)
    wide = flat.copy()
    wide["high"], wide["low"] = 110.0, 90.0
    v = garman_klass_gap(wide).iloc[1:]
    assert np.allclose(v, 0.5 * np.log(110 / 90) ** 2)
    # a dividend (close above adj_close) must not create a fake overnight gap
    div = flat.copy()
    div.loc[idx[:2], "adj_close"] = 99.0
    div.loc[idx[0], "close"] = 100.0
    assert garman_klass_gap(div).iloc[2] > 0  # day after the adjustment step sees the 1% adjusted move
    assert abs(garman_klass_gap(div).iloc[1]) < 1e-12


def test_har_forecast_uses_only_past_values():
    rng = np.random.default_rng(1)
    v = np.exp(rng.normal(-9, 0.5, 400))
    fit = fit_har(v)
    full = har_variance_path(v, fit)
    v2 = v.copy()
    v2[300:] *= 10
    moved = har_variance_path(v2, fit)
    # row k of the path is the forecast for day 22 + k
    assert np.allclose(full[: 300 - 22 + 1], moved[: 300 - 22 + 1])
    assert har_design(np.log(v)).shape == (400 - 22 + 1, 4)


def test_dcc_recovers_rising_correlation_and_is_one_step_ahead():
    rng = np.random.default_rng(2)
    n = 1200
    rho = np.where(np.arange(n) < 600, -0.5, 0.5)
    e = rng.standard_normal((n, 2))
    z = np.column_stack([e[:, 0], rho * e[:, 0] + np.sqrt(1 - rho**2) * e[:, 1]])
    fit = fit_dcc(z)
    path = correlation_path(z, fit["a"], fit["b"], fit["qbar"])
    assert path.shape == (n + 1, 2, 2)
    assert path[550, 0, 1] < 0 < path[-1, 0, 1]
    z2 = z.copy()
    z2[800:] = 0
    path2 = correlation_path(z2, fit["a"], fit["b"], fit["qbar"])
    assert np.allclose(path[:801], path2[:801])


def test_mixture_quantiles_match_single_t_when_one_regime():
    from scipy.stats import t as student_t
    grid = probability_grid()
    q = mixture_quantiles(np.array([1.0, 0.0]), np.array([1e-4, 4e-4]), 0.0, 6.0, grid)
    exact = student_t.ppf(grid, 6.0) * np.sqrt(1e-4 * 4 / 6)
    assert np.allclose(q, exact, rtol=2e-3, atol=1e-6)


def test_selection_rule_prefers_gate_passers_then_score_then_simplicity():
    rows = []
    for m, p, fz in (("a", 0.5, -3.0), ("b", 0.01, -3.5), ("c", 0.5, -3.2), ("d", 0.5, -3.2)):
        for c in (0.95, 0.99):
            rows.append({"method": m, "confidence": c, "dq_p": p, "mean_fz0": fz})
    table = pd.DataFrame(rows)
    cands = [{"name": "a", "complexity": 1}, {"name": "b", "complexity": 1},
             {"name": "c", "complexity": 3}, {"name": "d", "complexity": 2}]
    sel = {"gate_level": 0.05, "gates": [{"confidence": 0.99, "test": "dq_p"}], "score_confidences": [0.95, 0.99]}
    ranking = apply_selection_rule(table, cands, sel)
    assert list(ranking.method) == ["d", "c", "a", "b"]


def test_euler_pieces_add_up_and_trade_menu_respects_mandate():
    rng = np.random.default_rng(3)
    names = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
    scen = rng.multivariate_normal(np.zeros(7), np.diag([1, 1.2, 1.5, 0.2, 0.6, 1.4, 1.3]) * 1e-4, 4000)
    w = pd.Series([0.23, 0.10, 0.0, 0.45, 0.15, 0.01, 0.06], index=names)
    w = w / w.sum()
    held = w[w > 0]
    d = euler_contributions(scen[:, [names.index(a) for a in held.index]], held, 0.975, 1e7)
    assert d.component_es_usd.sum() == pytest.approx(d.attrs["portfolio_es_usd"], rel=1e-9)
    assert d.component_var_usd.sum() == pytest.approx(d.attrs["portfolio_var_usd"], rel=1e-9)
    assert d.attrs["diversification_benefit_usd"] > 0
    assert mandate_breaches(w, LIMITS) == []
    menu = trade_menu(scen, w, LIMITS, pd.Series(0.05, index=names), lambda x: 0.1, 0.975, 1e7, 0.05)
    assert len(menu) > 0
    for _, t in menu.iterrows():
        new = w.copy()
        new[t.sell] -= 0.05
        new[t.buy] += 0.05
        assert mandate_breaches(new, LIMITS) == []
    assert not ((menu.buy == "AGG").any())          # AGG is already at the 45% single-position cap


def test_reverse_stress_scales_with_sensitivity():
    w = pd.Series({"SPY": 0.5, "AGG": 0.5})
    assert reverse_stress(w, pd.Series({"SPY": 1.0, "AGG": 0.0}), 0.10) == pytest.approx(0.20)
    assert reverse_stress(w, pd.Series({"SPY": 1.0, "AGG": 1.0}), 0.10) == pytest.approx(0.10)
