"""GJR-GARCH-X(1,1): the usual GJR filter, plus one extra input in the variance equation.

    sigma_t^2 = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e_{t-1}^2 + beta * sigma_{t-1}^2 + delta * x_t

x_t has to be something we already knew before day t. We try two:
  * the VIX at yesterday's close, turned into a daily variance: (VIX_{t-1} / 100)^2 / 252
  * yesterday's high-low range variance of the portfolio (Garman-Klass, see range_vol.py)

The idea in plain words: normal GARCH only finds out a crisis has started from yesterday's
close-to-close return. That is one number per day, and it is noisy. The VIX (what options
traders expect) and the intraday range (how far prices actually swung) give it a second,
faster signal. Hansen, Huang and Shek (2012) and Molnar (2016) found that adding a better
volatility measure like this improves forecasts. Lynch, Hasan and Siddique (2023) found the
VIX level helped predict bank VaR breaches, including in COVID.

If delta = 0 (or x = None) this is exactly the GJR model in garch.py.
The likelihood uses Student-t shocks, but that only helps estimate sigma. The tail itself
is handled later by EVT on the standardised shocks.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

SCALE = 100.0  # fit in percent returns, same as garch.py


def _variance_path(e, x, omega, alpha, gamma, beta, delta, start_var):
    """sigma_t^2 for every t; sigma_t^2 only uses e up to t-1 and x up to t (x_t is lagged data)."""
    shock = (alpha + gamma * (e < 0)) * e**2
    drive = np.empty_like(e)
    drive[0] = start_var
    drive[1:] = omega + shock[:-1] + delta * x[1:]
    out = lfilter([1.0], [1.0, -beta], drive)
    out[0] = start_var
    return out


def _neg_loglik(params, r, x, start_var):
    mu, omega, alpha, gamma, beta, delta, nu = params
    if (omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or delta < 0 or nu <= 2.05
            or alpha + gamma / 2 + beta >= 0.999):
        return 1e10
    e = r - mu
    var = _variance_path(e, x, omega, alpha, gamma, beta, delta, start_var)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e10
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = const - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(e**2 / (var * (nu - 2)))
    value = -float(ll.sum())
    return value if np.isfinite(value) else 1e10


def fit_gjr_garch_x(returns, x=None, previous: dict | None = None) -> dict:
    """Fit on one window. returns and x in decimal units, same length; x[t] known before day t."""
    r = np.asarray(returns, dtype=float) * SCALE
    if r.ndim != 1 or len(r) < 250 or not np.isfinite(r).all():
        raise ValueError("GJR-GARCH-X needs at least 250 finite returns")
    has_x = x is not None
    xs = np.zeros_like(r) if not has_x else np.asarray(x, dtype=float) * SCALE**2
    if xs.shape != r.shape or not np.isfinite(xs).all() or (xs < 0).any():
        raise ValueError("x must be finite, non-negative and the same length as returns")
    start_var = float(np.var(r))
    # with x, start with part of the variance explained by x and a lower beta
    x0 = np.array([r.mean(), 0.02 * start_var, 0.02, 0.10, 0.88, 0.0, 8.0])
    starts = [x0]
    if has_x:
        ratio = start_var / max(float(xs.mean()), 1e-12)
        starts.append(np.array([r.mean(), 0.02 * start_var, 0.01, 0.06, 0.70, 0.2 * ratio, 8.0]))
    if previous is not None:
        starts.insert(0, np.array([previous["mu"] * SCALE, previous["omega"] * SCALE**2, previous["alpha"],
                                   previous["gamma"], previous["beta"], previous.get("delta", 0.0), previous["nu"]]))
    delta_bound = (0.0, 0.0) if not has_x else (0.0, 10.0 * start_var / max(float(xs.mean()), 1e-12))
    bounds = [(-1.0, 1.0), (1e-6, 10 * start_var), (0.0, 0.5), (0.0, 0.8), (0.0, 0.999), delta_bound, (2.1, 100.0)]
    best = None
    for s in starts:
        s = np.clip(s, [b[0] for b in bounds], [b[1] for b in bounds])
        fit = minimize(_neg_loglik, s, args=(r, xs, start_var), method="L-BFGS-B", bounds=bounds)
        if best is None or fit.fun < best.fun:
            best = fit
    if best is None or best.fun >= 1e9:
        raise ValueError("GJR-GARCH-X fit failed")
    mu, omega, alpha, gamma, beta, delta, nu = best.x
    var = _variance_path(r - mu, xs, omega, alpha, gamma, beta, delta, start_var)
    z = (r - mu) / np.sqrt(var)
    return {"mu": mu / SCALE, "omega": omega / SCALE**2, "alpha": alpha, "gamma": gamma, "beta": beta,
            "delta": delta, "nu": nu, "persistence": alpha + gamma / 2 + beta, "start_var": start_var / SCALE**2,
            "loglik": -best.fun, "converged": bool(best.success), "residuals": z}


def filter_volatility_x(returns, x, params: dict, x_next: float = 0.0) -> np.ndarray:
    """One-step-ahead sigma for every position plus the next unseen day (len(returns) + 1 values).

    x[t] is the regressor known before day t; x_next is the one known before the next day.
    """
    r = np.asarray(returns, dtype=float)
    xs = np.zeros(len(r) + 1) if x is None else np.append(np.asarray(x, dtype=float), x_next)
    e = np.append(r - params["mu"], 0.0)
    var = _variance_path(e, xs, params["omega"], params["alpha"], params["gamma"], params["beta"],
                         params.get("delta", 0.0), params["start_var"])
    return np.sqrt(var)
