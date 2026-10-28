"""GJR-GARCH(1,1) volatility filter with Student-t shocks.

Model (Glosten, Jagannathan and Runkle, 1993):
    r_t = mu + e_t,   e_t = sigma_t * z_t
    sigma_t^2 = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e_{t-1}^2 + beta * sigma_{t-1}^2
z_t is Student-t scaled to unit variance, so sigma_t is the real daily volatility.

Why GJR and not plain EWMA: EWMA has one fixed decay and treats up and down days the same.
Equity-heavy portfolios get more volatile after falls (leverage effect), and gamma lets the
filter react more to losses. The parameters are estimated from past data only.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

SCALE = 100.0  # fit in percent returns, it keeps the optimizer well behaved


def _variance_path(e: np.ndarray, omega: float, alpha: float, gamma: float, beta: float,
                   start_var: float) -> np.ndarray:
    """sigma_t^2 for every t in e. sigma_0^2 = start_var; sigma_t^2 only uses e up to t-1."""
    shock = (alpha + gamma * (e < 0)) * e**2
    drive = np.empty_like(e)
    drive[0] = start_var
    drive[1:] = omega + shock[:-1]
    # sigma2_t = drive_t + beta * sigma2_{t-1}, with sigma2_0 = start_var
    out = lfilter([1.0], [1.0, -beta], drive)
    out[0] = start_var
    return out


def _neg_loglik(params: np.ndarray, r: np.ndarray, start_var: float) -> float:
    mu, omega, alpha, gamma, beta, nu = params
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or alpha + gamma / 2 + beta >= 0.999 or nu <= 2.05:
        return 1e10
    e = r - mu
    var = _variance_path(e, omega, alpha, gamma, beta, start_var)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return 1e10
    # standardised Student-t log density with unit variance
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = const - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(e**2 / (var * (nu - 2)))
    value = -float(ll.sum())
    return value if np.isfinite(value) else 1e10


def fit_gjr_garch(returns: np.ndarray, previous: dict | None = None) -> dict:
    """Maximum likelihood fit on one window of log returns (decimal units).

    previous: last fitted parameters, used as a warm start so monthly refits move smoothly.
    Returns parameters in decimal units plus the in-sample standardised residuals.
    """
    r = np.asarray(returns, dtype=float) * SCALE
    if r.ndim != 1 or len(r) < 250 or not np.isfinite(r).all():
        raise ValueError("GJR-GARCH needs at least 250 finite returns")
    start_var = float(np.var(r))
    starts = [np.array([r.mean(), 0.02 * start_var, 0.02, 0.10, 0.88, 8.0])]
    if previous is not None:
        starts.insert(0, np.array([previous["mu"] * SCALE, previous["omega"] * SCALE**2, previous["alpha"],
                                   previous["gamma"], previous["beta"], previous["nu"]]))
    bounds = [(-1.0, 1.0), (1e-6, 10 * start_var), (0.0, 0.5), (0.0, 0.8), (0.0, 0.999), (2.1, 100.0)]
    best = None
    for x0 in starts:
        fit = minimize(_neg_loglik, x0, args=(r, start_var), method="L-BFGS-B", bounds=bounds)
        if best is None or fit.fun < best.fun:
            best = fit
    if best is None or best.fun >= 1e9:
        raise ValueError("GJR-GARCH fit failed")
    mu, omega, alpha, gamma, beta, nu = best.x
    var = _variance_path(r - mu, omega, alpha, gamma, beta, start_var)
    z = (r - mu) / np.sqrt(var)
    return {"mu": mu / SCALE, "omega": omega / SCALE**2, "alpha": alpha, "gamma": gamma, "beta": beta,
            "nu": nu, "persistence": alpha + gamma / 2 + beta, "start_var": start_var / SCALE**2,
            "loglik": -best.fun, "converged": bool(best.success), "residuals": z}


def filter_volatility(returns: np.ndarray, params: dict) -> np.ndarray:
    """One-step-ahead sigma for every position: sigma[t] uses returns before t only.

    Returns len(returns) + 1 values; the last one is the forecast for the next, unseen day.
    """
    r = np.asarray(returns, dtype=float)
    e = np.append(r - params["mu"], 0.0)
    var = _variance_path(e, params["omega"], params["alpha"], params["gamma"], params["beta"],
                         params["start_var"])
    return np.sqrt(var)
