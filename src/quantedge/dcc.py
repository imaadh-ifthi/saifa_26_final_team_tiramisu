"""DCC correlations (Engle 2002): let the correlation between assets move day by day.

Why we care: our portfolio leans on bonds (AGG) to cushion stock falls. Usually SPY and AGG
move in opposite directions, so the hedge works. In March 2020 that broke. AGG fell 4.0% on
12 March and 3.8% on 18 March, and the 60-day SPY-AGG correlation went from about -0.5 to
+0.2. A model that keeps the old correlation keeps counting on a hedge that's gone.

How it works. Take each asset's GARCH-standardised shock z_t (a vector). Then
    Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
    R_t = Q_t rescaled so its diagonal is 1   (that's the correlation matrix for day t)
Qbar is the average correlation of the shocks. a is how fast correlation reacts to a new
day, b is how long it remembers. Both are fitted by maximum likelihood on the window.
R_t only uses shocks up to t-1, so it is a real one-day-ahead forecast.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


def _q_path(z, a, b, qbar):
    """Q_t for t = 0..n (n+1 matrices). Q_0 = qbar. Done with a linear filter, so no Python loop."""
    n, k = z.shape
    outer = np.einsum("ti,tj->tij", z, z)
    drive = np.empty((n + 1, k, k))
    drive[0] = qbar
    drive[1:] = (1 - a - b) * qbar + a * outer
    # lfilter gives q_t = drive_t + b * q_{t-1} with q_{-1} = 0, so q_0 = qbar as wanted
    return lfilter([1.0], [1.0, -b], drive, axis=0)


def _to_corr(q):
    d = np.sqrt(np.einsum("tii->ti", q))
    return q / d[:, :, None] / d[:, None, :]


def correlation_path(z, a: float, b: float, qbar=None) -> np.ndarray:
    """One-step-ahead correlation matrices R_0..R_n for shocks z (n x k)."""
    z = np.asarray(z, dtype=float)
    qbar = np.corrcoef(z, rowvar=False) if qbar is None else qbar
    return _to_corr(_q_path(z, a, b, qbar))


def _neg_loglik(params, z, qbar):
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    r = correlation_path(z, a, b, qbar)[:-1]
    sign, logdet = np.linalg.slogdet(r)
    if np.any(sign <= 0):
        return 1e10
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(r, z[:, :, None])[:, :, 0])
    return float(0.5 * (logdet + quad).sum())


def fit_dcc(z) -> dict:
    """Fit a and b on standardised shocks z (n x k)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] < 2 or len(z) < 250 or not np.isfinite(z).all():
        raise ValueError("DCC needs at least 250 rows of finite shocks for two or more assets")
    qbar = np.corrcoef(z, rowvar=False)
    best = None
    for x0 in ([0.02, 0.95], [0.05, 0.90], [0.01, 0.80]):
        fit = minimize(_neg_loglik, x0, args=(z, qbar), method="L-BFGS-B", bounds=[(0.0, 0.5), (0.0, 0.998)])
        if best is None or fit.fun < best.fun:
            best = fit
    a, b = best.x
    if a + b >= 0.999:
        a, b = 0.0, 0.0
    return {"a": float(a), "b": float(b), "qbar": qbar, "converged": bool(best.success)}
