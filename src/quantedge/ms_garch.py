"""Two-regime Markov-switching GJR-GARCH with Student-t shocks (Haas, Mittnik and Paolella 2004).

The idea: markets seem to have a "calm" mode and a "stressed" mode, and in each mode
volatility behaves differently. A single GARCH has one set of parameters, so it has to
average the two. Here each regime k has its own GJR variance
    h_{k,t} = omega_k + (alpha_k + gamma_k * 1[e_{t-1} < 0]) e_{t-1}^2 + beta_k h_{k,t-1}
and a hidden Markov chain decides which regime we're in (stay probabilities p11 and p22).
Ardia, Bluteau, Boudt and Catania (2018, International Journal of Forecasting) tested this
on hundreds of stocks and found it gave better VaR than single-regime GARCH.

Tomorrow's return is a mix of two Student-t distributions, weighted by how likely each
regime is given everything up to today. Its quantiles give VaR and ES.
In the Haas version each regime's variance is driven by the actual past shocks, not by the
regime path, which keeps the likelihood exact and quick to compute (Hamilton filter).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import t as student_t

SCALE = 100.0
NAMES = ["mu", "omega1", "alpha1", "gamma1", "beta1", "omega2", "alpha2", "gamma2", "beta2", "p11", "p22", "nu"]


def _regime_var(e, omega, alpha, gamma, beta, start_var):
    drive = np.empty_like(e)
    drive[0] = start_var
    drive[1:] = omega + (alpha + gamma * (e[:-1] < 0)) * e[:-1] ** 2
    out = lfilter([1.0], [1.0, -beta], drive)
    out[0] = start_var
    return out


def _t_logpdf(e, var, nu):
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return const - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(e**2 / (var * (nu - 2)))


def _filter(params, r, start_var):
    """Returns (loglik, predicted regime probs for t = 0..n, regime variances for t = 0..n)."""
    mu, o1, a1, g1, b1, o2, a2, g2, b2, p11, p22, nu = params
    e = np.append(r - mu, 0.0)                      # extra row = the next, unseen day
    h = np.column_stack([_regime_var(e, o1, a1, g1, b1, start_var),
                         _regime_var(e, o2, a2, g2, b2, start_var)])
    dens = np.exp(np.column_stack([_t_logpdf(e[:-1], h[:-1, 0], nu), _t_logpdf(e[:-1], h[:-1, 1], nu)]))
    n = len(r)
    pred = np.empty((n + 1, 2))
    # start from the long-run regime probabilities
    pi1 = (1 - p22) / (2 - p11 - p22)
    pred[0] = [pi1, 1 - pi1]
    loglik = 0.0
    p = pred[0]
    for t in range(n):
        joint = p * dens[t]
        total = joint[0] + joint[1]
        if not total > 0 or not np.isfinite(total):
            return -np.inf, None, None
        loglik += np.log(total)
        post = joint / total
        p = np.array([post[0] * p11 + post[1] * (1 - p22), post[0] * (1 - p11) + post[1] * p22])
        pred[t + 1] = p
    return loglik, pred, h


def _neg_loglik(params, r, start_var):
    mu, o1, a1, g1, b1, o2, a2, g2, b2, p11, p22, nu = params
    if (min(o1, o2) <= 0 or min(a1, g1, b1, a2, g2, b2) < 0 or nu <= 2.05
            or a1 + g1 / 2 + b1 >= 0.999 or a2 + g2 / 2 + b2 >= 0.999
            or not (0.5 < p11 < 0.9999 and 0.5 < p22 < 0.9999)):
        return 1e10
    ll, _, _ = _filter(params, r, start_var)
    return -ll if np.isfinite(ll) else 1e10


def fit_ms_gjr(returns, previous: dict | None = None) -> dict:
    """Fit on one window of log returns (decimal). Regime 1 is labelled as the calmer one."""
    r = np.asarray(returns, dtype=float) * SCALE
    if r.ndim != 1 or len(r) < 250 or not np.isfinite(r).all():
        raise ValueError("MS-GARCH needs at least 250 finite returns")
    v = float(np.var(r))
    starts = [np.array([r.mean(), 0.01 * v, 0.01, 0.06, 0.92, 0.10 * v, 0.03, 0.15, 0.80, 0.98, 0.95, 8.0])]
    if previous is not None:
        starts.insert(0, np.array([previous[k] * (SCALE if k == "mu" else SCALE**2 if k.startswith("omega") else 1)
                                   for k in NAMES]))
    bounds = [(-1, 1), (1e-6, 10 * v), (0, 0.5), (0, 0.8), (0, 0.999), (1e-6, 10 * v), (0, 0.5), (0, 0.8),
              (0, 0.999), (0.51, 0.9998), (0.51, 0.9998), (2.1, 100)]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        fit = minimize(_neg_loglik, x0, args=(r, v), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300})
        if best is None or fit.fun < best.fun:
            best = fit
    if best.fun >= 1e9:
        raise ValueError("MS-GARCH fit failed")
    x = best.x.copy()
    # label regimes so that regime 1 has the lower long-run variance
    lr = [x[1] / max(1 - x[2] - x[3] / 2 - x[4], 1e-6), x[5] / max(1 - x[6] - x[7] / 2 - x[8], 1e-6)]
    if lr[0] > lr[1]:
        x = np.array([x[0], *x[5:9], *x[1:5], x[10], x[9], x[11]])
    out = {k: float(val) for k, val in zip(NAMES, x)}
    out["mu"] /= SCALE
    out["omega1"] /= SCALE**2
    out["omega2"] /= SCALE**2
    out.update({"start_var": v / SCALE**2, "loglik": -float(best.fun), "converged": bool(best.success)})
    return out


def predictive(returns, params: dict):
    """Regime probabilities and regime variances for every day in returns plus the next day."""
    x = [params[k] for k in NAMES]
    x[0] *= SCALE
    x[1] *= SCALE**2
    x[5] *= SCALE**2
    _, pred, h = _filter(np.array(x), np.asarray(returns, dtype=float) * SCALE, params["start_var"] * SCALE**2)
    return pred, h / SCALE**2


def mixture_quantiles(prob, var, mu, nu, grid) -> np.ndarray:
    """Quantiles of a two-regime Student-t mixture on the probability grid (numerical inversion)."""
    sd = np.sqrt(np.asarray(var) * (nu - 2) / nu)
    lo = mu + student_t.ppf(grid[0] / 4, nu) * sd.max()
    hi = mu + student_t.ppf(min(0.6, grid[-1] + 0.2), nu) * sd.max()
    x = np.linspace(lo, hi, 6000)
    cdf = prob[0] * student_t.cdf((x - mu) / sd[0], nu) + prob[1] * student_t.cdf((x - mu) / sd[1], nu)
    return np.interp(grid, cdf, x)
