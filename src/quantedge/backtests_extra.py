"""Extra VaR backtests that go beyond Kupiec + Christoffersen.

* Dynamic Quantile test (Engle and Manganelli 2004). Regress Hit_t = 1[breach] - a on a
  constant, four lagged hits and the VaR forecast. Under a good model nothing should
  predict a breach. This catches clustering over several days, which the one-lag
  Christoffersen test can miss.
* Duration test (Christoffersen and Pelletier 2004). Days between breaches should be
  memoryless (exponential). Fit a Weibull; shape b < 1 means breaches come in bunches.
* Monte Carlo p-values (Dufour 2006 style). Chi-square p-values are only approximate
  when breaches are rare (about 22 in our 2,212 days at 99%). We simulate the exact null.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


def dynamic_quantile(exceptions, var_forecast, alpha: float, lags: int = 4) -> dict:
    hits = np.asarray(exceptions, dtype=float) - alpha
    var = np.asarray(var_forecast, dtype=float)
    n = len(hits)
    if n <= lags + 10:
        raise ValueError("series too short for DQ test")
    y = hits[lags:]
    cols = [np.ones(n - lags)] + [hits[lags - k:n - k] for k in range(1, lags + 1)] + [var[lags:]]
    x = np.column_stack(cols)
    xtx = x.T @ x
    try:
        beta = np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        return {"dq_stat": np.nan, "dq_p": np.nan, "dq_df": x.shape[1]}
    stat = float(beta @ xtx @ beta / (alpha * (1 - alpha)))
    return {"dq_stat": stat, "dq_df": x.shape[1], "dq_p": float(chi2.sf(stat, x.shape[1]))}


def _durations(exceptions):
    hits = np.flatnonzero(np.asarray(exceptions, dtype=bool))
    n = len(exceptions)
    if len(hits) < 2:
        return None, None
    d = [hits[0] + 1] + list(np.diff(hits))
    censored = [hits[0] != 0] + [False] * (len(hits) - 1)
    tail = n - 1 - hits[-1]
    if tail > 0:
        d.append(tail)
        censored.append(True)
    return np.asarray(d, dtype=float), np.asarray(censored, dtype=bool)


def _weibull_ll(log_a, b, d, c):
    a = np.exp(log_a)
    log_s = -(a * d) ** b                         # log survival
    log_f = b * np.log(a) + np.log(b) + (b - 1) * np.log(d) + log_s
    return float(np.where(c, log_s, log_f).sum())


def duration_test(exceptions) -> dict:
    d, c = _durations(exceptions)
    if d is None or (~c).sum() < 3:
        return {"dur_weibull_b": np.nan, "dur_lr": np.nan, "dur_p": np.nan, "dur_status": "too_few_breaches"}
    # restricted model b = 1: exponential, closed form a = events / total time
    a0 = (~c).sum() / d.sum()
    ll0 = _weibull_ll(np.log(a0), 1.0, d, c)
    fit = minimize(lambda x: -_weibull_ll(x[0], np.exp(x[1]), d, c), x0=[np.log(a0), 0.0], method="Nelder-Mead")
    ll1 = max(-fit.fun, ll0)
    lr = 2 * (ll1 - ll0)
    return {"dur_weibull_b": float(np.exp(fit.x[1])), "dur_lr": float(lr), "dur_p": float(chi2.sf(lr, 1)),
            "dur_status": "ok"}


def _kupiec_lr(x, n, a):
    from scipy.special import xlogy
    phat = x / n
    return 2 * (xlogy(x, phat) + xlogy(n - x, 1 - phat) - xlogy(x, a) - xlogy(n - x, 1 - a))


def _independence_lr(hits):
    from scipy.special import xlogy
    prev, nxt = hits[:-1], hits[1:]
    n00 = np.sum((prev == 0) & (nxt == 0)); n01 = np.sum((prev == 0) & (nxt == 1))
    n10 = np.sum((prev == 1) & (nxt == 0)); n11 = np.sum((prev == 1) & (nxt == 1))
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0
    p0, p1 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / (len(hits) - 1)
    un = xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
    re = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    return max(0.0, 2 * (un - re))


def monte_carlo_pvalues(exceptions, alpha: float, n_sim: int = 5000, seed: int = 7) -> dict:
    """Simulated p-values for Kupiec, independence and their sum (joint coverage)."""
    hits = np.asarray(exceptions, dtype=int)
    n = len(hits)
    obs_k = _kupiec_lr(hits.sum(), n, alpha)
    obs_i = _independence_lr(hits)
    rng = np.random.default_rng(seed)
    sims = rng.random((n_sim, n)) < alpha
    sim_k = np.array([_kupiec_lr(s.sum(), n, alpha) for s in sims])
    sim_i = np.array([_independence_lr(s.astype(int)) for s in sims])
    # counting ties as "at least as extreme" keeps the test on the safe (conservative) side
    def pval(obs, sim):
        return float((1 + np.sum(sim >= obs - 1e-12)) / (n_sim + 1))
    return {"mc_kupiec_p": pval(obs_k, sim_k), "mc_independence_p": pval(obs_i, sim_i),
            "mc_joint_p": pval(obs_k + obs_i, sim_k + sim_i)}
