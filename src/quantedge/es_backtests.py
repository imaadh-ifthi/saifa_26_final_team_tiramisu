"""Backtests that check Expected Shortfall itself, not only the VaR line.

Kupiec and Christoffersen only count VaR breaches. They cannot tell whether losses
beyond VaR are as big as ES says. These tests look at the size of tail losses.

1. Du and Escanciano (2017, Management Science) - cumulative violations.
   u_t = model probability of a return as bad as today's (the PIT).
   H_t = (a - u_t)/a if u_t <= a, else 0.   A correct model has E[H_t] = a/2.
   Unconditional test: mean of H against a/2.  Conditional test: Box-Pierce on H (clustering).
2. Kratz, Lok and McNeil (2018, Journal of Banking and Finance) - multinomial VaR test.
   ES is roughly an average of several VaRs, so check breaches at 4 levels together.
   They show 4 levels give more power than a single binomial count.
3. McNeil and Frey (2000) exceedance residual test.
   On breach days, (loss - ES) / sigma should average zero. A positive mean means ES is too small.
"""
import numpy as np
from scipy.stats import chi2, norm


def du_escanciano(pit, alpha: float, lags: int = 5) -> dict:
    u = np.asarray(pit, dtype=float)
    if u.ndim != 1 or len(u) < 50 or not np.isfinite(u).all() or not 0 < alpha < 1:
        raise ValueError("need at least 50 finite PIT values and alpha in (0, 1)")
    n = len(u)
    h = np.where(u <= alpha, (alpha - u) / alpha, 0.0)
    mean_h = float(h.mean())
    # Var(H) = a/3 - a^2/4 under a correct model
    z = np.sqrt(n) * (mean_h - alpha / 2) / np.sqrt(alpha * (1 / 3 - alpha / 4))
    p_ues = float(2 * norm.sf(abs(z)))
    centred = h - alpha / 2
    denom = float(centred @ centred)
    rho = np.array([float(centred[k:] @ centred[:-k]) / denom for k in range(1, lags + 1)]) if denom > 0 else np.zeros(lags)
    q_stat = float(n * (rho**2).sum())
    p_ces = float(chi2.sf(q_stat, lags))
    return {"de_mean_H": mean_h, "de_expected_H": alpha / 2, "de_ues_z": float(z), "de_ues_p": p_ues,
            "de_ces_stat": q_stat, "de_ces_lags": lags, "de_ces_p": p_ces,
            "de_direction": "ES too small" if z > 0 else "ES too large or fine"}


def kratz_multinomial(pit, base_confidence: float = 0.975, levels: int = 4) -> dict:
    """Pearson multinomial test of breach counts at `levels` VaR levels between base and 1.

    Levels follow the paper's design a_j = base + (j-1)/N * (1 - base), j=1..N.
    For base 97.5% and N=4: 97.5%, 98.125%, 98.75%, 99.375%.
    """
    u = np.asarray(pit, dtype=float)
    n = len(u)
    conf = base_confidence + np.arange(levels) / levels * (1 - base_confidence)
    tails = np.sort(1 - conf)                          # lower-tail probabilities, small to large
    edges = np.concatenate([[0.0], tails, [1.0]])
    expected_p = np.diff(edges)
    observed = np.histogram(u, bins=edges)[0]
    expected = n * expected_p
    pearson = float(((observed - expected) ** 2 / expected).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(observed > 0, observed * np.log(observed / expected), 0.0)
    lrt = float(2 * terms.sum())
    p = float(chi2.sf(pearson, levels))
    return {"mn_levels": ";".join(f"{c:.5f}" for c in conf), "mn_observed": ";".join(map(str, observed[::-1])),
            "mn_expected": ";".join(f"{e:.1f}" for e in expected[::-1]), "mn_pearson": pearson,
            "mn_pearson_p": p, "mn_lrt": lrt, "mn_lrt_p": float(chi2.sf(lrt, levels)),
            # our own reading of the p-value, shown as a light for the dashboard
            "mn_light": "green" if p >= 0.05 else "yellow" if p >= 0.001 else "red"}


def exceedance_residual_test(actual_log_return, var_log, es_log, sigma, n_boot: int = 5000,
                             seed: int = 42) -> dict:
    """One-sided bootstrap test (McNeil and Frey 2000): H0 mean residual = 0, H1 mean > 0."""
    r, v, e, s = (np.asarray(x, dtype=float) for x in (actual_log_return, var_log, es_log, sigma))
    breach = -r > v
    k = int(breach.sum())
    if k < 5:
        return {"mf_breaches": k, "mf_mean_residual": np.nan, "mf_p": np.nan, "mf_status": "too_few_breaches"}
    resid = ((-r - e) / s)[breach]
    mean = float(resid.mean())
    # bootstrap under the null: centre the residuals, resample, compare t-like statistics
    rng = np.random.default_rng(seed)
    centred = resid - mean
    sd = resid.std(ddof=1)
    t_obs = mean / (sd / np.sqrt(k))
    draws = rng.choice(centred, size=(n_boot, k), replace=True)
    t_boot = draws.mean(axis=1) / (draws.std(axis=1, ddof=1) / np.sqrt(k))
    p = float((1 + np.sum(t_boot >= t_obs)) / (n_boot + 1))
    return {"mf_breaches": k, "mf_mean_residual": mean, "mf_p": p, "mf_status": "ok"}
