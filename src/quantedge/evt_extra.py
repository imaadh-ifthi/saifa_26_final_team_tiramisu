"""Layer 4 extras: data-driven threshold choice, clustering of extremes, and error bars.

1. EQD threshold choice (Murphy, Tawn and Varty 2025, "Automated threshold selection and
   associated inference uncertainty for univariate extremes"). For each candidate threshold,
   bootstrap the excesses, fit a GPD, and measure the average gap between fitted and
   empirical quantiles (a QQ-plot distance). Pick the threshold with the smallest gap.
   It replaces the "adjacent thresholds look stable" rule with one number per threshold.
2. Extremal index (Ferro and Segers 2003, intervals estimator). theta = 1 means extremes
   arrive one at a time; theta = 0.5 means they come in clusters of about 2 days. Raw EVT
   assumes theta = 1, so a low theta explains clustered VaR breaches.
3. Bootstrap confidence interval for the latest EVT VaR/ES, answering the committee's
   question "how confident are you in your risk estimates?".
"""
import numpy as np
from scipy.stats import genpareto


def fit_gpd(excess: np.ndarray) -> tuple[float, float]:
    shape, _, scale = genpareto.fit(excess, floc=0)
    return float(shape), float(scale)


def gpd_quantile(p, shape, scale):
    """Quantile of GPD(shape, scale) at non-exceedance probability p."""
    p = np.asarray(p, dtype=float)
    if abs(shape) < 1e-8:
        return -scale * np.log1p(-p)
    return scale / shape * ((1 - p) ** (-shape) - 1)


def eqd_threshold(losses: np.ndarray, candidate_quantiles, n_boot: int = 50, min_excess: int = 40,
                  seed: int = 3) -> dict:
    """Expected quantile discrepancy for every candidate; returns the winner and the table."""
    x = np.asarray(losses, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for q in sorted(candidate_quantiles):
        u = float(np.quantile(x, q))
        excess = x[x > u] - u
        m = len(excess)
        if m < min_excess:
            rows.append({"threshold_quantile": q, "threshold": u, "excesses": m, "eqd": np.inf})
            continue
        probs = np.arange(1, m + 1) / (m + 1)
        scores = []
        for _ in range(n_boot):
            sample = rng.choice(excess, size=m, replace=True)
            try:
                shape, scale = fit_gpd(sample)
            except Exception:
                continue
            scores.append(np.mean(np.abs(gpd_quantile(probs, shape, scale) - np.quantile(sample, probs))))
        rows.append({"threshold_quantile": q, "threshold": u, "excesses": m,
                     "eqd": float(np.mean(scores)) if scores else np.inf})
    best = min(rows, key=lambda r: r["eqd"])
    if not np.isfinite(best["eqd"]):
        raise ValueError("no candidate threshold has enough excesses")
    return {"best": best, "table": rows}


def extremal_index(series: np.ndarray, threshold: float) -> float:
    """Ferro-Segers intervals estimator for exceedances of `threshold` (no tuning needed)."""
    hits = np.flatnonzero(np.asarray(series, dtype=float) > threshold)
    if len(hits) < 3:
        return float("nan")
    t = np.diff(hits).astype(float)
    n = len(hits)
    if t.max() <= 2:
        theta = 2 * t.sum() ** 2 / ((n - 1) * (t**2).sum())
    else:
        theta = 2 * (t - 1).sum() ** 2 / ((n - 1) * ((t - 1) * (t - 2)).sum())
    return float(min(1.0, theta))


def bootstrap_evt_interval(losses: np.ndarray, threshold_quantile: float, confidence: float,
                           n_boot: int = 500, scale_factor: float = 1.0, location: float = 0.0,
                           seed: int = 5) -> dict:
    """Non-parametric bootstrap of the excesses -> 90% interval for log VaR and log ES.

    For a filtered model pass scale_factor = tomorrow's sigma and location = -mu so the
    residual-scale tail is turned back into a return-scale loss.
    The GARCH parameters themselves are held fixed, so the interval is a lower bound on
    the true uncertainty.
    """
    x = np.asarray(losses, dtype=float)
    u = float(np.quantile(x, threshold_quantile))
    excess = x[x > u] - u
    p_u = len(excess) / len(x)
    a = 1 - confidence
    rng = np.random.default_rng(seed)

    def risk(sample):
        shape, scale = fit_gpd(sample)
        var = u + (scale * np.log(p_u / a) if abs(shape) < 1e-8 else scale / shape * ((p_u / a) ** shape - 1))
        es = (var + scale - shape * u) / (1 - shape) if shape < 1 else np.inf
        return location + scale_factor * var, location + scale_factor * es, shape

    point = risk(excess)
    draws = np.array([risk(rng.choice(excess, size=len(excess), replace=True)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(draws[:, :2], [5, 95], axis=0)
    return {"var_log": point[0], "es_log": point[1], "shape": point[2],
            "var_log_5": lo[0], "var_log_95": hi[0], "es_log_5": lo[1], "es_log_95": hi[1],
            "excesses": len(excess)}
