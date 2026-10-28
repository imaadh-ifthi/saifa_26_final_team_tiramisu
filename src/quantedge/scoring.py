"""Comparing risk models with proper scoring rules instead of only pass/fail tests.

A backtest only says "reject" or "not reject". Committees also ask "which model is best?"
ES on its own cannot be scored, but the pair (VaR, ES) can (Fissler and Ziegel 2016).
We use the FZ0 loss of Patton, Ziegel and Chen (2019); lower is better, and a model that
reports the true VaR and ES gets the lowest expected score.

    L = -1/(a*e) * 1[y <= v] * (v - y) + v/e + log(-e) - 1
    y = realised log return, v = -VaR, e = -ES  (both negative numbers), a = 1 - confidence

Then:
* Diebold-Mariano test: is model A's average loss really lower than model B's?
* Model Confidence Set (Hansen, Lunde and Nason 2011): the set of models that cannot be
  told apart from the best one at 90% confidence. Models outside it are beaten.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def fz0_loss(actual_log_return, var_log, es_log, confidence: float) -> np.ndarray:
    a = 1 - confidence
    y = np.asarray(actual_log_return, dtype=float)
    v = -np.asarray(var_log, dtype=float)
    e = -np.maximum(np.maximum(np.asarray(es_log, dtype=float), np.asarray(var_log, dtype=float)), 1e-8)
    v = np.minimum(v, -1e-8)
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (a * e) + v / e + np.log(-e) - 1


def quantile_loss(actual_log_return, var_log, confidence: float) -> np.ndarray:
    """Tick loss for the VaR alone (lower is better)."""
    a = 1 - confidence
    y = np.asarray(actual_log_return, dtype=float)
    v = -np.asarray(var_log, dtype=float)
    return ((y <= v).astype(float) - a) * (v - y)


def _hac_variance(d: np.ndarray, lags: int | None = None) -> float:
    n = len(d)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    x = d - d.mean()
    var = float(x @ x) / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * float(x[k:] @ x[:-k]) / n
    return max(var, 1e-300)


def diebold_mariano(loss_a, loss_b) -> dict:
    """Negative statistic means A has lower (better) average loss than B."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    stat = d.mean() / np.sqrt(_hac_variance(d) / len(d))
    return {"dm_mean_diff": float(d.mean()), "dm_stat": float(stat), "dm_p": float(2 * norm.sf(abs(stat)))}


def model_confidence_set(losses: pd.DataFrame, level: float = 0.10, n_boot: int = 2000,
                         block: int = 10, seed: int = 11) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS with the T_max statistic and a moving-block bootstrap.

    Returns each model's average loss, its MCS p-value and whether it stays in the set.
    """
    names = list(losses.columns)
    x = losses.to_numpy(dtype=float)
    n, m = x.shape
    if n < 5 * block or m < 2 or not np.isfinite(x).all():
        raise ValueError("MCS needs a finite loss matrix with at least two models")
    rng = np.random.default_rng(seed)
    # pre-compute every block mean, then a bootstrap mean is an average of chosen block means
    csum = np.vstack([np.zeros(m), np.cumsum(x, axis=0)])
    block_means = (csum[block:] - csum[:-block]) / block
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    boot_means = block_means[starts].mean(axis=1)          # n_boot x m
    sample_means = x.mean(axis=0)
    alive = list(range(m))
    pvals = np.zeros(m)
    running = 0.0
    while len(alive) > 1:
        idx = np.array(alive)
        mu = sample_means[idx]
        bm = boot_means[:, idx]
        d_bar = mu - mu.mean()                           # each model vs the average of the set
        d_boot = (bm - bm.mean(axis=1, keepdims=True)) - d_bar
        se = np.sqrt((d_boot ** 2).mean(axis=0))
        t = d_bar / se
        t_max = t.max()
        t_boot = (d_boot / se).max(axis=1)
        p = float((t_boot >= t_max).mean())
        running = max(running, p)
        worst = idx[int(np.argmax(t))]
        pvals[worst] = running
        alive.remove(worst)
    pvals[alive[0]] = 1.0
    out = pd.DataFrame({"model": names, "mean_loss": sample_means, "mcs_p": pvals})
    out["in_mcs"] = out.mcs_p >= level
    return out.sort_values("mean_loss").reset_index(drop=True)
