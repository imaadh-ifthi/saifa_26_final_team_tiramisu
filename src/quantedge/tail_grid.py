"""Turn any model's lower-tail quantile function into VaR, ES and a PIT value.

Every model in the Layer 3-5 upgrade describes tomorrow's return through q(p), the
p-quantile of the log return, on a fixed grid of small probabilities. From q(p):
    VaR_c   = -q(1-c)
    ES_c    = -(1/a) * integral_0^a q(p) dp,           a = 1-c      (log units)
    ES_c(%) =  (1/a) * integral_0^a (1 - exp(q(p))) dp            (simple-loss units)
    PIT     = F(r) = the p for which q(p) = r   (needed by the ES backtests)
Doing it this way means every model is scored with the same arithmetic.
"""
import numpy as np

# numpy 2 renamed trapz to trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz
TAIL_TOP = 0.30  # we only need the left 30% of the distribution for 90%+ risk measures


def probability_grid(levels=(0.10, 0.05, 0.025, 0.01875, 0.0125, 0.01, 0.00625), points: int = 1200) -> np.ndarray:
    """Log-spaced grid from 1e-6 to TAIL_TOP, with the tail levels inserted exactly."""
    grid = np.geomspace(1e-6, TAIL_TOP, points)
    return np.unique(np.concatenate([grid, np.asarray(levels, dtype=float)]))


def risk_from_quantiles(q: np.ndarray, grid: np.ndarray, confidence: float) -> dict:
    """VaR and ES (log and simple-loss) from quantiles q evaluated on grid."""
    a = 1 - confidence
    k = int(np.searchsorted(grid, a + 1e-15))  # grid points with p <= a
    p, qq = grid[:k], q[:k]
    if not np.isclose(p[-1], a):
        raise ValueError("tail level must be on the probability grid")
    # piece below the first grid point: treat q as flat there (it is 1e-6 wide, tiny)
    head = p[0] * qq[0]
    es_log = -(head + _trapz(qq, p)) / a
    simple = -np.expm1(qq)
    es_simple = (p[0] * simple[0] + _trapz(simple, p)) / a
    var_log = -qq[-1]
    return {"var_log": max(0.0, float(var_log)), "es_log": max(0.0, float(es_log)),
            "var_simple_pct": max(0.0, float(-np.expm1(-var_log))), "es_simple_pct": max(0.0, float(es_simple))}


def pit_from_quantiles(r: float, q: np.ndarray, grid: np.ndarray) -> float:
    """Model probability of a return at or below r. Returns 0.5 when r is outside the left tail
    (above q(TAIL_TOP)); the tail tests only look at values below 0.10, so that is safe."""
    q = np.maximum.accumulate(np.asarray(q, dtype=float))  # guard against tiny numerical dips
    if r >= q[-1]:
        return 0.5
    if r <= q[0]:
        return float(grid[0] / 2)
    return float(np.interp(r, q, grid))


def empirical_quantiles(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear-interpolated empirical quantiles, same convention as np.quantile in the baseline."""
    return np.quantile(np.asarray(sample, dtype=float), grid)


def gpd_tail_quantiles(sample: np.ndarray, grid: np.ndarray, threshold_loss: float, shape: float,
                       scale: float, exceed_prob: float) -> np.ndarray:
    """Return quantiles where the loss tail above threshold_loss is GPD and the rest is empirical.

    For p <= exceed_prob:  loss quantile = u + scale/shape * ((exceed_prob/p)^shape - 1)
    """
    q = empirical_quantiles(sample, grid)
    tail = grid <= exceed_prob
    ratio = exceed_prob / grid[tail]
    if abs(shape) < 1e-8:
        extra = scale * np.log(ratio)
    else:
        extra = scale * np.expm1(shape * np.log(ratio)) / shape
    q[tail] = -(threshold_loss + extra)
    return np.minimum.accumulate(q[::-1])[::-1]  # keep the quantile function non-decreasing
