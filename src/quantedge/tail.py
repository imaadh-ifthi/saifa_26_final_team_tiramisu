"""Tail functionals with explicit log-loss versus dollar-loss units."""
import numpy as np
from scipy.stats import norm


def tail_weights(losses: np.ndarray, confidence: float) -> np.ndarray:
    """Exact worst (1-c) empirical probability mass, including a fractional boundary."""
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("losses must be a nonempty finite vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    order = np.argsort(-values, kind="stable")
    mass = len(values) * (1 - confidence)
    allocated = np.minimum(1.0, np.maximum(0.0, mass - np.arange(len(values))))
    weights = np.zeros(len(values))
    weights[order] = allocated / mass
    return weights


def empirical_risk(log_returns: np.ndarray, confidence: float) -> dict:
    values = np.asarray(log_returns, dtype=float)
    weights = tail_weights(-values, confidence)
    var_log = max(0.0, -float(np.quantile(values, 1 - confidence)))
    return {
        "var_log": var_log,
        "es_log": max(0.0, float(weights @ -values)),
        "var_simple_pct": float(-np.expm1(-var_log)),
        # ES is the mean transformed loss, NOT the transform of mean log loss.
        "es_simple_pct": max(0.0, float(weights @ -np.expm1(values))),
    }


def normal_risk(volatility: float, confidence: float) -> dict:
    """Zero-mean Gaussian log-return VaR/ES; exact lognormal simple-loss ES."""
    if not np.isfinite(volatility) or volatility < 0 or not 0 < confidence < 1:
        raise ValueError("invalid volatility or confidence")
    z = norm.ppf(confidence)
    var_log = max(0.0, float(z * volatility))
    es_log = float(volatility * norm.pdf(z) / (1 - confidence))
    # logcdf prevents underflow in the conditional exponential moment.
    conditional_growth = np.exp(volatility**2 / 2 + norm.logcdf(-z - volatility)
                                - np.log1p(-confidence))
    return {"var_log": var_log, "es_log": es_log,
            "var_simple_pct": float(-np.expm1(-var_log)),
            "es_simple_pct": max(0.0, float(1 - conditional_growth))}
