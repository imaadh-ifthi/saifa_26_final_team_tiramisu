"""Euler risk allocations on simple returns for the current portfolio."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .tail import tail_weights


def risk_contributions(simple_returns: pd.DataFrame, weights: pd.Series,
                       confidences: list[float], portfolio_value: float) -> pd.DataFrame:
    sample = simple_returns.loc[:, weights.index].dropna()
    if (len(sample) < 2 or not np.isfinite(sample.to_numpy()).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1) or portfolio_value <= 0
            or not confidences or any(not .5 < c < 1 for c in confidences)):
        raise ValueError("Invalid sample, weights, confidence or portfolio value")
    w = weights.to_numpy()
    asset_losses = -sample.to_numpy()
    losses = asset_losses @ w
    covariance = sample.cov().to_numpy()
    volatility = float(np.sqrt(w @ covariance @ w))
    mean = sample.mean().to_numpy()
    order = np.argsort(losses, kind="stable")
    rows = []
    for c in confidences:
        position = (len(losses)-1)*c
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        fraction = position-lo
        empirical_var_gradient = ((1-fraction)*asset_losses[order[lo]] + fraction*asset_losses[order[hi]])
        empirical_es_gradient = tail_weights(losses, c) @ asset_losses
        for method, var_gradient, es_gradient in [
            ("historical_simple", empirical_var_gradient, empirical_es_gradient),
            ("parametric_normal_simple", -mean+norm.ppf(c)*(covariance @ w)/volatility if volatility else -mean,
             -mean+norm.pdf(norm.ppf(c))/(1-c)*(covariance @ w)/volatility if volatility else -mean),
        ]:
            var_total, es_total = float(w @ var_gradient), float(w @ es_gradient)
            for j, asset in enumerate(weights.index):
                rows.append({"method": method, "confidence": c, "asset": asset, "weight": w[j],
                             "marginal_var": var_gradient[j], "component_var": w[j]*var_gradient[j],
                             "marginal_es": es_gradient[j], "component_es": w[j]*es_gradient[j],
                             "component_var_usd": w[j]*var_gradient[j]*portfolio_value,
                             "component_es_usd": w[j]*es_gradient[j]*portfolio_value,
                             "var_risk_share": w[j]*var_gradient[j]/var_total if var_total else np.nan,
                             "es_risk_share": w[j]*es_gradient[j]/es_total if es_total else np.nan,
                             "portfolio_var": var_total, "portfolio_es": es_total,
                             "sample_start": sample.index.min(), "sample_end": sample.index.max(),
                             "observations": len(sample), "volatility_component": w[j]*(covariance @ w)[j]/volatility if volatility else 0})
    return pd.DataFrame(rows)
