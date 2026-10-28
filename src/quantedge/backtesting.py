"""Likelihood-ratio VaR tests and the legacy Basel 99% / 250-day traffic light."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def _bernoulli_ll(failures: int, observations: int, probability: float) -> float:
    return float(xlogy(failures, probability) + xlogy(observations - failures, 1 - probability))


def _exceptions(values) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not len(result) or not np.isin(result, [0, 1]).all():
        raise ValueError("exceptions must be a nonempty boolean or 0/1 vector")
    return result.astype(int)


def kupiec_test(exceptions, confidence: float, significance: float = .05) -> dict:
    hits = _exceptions(exceptions)
    if not 0 < confidence < 1 or not 0 < significance < 1:
        raise ValueError("confidence and significance must be in (0, 1)")
    n, x = len(hits), int(hits.sum())
    lr = max(0.0, 2 * (_bernoulli_ll(x, n, x/n) - _bernoulli_ll(x, n, 1-confidence)))
    p = float(chi2.sf(lr, 1))
    return {"observations": n, "exceptions": x, "exception_rate": x/n,
            "expected_exception_rate": 1-confidence, "kupiec_lr": lr,
            "kupiec_p_value": p, "kupiec_reject": p < significance}


def christoffersen_test(exceptions, significance: float = .05) -> dict:
    hits = _exceptions(exceptions)
    if not 0 < significance < 1:
        raise ValueError("significance must be in (0, 1)")
    counts = {f"n{i}{j}": int(((hits[:-1] == i) & (hits[1:] == j)).sum())
              for i in (0, 1) for j in (0, 1)}
    n00, n01, n10, n11 = (counts[k] for k in ("n00", "n01", "n10", "n11"))
    if not (n00+n01) or not (n10+n11):
        return {**counts, "independence_lr": np.nan, "independence_p_value": np.nan,
                "independence_reject": None, "independence_status": "insufficient_transition_states"}
    p0, p1 = n01/(n00+n01), n11/(n10+n11)
    pooled = (n01+n11)/(len(hits)-1)
    unrestricted = _bernoulli_ll(n01, n00+n01, p0) + _bernoulli_ll(n11, n10+n11, p1)
    restricted = _bernoulli_ll(n01+n11, len(hits)-1, pooled)
    lr = max(0.0, 2*(unrestricted-restricted))
    p = float(chi2.sf(lr, 1))
    return {**counts, "independence_lr": lr, "independence_p_value": p,
            "independence_reject": p < significance, "independence_status": "ok"}


def basel_traffic_light(exceptions, confidence: float) -> dict:
    hits = _exceptions(exceptions)
    result = {"traffic_light_observations": min(250, len(hits)),
              "traffic_light_exceptions": int(hits[-250:].sum())}
    if not np.isclose(confidence, .99):
        return {**result, "traffic_light": "not_applicable"}
    if len(hits) < 250:
        return {**result, "traffic_light": "insufficient_data"}
    x = result["traffic_light_exceptions"]
    return {**result, "traffic_light": "green" if x <= 4 else "yellow" if x <= 9 else "red"}


def backtest_forecasts(forecasts: pd.DataFrame, significance: float = .05) -> pd.DataFrame:
    rows = []
    for (method, confidence), frame in forecasts.groupby(["method", "confidence"], sort=True):
        frame = frame.sort_values("date")
        if frame.date.duplicated().any():
            raise ValueError("Duplicate forecast dates within a model/confidence")
        # Each model must cover consecutive evaluation observations. Do not bridge gaps.
        if "evaluation_index" in frame and (frame.evaluation_index.diff().dropna() != 1).any():
            raise ValueError("Backtest contains gaps in the evaluation sequence")
        hits = frame["exception"].to_numpy()
        kupiec = kupiec_test(hits, confidence, significance)
        independent = christoffersen_test(hits, significance)
        lr_cc = kupiec["kupiec_lr"] + independent["independence_lr"]
        p_cc = float(chi2.sf(lr_cc, 2))
        rows.append({"method": method, "confidence": confidence,
                     "start_date": frame.date.min(), "end_date": frame.date.max(),
                     "significance": significance, **kupiec, **independent,
                     "conditional_coverage_lr": lr_cc, "conditional_coverage_p_value": p_cc,
                     "conditional_coverage_reject": p_cc < significance if np.isfinite(p_cc) else None,
                     **basel_traffic_light(hits, confidence)})
    return pd.DataFrame(rows)
