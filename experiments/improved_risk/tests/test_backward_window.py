import numpy as np
import pandas as pd
import pytest

from experiments.improved_risk.backward_window import (
    _empirical_cache, _scaled_empirical, evaluate_window,
)
from src.quantedge.tail import empirical_risk


def test_cached_scaled_empirical_matches_reference_tail():
    shocks = np.random.default_rng(12).standard_t(5, size=1000) / 100
    confidences = [.90, .95, .99]
    cache = _empirical_cache(shocks, confidences)
    for confidence in confidences:
        for volatility in (.3, 1.0, 1.7):
            expected = empirical_risk(shocks * volatility, confidence)
            actual = _scaled_empirical(cache, confidence, volatility)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], abs=1e-12)


def test_training_and_scoring_dates_must_be_disjoint():
    dates = pd.date_range("2024-01-01", periods=4)
    training = pd.Series([.01, -.01], index=dates[:2])
    testing = pd.Series([-.02, .005], index=dates[1:3])
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_window(training, testing, "bad", {}, {}, {})
