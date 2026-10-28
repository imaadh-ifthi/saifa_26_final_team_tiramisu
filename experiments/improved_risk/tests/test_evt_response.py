import numpy as np
import pandas as pd

from experiments.improved_risk.evt_response import detector_path, calibrate_threshold


def test_cusum_uses_close_only_for_next_forecast_and_expires():
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.Series([0, -.03, 0, 0, 0, 0, 0, 0, 0, 0], index=dates)
    volatility = pd.Series(.01, index=dates)
    path = detector_path(returns, volatility, threshold=2., drift=1., active_days=3)
    assert not path.loc[1, "alert_active_for_forecast"]
    assert path.loc[1, "trigger_after_close"]
    assert path.loc[2:4, "alert_active_for_forecast"].all()
    assert not path.loc[5, "alert_active_for_forecast"]
    changed = returns.copy()
    changed.iloc[6] = -.08
    revised = detector_path(changed, volatility, threshold=2., drift=1., active_days=3)
    pd.testing.assert_frame_equal(path.iloc[:6], revised.iloc[:6])
    assert revised.loc[6, "trigger_after_close"]
    assert not revised.loc[6, "alert_active_for_forecast"]
    assert revised.loc[7, "alert_active_for_forecast"]


def test_detector_threshold_calibration_ignores_evaluation_period():
    dates = pd.bdate_range("2010-01-01", periods=600)
    returns = pd.Series(np.random.default_rng(17).normal(0, .01, 600), index=dates)
    volatility = pd.Series(.01, index=dates)
    spec = {"standardized_loss_drift": 1., "threshold_grid_start": .5,
            "threshold_grid_stop": 8., "threshold_grid_step": .25,
            "maximum_training_triggers_per_year": 3., "active_forecast_days": 5}
    end = dates[349]
    before = calibrate_threshold(returns, volatility, end, spec)
    altered = returns.copy()
    altered.iloc[350:] = -.5
    assert calibrate_threshold(altered, volatility, end, spec) == before
