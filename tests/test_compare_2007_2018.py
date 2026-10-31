import pandas as pd

from experiments.compare_2007_2018 import EXPECTED_99, rolling_lights


def test_candidate_list_and_rolling_traffic_light():
    assert len(EXPECTED_99) == 15
    dates = pd.bdate_range("2024-01-01", periods=251)
    frame = pd.DataFrame({"date": dates, "method": "historical",
                          "confidence": .99, "exception": [True] * 5 + [False] * 246})
    result = rolling_lights(frame)
    assert len(result) == 2
    assert result.iloc[0].exceptions_250 == 5
    assert result.iloc[0].traffic_light == "yellow"
    assert result.iloc[1].exceptions_250 == 4
    assert result.iloc[1].traffic_light == "green"
