import numpy as np


import pandas as pd


import pytest


from scipy.stats import norm, chi2


from src.quantedge.kpis import max_drawdown


from src.quantedge.config import load_config


from src.quantedge.var import build_var_suite


def test_initial_drawdown_includes_starting_wealth():
    assert max_drawdown(pd.Series([-.1, 0.])) == pytest.approx(-.1)
