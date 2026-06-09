"""The data layer's one job: keep the article's correct lesson — model returns, not prices."""

import numpy as np
import pandas as pd

from paper_prophet import data


def _random_walk_prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    return pd.Series(p, index=idx, name="Close")


def test_log_returns_scale_and_length():
    p = _random_walk_prices()
    r = data.log_returns(p)
    assert len(r) == len(p) - 1
    # percent scale: a 1% daily move is ~1.0, not ~0.01
    assert 0.3 < r.std() < 3.0


def test_stationarity_pattern_holds():
    # A random-walk price has a unit root (ADF p high); its returns are stationary (ADF p low).
    p = _random_walk_prices()
    chk = data.stationarity_check(p)
    assert chk["price_adf_p"] > 0.05      # unit root: do not model price levels
    assert chk["returns_adf_p"] < 0.05    # stationary: model returns
    assert chk["n_returns"] == chk["n_prices"] - 1
