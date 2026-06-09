"""Shared fixtures — a small, fast synthetic return series and a short walk-forward.

The tests must not depend on the desk cache or run the full ~8,000-day fit, so we build a tiny
stationary return series and grade only a handful of steps. The point of the tests is that the
*port behaves like the article's code* and the *decomposition arithmetic is right* — not to
reproduce the headline numbers (that is verify.py's job).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))


@pytest.fixture(scope="session")
def returns():
    """A short, stationary, white-ish return series in percent (no real forecastability)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=340)
    r = rng.normal(0.03, 1.0, size=len(idx))  # ~3 bp drift, 1% daily vol, in percent
    return pd.Series(r, index=idx, name="ret")


@pytest.fixture(scope="session")
def wf(returns):
    """A short sequential walk-forward (warm-started for speed) over the synthetic series."""
    from paper_prophet.stack import generate_signals

    return generate_signals(returns, lookback=252, max_steps=40)
