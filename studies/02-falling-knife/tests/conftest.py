"""Shared pytest fixtures. Adds the repo root to the path and builds a small,
deterministic OHLC frame so tests never touch the network."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def synth_ohlc():
    """A 600-bar OHLC frame with mild drift and consistent intrabar geometry."""
    rng = np.random.default_rng(42)
    n = 600
    r = 0.0003 + 0.012 * rng.standard_normal(n)
    close = 100.0 * np.cumprod(1.0 + r)
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * (1.0 + 0.003 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1.0 + np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(0.004 * rng.standard_normal(n)))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)
