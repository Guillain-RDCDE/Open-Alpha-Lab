"""Shared pytest fixtures. Adds the study root to the path and builds small,
deterministic market + gauge frames so tests never touch the network."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def synth_market():
    """A 600-bar S&P-like OHLC frame with mild drift and clean intrabar geometry."""
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


@pytest.fixture
def synth_gauge(synth_market):
    """A VIX-like frame on the SAME index: a fear gauge that spikes when the
    market falls, so triggers actually fire in tests."""
    r = synth_market["Close"].pct_change().fillna(0.0).to_numpy()
    # Spiky enough that ordinary down days cross 30 and the worst cross 50, so
    # level/spike triggers actually fire across a 600-bar test sample.
    level = 14.0 + 1100.0 * np.clip(-r, 0.0, None)
    level = pd.Series(level, index=synth_market.index).rolling(2, min_periods=1).max()
    out = pd.DataFrame({"Close": level})
    out["vix_prev"] = out["Close"].shift(1)
    out["vix_chg"] = out["Close"] / out["vix_prev"] - 1.0
    return out
