"""Data-layer tests — synthetic tape shape, determinism, knob behaviour, and fingerprint.

All tests run offline (no network, no cache files); the synthetic generator is the only
data source.  Tests that load real cached data are cache-gated with ``requires_cache``
and skip on CI where the _cache/ directory is absent.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_indicator import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache gate — every test that reads real data must use this mark.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
_SAMPLE_CACHE = data._cache_path("SPY", _CACHE_DIR)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic generator — shape and consistency
# ---------------------------------------------------------------------------
def test_synthetic_returns_dataframe_with_expected_columns():
    bars, truth = data.synthetic_daily(n_days=200, seed=182)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 200


def test_synthetic_index_is_business_days():
    bars, _ = data.synthetic_daily(n_days=50, seed=182)
    # All index entries must be weekdays (Monday=0 … Friday=4)
    assert (bars.index.dayofweek < 5).all()


def test_synthetic_ohlcv_constraints():
    """OHLCV sanity: high >= max(open, close), low <= min(open, close), volume > 0."""
    bars, _ = data.synthetic_daily(n_days=300, seed=182)
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["volume"] > 0).all()


def test_synthetic_deterministic():
    """Same seed must produce identical frames."""
    b1, t1 = data.synthetic_daily(n_days=100, seed=7)
    b2, t2 = data.synthetic_daily(n_days=100, seed=7)
    assert (b1["close"] == b2["close"]).all()
    assert t1 == t2


def test_synthetic_different_seeds_differ():
    b1, _ = data.synthetic_daily(n_days=100, seed=1)
    b2, _ = data.synthetic_daily(n_days=100, seed=2)
    assert not (b1["close"] == b2["close"]).all()


def test_momentum_knob_increases_autocorrelation():
    """Positive momentum should produce positive return autocorrelation vs the martingale."""
    import pandas as pd

    b_flat, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=42)
    b_trend, _ = data.synthetic_daily(n_days=500, momentum=0.3, seed=42)

    ret_flat = b_flat["close"].pct_change().dropna()
    ret_trend = b_trend["close"].pct_change().dropna()

    ac_flat = float(pd.Series(ret_flat.values).autocorr(lag=1))
    ac_trend = float(pd.Series(ret_trend.values).autocorr(lag=1))
    # Trending tape must exhibit more positive autocorrelation than the random walk.
    assert ac_trend > ac_flat


def test_fingerprint_changes_with_data():
    b1, _ = data.synthetic_daily(n_days=100, seed=1)
    b2, _ = data.synthetic_daily(n_days=100, seed=2)
    assert data.fingerprint(b1) != data.fingerprint(b2)


def test_fingerprint_stable_same_data():
    bars, _ = data.synthetic_daily(n_days=100, seed=99)
    assert data.fingerprint(bars) == data.fingerprint(bars)


# ---------------------------------------------------------------------------
# Real data — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_fetch_returns_ohlcv():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) > 100


@requires_cache
def test_real_fetch_no_timezone():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert bars.index.tz is None
