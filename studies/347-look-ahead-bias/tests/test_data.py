"""The synthetic price tape is well-formed and deterministic; the real loader is
cache-safe and offline by default."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look_ahead_bias import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
REAL_CACHE = os.path.join(CACHE, "lab_prices_SPY.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_prices):
    prices, truth = null_prices
    assert isinstance(prices, pd.Series)
    assert len(prices) == truth["n_days"]
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices > 0).all()


def test_synthetic_index_no_overflow(null_prices):
    """The decorative business-day index is built without overflowing ns-Timestamps."""
    prices, _ = null_prices
    assert prices.index.is_monotonic_increasing
    assert prices.index[0].year >= 1900
    assert prices.index[-1].year < 2200


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_prices(n_days=500, edge=0.002, seed=7)
    b, _ = data.synthetic_prices(n_days=500, edge=0.002, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_prices(n_days=500, edge=0.002, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_null_is_a_random_walk(null_prices):
    """At edge=0 daily log-returns have ~no lag-1 autocorrelation (a random walk)."""
    prices, _ = null_prices
    lr = np.diff(np.log(prices.to_numpy()))
    ac1 = np.corrcoef(lr[:-1], lr[1:])[0, 1]
    assert abs(ac1) < 0.10


def test_edge_plants_mean_reversion(edge_prices):
    """A positive edge induces NEGATIVE lag-1 autocorrelation (mean reversion)."""
    prices, _ = edge_prices
    lr = np.diff(np.log(prices.to_numpy()))
    ac1 = np.corrcoef(lr[:-1], lr[1:])[0, 1]
    assert ac1 < -0.02


def test_fingerprint_stable_and_sensitive(null_prices):
    prices, _ = null_prices
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_prices(n_days=3000, edge=0.0, seed=99)
    assert data.fingerprint(prices) != data.fingerprint(other)


# ---- real loader: cache contract -------------------------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="offline CI: no real cache")
def test_real_tape_loads_from_cache():
    s = data.load_real(fetch=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 100
    assert (s > 0).all()
