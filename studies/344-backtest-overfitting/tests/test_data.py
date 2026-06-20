"""The synthetic price generator is well-formed and deterministic; the real loader is
cache- and offline-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_overfitting import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
SPY_CACHE = os.path.join(CACHE, "overfit_SPY.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_prices):
    prices, truth = null_prices
    assert len(prices) == truth["n_days"]
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_prices(n_days=500, edge=0.0001, seed=7)
    b, _ = data.synthetic_prices(n_days=500, edge=0.0001, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_prices(n_days=500, edge=0.0001, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_edge_lifts_the_drift():
    """A positive edge gives a higher terminal price than the null on the same seed."""
    p0, _ = data.synthetic_prices(n_days=2000, edge=0.0, seed=344)
    pe, _ = data.synthetic_prices(n_days=2000, edge=0.0006, seed=344)
    assert pe.iloc[-1] > p0.iloc[-1]


def test_null_has_no_timing_edge_flag(null_prices):
    """The ground truth records that there is no timing edge to find (the whole point)."""
    _, truth = null_prices
    assert truth["has_timing_edge"] is False


def test_index_uses_period_range_not_overflow():
    """The decorative daily index is built without overflowing ns-Timestamps."""
    prices, _ = data.synthetic_prices(n_days=1260, edge=0.0, seed=1)
    assert prices.index.is_monotonic_increasing
    assert prices.index[0].year >= 1900


def test_panel_of_signals_shape():
    mat, truth = data.synthetic_panel_of_signals(n_days=600, n_trials=120, edge=0.0, seed=344)
    assert mat.shape[1] == truth["n_trials"] <= 120
    assert mat.shape[0] <= 600
    assert np.isfinite(mat.to_numpy()).all()


def test_fingerprint_stable_and_sensitive(null_prices):
    prices, _ = null_prices
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_prices(n_days=1260, edge=0.0, seed=99)
    assert data.fingerprint(prices) != data.fingerprint(other)


# ---- real loader: cache contract (cache-gated) -----------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real("SPY", fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(SPY_CACHE), reason="offline CI: no real SPY cache")
def test_real_series_loads_from_cache():
    px = data.load_real("SPY", fetch=False)
    assert len(px) > 100
    assert (px > 0).all()
    assert px.index.is_monotonic_increasing
