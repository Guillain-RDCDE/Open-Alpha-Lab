"""The synthetic series is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curve_fitting import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
SPY_CACHE = os.path.join(CACHE, "curvefit_SPY.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_prices):
    assert len(null_prices) == 5000
    assert np.isfinite(null_prices.to_numpy()).all()
    assert (null_prices > 0).all()
    assert null_prices.index.is_monotonic_increasing


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_series(n_days=1000, trend_strength=0.001, seed=7)
    b, _ = data.synthetic_series(n_days=1000, trend_strength=0.001, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_series(n_days=1000, trend_strength=0.001, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_trend_strength_adds_autocorrelated_drift(null_prices, trend_prices):
    """A positive trend strength makes daily returns more positively autocorrelated than a
    pure random walk (the structure a crossover can lock onto)."""
    rn = null_prices.pct_change().dropna()
    rt = trend_prices.pct_change().dropna()
    ac_null = rn.autocorr(lag=1)
    ac_trend = rt.autocorr(lag=1)
    assert ac_trend > ac_null


def test_null_returns_have_near_zero_autocorr(null_prices):
    r = null_prices.pct_change().dropna()
    assert abs(r.autocorr(lag=1)) < 0.1


def test_index_uses_short_span_not_overflow():
    """The decorative business-day index is built without overflowing ns-Timestamps."""
    px, _ = data.synthetic_series(n_days=4000, seed=1)
    assert px.index[0].year >= 1900
    assert px.index[-1].year < 2100


def test_fingerprint_stable_and_sensitive(null_prices):
    assert data.fingerprint(null_prices) == data.fingerprint(null_prices)
    other, _ = data.synthetic_series(n_days=5000, trend_strength=0.0, seed=99)
    assert data.fingerprint(null_prices) != data.fingerprint(other)


# ---- real loader: cache contract -------------------------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(ticker="SPY", fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(SPY_CACHE), reason="offline CI: no real cache")
def test_real_series_loads_from_cache():
    s = data.load_real(ticker="SPY", fetch=False)
    assert (s > 0).all()
    assert s.index.is_monotonic_increasing
    assert len(s) > 1000
