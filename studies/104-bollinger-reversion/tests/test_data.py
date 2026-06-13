"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bollinger_reversion import data  # noqa: E402


def test_synthetic_shape_and_ohlc(random_walk):
    bars, truth = random_walk
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: high brackets open and close; low is below both.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, reversion=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, reversion=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, reversion=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_is_business_days_only(random_walk):
    bars, _ = random_walk
    # All index values are weekdays (Mon=0 .. Fri=4, i.e. dayofweek < 5).
    assert (bars.index.dayofweek < 5).all()


def test_reversion_creates_mean_reversion(random_walk, reverting):
    """The reversion knob really induces negative autocorrelation in returns."""
    bars_flat, _ = random_walk
    bars_rev, _ = reverting
    r_flat = np.diff(np.log(bars_flat["close"].to_numpy()))
    r_rev = np.diff(np.log(bars_rev["close"].to_numpy()))
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    # Flat should be near zero; reverting should be negative (autocorrelation < flat).
    assert ac(r_rev) < ac(r_flat)
    assert ac(r_rev) < 0.0


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    bars, _ = random_walk
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, reversion=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
