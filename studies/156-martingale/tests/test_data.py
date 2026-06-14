"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from martingale import data  # noqa: E402


def test_synthetic_shape_and_ohlc(random_walk):
    prices, truth = random_walk
    assert len(prices) == truth["n_days"]
    assert list(prices.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: range brackets open and close.
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (prices["high"] >= prices["low"]).all()
    assert (prices["close"] > 0).all()
    assert (prices["volume"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, mu=0.05, seed=5)
    b, _ = data.synthetic_daily(n_days=200, mu=0.05, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, mu=0.05, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_index_is_business_days(random_walk):
    prices, _ = random_walk
    # All index entries are weekdays (0=Mon..4=Fri)
    assert (prices.index.dayofweek < 5).all()


def test_downtrend_falls(downtrend):
    """A mu=-0.20 tape trends downward over 1000 days."""
    prices, _ = downtrend
    # Final price should be below start price most of the time on a strong downtrend.
    start = prices["close"].iloc[0]
    final = prices["close"].iloc[-1]
    assert final < start  # deterministic given seed=156 and strong drift


def test_mean_reversion_stabilises_price(mean_reverting, random_walk):
    """Mean-reverting tape has lower long-run price drift than random walk."""
    mr_prices, _ = mean_reverting
    rw_prices, _ = random_walk
    # Mean-reverting tape stays closer to the start price in relative terms.
    mr_spread = mr_prices["close"].std()
    rw_spread = rw_prices["close"].std()
    # Random walk accumulates variance ~ sigma^2 * t; mean-reverting stays bounded.
    assert mr_spread < rw_spread


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    prices, _ = random_walk
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_daily(n_days=1000, mu=0.0, seed=99)
    assert data.fingerprint(prices) != data.fingerprint(other)
