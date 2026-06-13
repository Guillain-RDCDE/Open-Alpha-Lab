"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_spreads import data  # noqa: E402


def test_synthetic_shape_and_columns(null_prices):
    prices, truth = null_prices
    assert len(prices) == truth["n_days"]
    assert set(prices.columns) == {"hyg", "lqd", "ief", "spy"}
    assert (prices > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, spread_signal=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=200, spread_signal=0.0, seed=5)
    assert np.allclose(a["spy"].to_numpy(), b["spy"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, spread_signal=0.0, seed=6)
    assert not np.allclose(a["spy"].to_numpy(), c["spy"].to_numpy())


def test_synthetic_is_business_days_only(null_prices):
    prices, _ = null_prices
    # No weekends (weekday 0=Mon, 4=Fri; 5=Sat, 6=Sun should be absent)
    assert (prices.index.dayofweek < 5).all()


def test_spread_signal_changes_spy_returns():
    """Planted signal must make spy returns different from the null model."""
    null, _ = data.synthetic_daily(n_days=500, spread_signal=0.0, seed=115)
    planted, _ = data.synthetic_daily(n_days=500, spread_signal=2.0, seed=115)
    # SPY prices must diverge
    assert not np.allclose(null["spy"].to_numpy(), planted["spy"].to_numpy())
    # HYG and IEF should be the same (signal only affects SPY)
    assert np.allclose(null["hyg"].to_numpy(), planted["hyg"].to_numpy())


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_prices):
    prices, _ = null_prices
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_daily(n_days=500, spread_signal=0.0, seed=99)
    assert data.fingerprint(prices) != data.fingerprint(other)


def test_load_all_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_all(fetch=False, cache_dir=str(tmp_path))
