"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obv_divergence import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: high >= max(open, close) and low <= min(open, close).
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert (bars["volume"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=100, price_momentum=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=100, price_momentum=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=100, price_momentum=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_index_is_business_days(null_tape):
    bars, _ = null_tape
    assert bars.index.name == "date"
    # Check no weekends: day of week 5 = Saturday, 6 = Sunday
    assert (bars.index.dayofweek < 5).all()


def test_price_momentum_creates_autocorrelation():
    """The momentum knob really induces return persistence (and 0 induces ~none)."""
    flat, _ = data.synthetic_daily(n_days=1000, price_momentum=0.0, seed=109)
    trend, _ = data.synthetic_daily(n_days=1000, price_momentum=0.25, seed=109)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_trend = np.diff(np.log(trend["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_trend) > 0.10


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, price_momentum=0.0, seed=42)
    assert data.fingerprint(bars) != data.fingerprint(other)
