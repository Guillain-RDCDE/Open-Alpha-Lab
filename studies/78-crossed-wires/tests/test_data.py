"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossed_wires import data  # noqa: E402


def test_synthetic_shape_and_ohlc(coin):
    bars, truth = coin
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=100, momentum=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=100, momentum=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=100, momentum=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_daily_is_business_days(coin):
    bars, _ = coin
    # All index entries should be weekdays (Mon=0 … Fri=4)
    assert (bars.index.dayofweek < 5).all()


def test_momentum_creates_autocorrelation():
    """The momentum knob really induces bar-level return persistence (and 0 induces ~none)."""
    flat, _ = data.synthetic_daily(n_days=600, momentum=0.0, seed=78)
    trend, _ = data.synthetic_daily(n_days=600, momentum=0.25, seed=78)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_trend = np.diff(np.log(trend["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_trend) > 0.15


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fetch_5m_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_5m("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin):
    bars, _ = coin
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_synthetic_5m_shape(coin):
    """5-minute tape also works and is deterministic."""
    bars5, truth5 = data.synthetic_5m(n_days=10, momentum=0.0, seed=78)
    assert len(bars5) == truth5["n_days"] * truth5["bars_per_day"]
    assert list(bars5.columns) == ["open", "high", "low", "close", "volume"]
