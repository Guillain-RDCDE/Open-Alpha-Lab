"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtle_trader import data  # noqa: E402


def test_synthetic_shape_and_ohlc(flat_tape):
    bars, truth = flat_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"]  <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, trend=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, trend=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, trend=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_index_is_business_days(flat_tape):
    bars, _ = flat_tape
    # Business days: no Saturday (5) or Sunday (6).
    days_of_week = bars.index.dayofweek
    assert (days_of_week < 5).all()


def test_trend_creates_autocorrelation():
    """The trend knob really induces day-level return persistence (and 0 induces ~none)."""
    flat, _  = data.synthetic_daily(n_days=1000, trend=0.0,  seed=103)
    trnd, _  = data.synthetic_daily(n_days=1000, trend=0.30, seed=103)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_trnd = np.diff(np.log(trnd["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_trnd) > 0.15


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(flat_tape):
    bars, _ = flat_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, trend=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
