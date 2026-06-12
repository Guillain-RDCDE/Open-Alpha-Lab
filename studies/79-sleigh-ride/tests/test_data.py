"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleigh_ride import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_bars"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, santa_bps=0.0, seed=5)
    b, _ = data.synthetic_daily(n_years=10, santa_bps=0.0, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, santa_bps=0.0, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_different_santa_bps_produces_different_close():
    a, _ = data.synthetic_daily(n_years=30, santa_bps=0.0, seed=79)
    b, _ = data.synthetic_daily(n_years=30, santa_bps=100.0, seed=79)
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_synthetic_santa_window_bars_count(planted_tape):
    bars, truth = planted_tape
    # Each year should contribute exactly 7 santa bars (5 tail + 2 head), except edge years.
    expected_approx = truth["n_years"] * 7
    # Allow some slack for the first and last year boundary.
    assert abs(truth["santa_window_bars"] - expected_approx) <= 14


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=50, santa_bps=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
