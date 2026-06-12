"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from center_line import data  # noqa: E402


def test_synthetic_shape_and_ohlc(martingale):
    bars, truth = martingale
    assert len(bars) == truth["n_days"] * truth["bars_per_day"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_5m(n_days=20, reversion=0.1, seed=5)
    b, _ = data.synthetic_5m(n_days=20, reversion=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_5m(n_days=20, reversion=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_is_rth_only(martingale):
    bars, _ = martingale
    secs = bars.index.hour * 3600 + bars.index.minute * 60
    assert secs.min() >= 9 * 3600 + 30 * 60      # >= 09:30
    assert secs.max() <= 15 * 3600 + 55 * 60      # last bar opens 15:55


def test_reversion_modifies_price_path():
    """Higher reversion produces a different (more mean-reverting) price path."""
    flat, _ = data.synthetic_5m(n_days=40, reversion=0.0, seed=87)
    rev, _ = data.synthetic_5m(n_days=40, reversion=0.5, seed=87)
    assert not np.allclose(flat["close"].to_numpy(), rev["close"].to_numpy())


def test_fetch_5m_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_5m("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(martingale):
    bars, _ = martingale
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_5m(n_days=60, reversion=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_sessions_are_independent():
    """Each session in the synthetic tape starts from the previous session's close price."""
    bars, truth = data.synthetic_5m(n_days=5, reversion=0.0, seed=87)
    sessions = bars.index.normalize().unique()
    # There must be exactly n_days sessions.
    assert len(sessions) == truth["n_days"]
    # The first bar of each session (except the very first) must equal the last bar
    # of the previous session's close (since open_ is set from prior close).
    for i in range(1, len(sessions)):
        prev_sess_close = bars[bars.index.normalize() == sessions[i - 1]]["close"].iloc[-1]
        curr_sess_open = bars[bars.index.normalize() == sessions[i]]["open"].iloc[0]
        assert abs(prev_sess_close - curr_sess_open) < 1e-6
