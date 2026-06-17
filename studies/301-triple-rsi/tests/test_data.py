"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triple_rsi import data  # noqa: E402


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
    a, _ = data.synthetic_daily(n_days=300, reversion=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=300, reversion=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=300, reversion=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_reversion_creates_negative_autocorrelation():
    """The reversion knob really induces anti-persistence (and 0 induces ~none)."""
    flat, _ = data.synthetic_daily(n_days=2000, reversion=0.0, seed=301)
    rev, _ = data.synthetic_daily(n_days=2000, reversion=0.30, seed=301)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_rev = np.diff(np.log(rev["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_rev) < -0.10  # negative autocorrelation = mean-reversion


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin):
    bars, _ = coin
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=8000, reversion=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
