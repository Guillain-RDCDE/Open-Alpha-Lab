"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice_paper import data  # noqa: E402


def test_synthetic_shape_and_ohlc(random_walk):
    bars, truth = random_walk
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
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


def test_reversion_creates_mean_reversion():
    """The reversion knob really induces daily return mean-reversion (and 0 induces ~none)."""
    flat, _ = data.synthetic_daily(n_days=1000, reversion=0.0, seed=76)
    rev, _ = data.synthetic_daily(n_days=1000, reversion=0.40, seed=76)
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_rev = np.diff(np.log(rev["close"].to_numpy()))
    # Mean-reversion = negative lag-1 autocorrelation.
    ac_flat = np.corrcoef(r_flat[:-1], r_flat[1:])[0, 1]
    ac_rev = np.corrcoef(r_rev[:-1], r_rev[1:])[0, 1]
    assert abs(ac_flat) < 0.10, f"flat tape has AC={ac_flat:.3f}"
    assert ac_rev < -0.05, f"reverting tape has AC={ac_rev:.3f} — expected negative"


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    bars, _ = random_walk
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, reversion=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
