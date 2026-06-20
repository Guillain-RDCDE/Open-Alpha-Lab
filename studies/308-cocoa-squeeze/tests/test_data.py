"""The synthetic blow-off tape is well-formed, deterministic, and plants the truth it
claims (a parabola when bubble>0, a random walk when bubble=0); the real fetch is
cache-safe and never touches the network in the test-suite."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cocoa_squeeze import data  # noqa: E402


def test_synthetic_shape_and_ohlc(blowoff):
    bars, truth = blowoff
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_blowoff(n_days=600, bubble=1.0, seed=5)
    b, _ = data.synthetic_blowoff(n_days=600, bubble=1.0, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_blowoff(n_days=600, bubble=1.0, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_bubble_creates_a_real_parabola(blowoff, random_walk):
    """bubble>0 must produce a peak that towers over its pre-peak trough; bubble=0 must not."""
    bb, truth = blowoff
    rw, _ = random_walk
    peak = truth["peak_idx"]
    # Around the planted peak, the bubble tape is far above the random-walk tape's level.
    bb_runup = bb["close"].iloc[peak] / bb["close"].iloc[: peak].min()
    rw_runup = rw["close"].iloc[peak] / rw["close"].iloc[: peak].min()
    assert bb_runup > 2.0          # a genuine multi-bagger parabola
    assert bb_runup > 1.5 * rw_runup


def test_random_walk_has_no_persistent_drift_relative_to_bubble(blowoff, random_walk):
    """The null tape's terminal level is nowhere near the bubble tape's peak amplitude."""
    bb, truth = blowoff
    rw, _ = random_walk
    peak = truth["peak_idx"]
    assert bb["close"].iloc[peak] > 2.0 * rw["close"].iloc[peak]


def test_no_timestamp_overflow():
    """A long synthetic tape stays inside the ns-Timestamp range (no pd.date_range blowup)."""
    bars, _ = data.synthetic_blowoff(n_days=5000, bubble=1.0, seed=1)
    assert bars.index[-1].year < 2100


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(blowoff):
    bars, _ = blowoff
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_blowoff(n_days=2500, bubble=1.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# --- cache-gated real-tape test (skips offline / in CI) ---------------------
_CACHE = data._cache_path("CC=F", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="offline CI / no real cache")
def test_real_cocoa_tape_loads_and_is_naive():
    bars = data.fetch_daily("CC=F", fetch=False)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()
