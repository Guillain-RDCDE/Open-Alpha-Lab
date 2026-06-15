"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_soldiers import data  # noqa: E402

# ---------------------------------------------------------------------------
# Guard for any test that reads real cached data
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_SAMPLE_CACHE = data._cache_path("SPY", _CACHE_DIR)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


def test_synthetic_shape_and_ohlc(neutral):
    bars, truth = neutral
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets open and close.
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


def test_synthetic_index_is_business_days(neutral):
    bars, _ = neutral
    assert bars.index.name == "date"
    # Check no weekends in index.
    assert not any(d.dayofweek >= 5 for d in bars.index)


def test_momentum_creates_persistence():
    """The momentum knob really induces day-level return persistence (and 0 does not)."""
    flat, _ = data.synthetic_daily(n_days=800, momentum=0.0, seed=187)
    trend, _ = data.synthetic_daily(n_days=800, momentum=0.35, seed=187)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_trend = np.diff(np.log(trend["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.10
    assert ac(r_trend) > 0.15


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(neutral):
    bars, _ = neutral
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


@requires_cache
def test_real_tape_shape_and_columns():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert set(["open", "high", "low", "close", "volume"]).issubset(bars.columns)
    assert len(bars) > 100
    assert bars.index.tz is None  # daily bars must be timezone-naive
