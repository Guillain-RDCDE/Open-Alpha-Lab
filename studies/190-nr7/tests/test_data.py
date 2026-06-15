"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nr7 import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for any test touching real data
# ---------------------------------------------------------------------------
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
_SPY_CACHE = os.path.join(_CACHE, "nr7_SPY_daily.parquet")
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests — always run (no cache required)
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range must bracket its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert (bars["volume"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, seed=42)
    b, _ = data.synthetic_daily(n_days=200, seed=42)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, seed=43)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_nr7_count_recorded(null_tape):
    bars, truth = null_tape
    # Spot-check: truth["nr7_count"] should be > 0 on a real tape.
    assert truth["nr7_count"] > 0


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real-tape tests — skipped if cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_spy_loads_and_has_expected_shape():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE)
    assert not bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    # Expect at least 5 years of daily data.
    assert len(bars) >= 5 * 252
