"""Tests for the data layer — Study 188 (Head-Shoulders).

All tests run on the deterministic synthetic tape; real-data tests are cache-gated
and skipped in offline CI.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_shoulders import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: real-data tests require the per-study _cache/ to be populated.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = os.path.join(_CACHE_DIR, "bars_SPY_1d.parquet")

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape — runs everywhere, no network
# ---------------------------------------------------------------------------
def test_synthetic_daily_shape(null_tape):
    bars, truth = null_tape
    assert bars.shape[0] == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.name == "date"


def test_synthetic_daily_ohlc_valid(null_tape):
    bars, _ = null_tape
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["low"] <= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()
    assert (bars["close"] > 0).all()
    assert (bars["volume"] > 0).all()


def test_synthetic_daily_deterministic():
    b1, _ = data.synthetic_daily(n_days=200, seed=42)
    b2, _ = data.synthetic_daily(n_days=200, seed=42)
    assert (b1["close"].values == b2["close"].values).all()


def test_synthetic_daily_different_seeds():
    b1, _ = data.synthetic_daily(n_days=200, seed=1)
    b2, _ = data.synthetic_daily(n_days=200, seed=2)
    assert not (b1["close"].values == b2["close"].values).all()


def test_synthetic_hs_injection_changes_prices():
    b_null, _ = data.synthetic_daily(n_days=500, hs_strength=0.0, seed=10)
    b_hs, _ = data.synthetic_daily(n_days=500, hs_strength=0.15, seed=10)
    # The tapes start from the same RNG seed so they differ where patterns are injected.
    assert not np.allclose(b_null["close"].values, b_hs["close"].values)


def test_fingerprint_changes_with_data():
    b1, _ = data.synthetic_daily(n_days=200, seed=1)
    b2, _ = data.synthetic_daily(n_days=200, seed=2)
    assert data.fingerprint(b1) != data.fingerprint(b2)


def test_fingerprint_length():
    bars, _ = data.synthetic_daily(n_days=100, seed=42)
    fp = data.fingerprint(bars)
    assert len(fp) == 12
    assert fp.isalnum()


# ---------------------------------------------------------------------------
# Real-tape tests — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_daily_shape():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert "close" in bars.columns
    assert len(bars) > 500
    assert bars.index.name == "date"


@requires_cache
def test_real_daily_ohlc_valid():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["close"]).all()
    assert (bars["close"] > 0).all()
