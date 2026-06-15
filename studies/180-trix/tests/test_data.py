"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trix import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: real-data tests skip when the per-study cache is absent.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic-tape tests — no cache required
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_ohlc(martingale):
    bars, truth = martingale
    assert len(bars) == truth["n_days"] == truth["n_bars"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: high >= max(open, close) and low <= min(open, close).
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, momentum=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, momentum=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, momentum=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_momentum_creates_autocorrelation():
    """The momentum knob really induces bar-level return persistence (and 0 induces ~none)."""
    flat, _ = data.synthetic_daily(n_days=2000, momentum=0.0, seed=180)
    trend, _ = data.synthetic_daily(n_days=2000, momentum=0.20, seed=180)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_trend = np.diff(np.log(trend["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.05
    assert ac(r_trend) > 0.10


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(martingale):
    bars, _ = martingale
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=1000, momentum=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_index_is_business_days_only(martingale):
    bars, _ = martingale
    assert bars.index.name == "date"
    # All days are Monday-Friday (weekday 0-4).
    assert (bars.index.dayofweek < 5).all()


# ---------------------------------------------------------------------------
# Real-data test — skip when cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_spy_shape():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert len(bars) > 500
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["close"] > 0).all()
