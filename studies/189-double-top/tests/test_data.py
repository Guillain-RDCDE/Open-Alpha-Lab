"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from double_top import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — any test that reads a real yfinance parquet must use this.
# ---------------------------------------------------------------------------
_REAL_CACHE = data.DEFAULT_CACHE
_SAMPLE_TICKER = "SPY"
_SAMPLE_CACHE = data._cache_path(_SAMPLE_TICKER, _REAL_CACHE)

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_ohlc(random_walk):
    bars, truth = random_walk
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert bars.index.name == "date"


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, reversal=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, reversal=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, reversal=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_different_reversal_different_series():
    a, _ = data.synthetic_daily(n_days=300, reversal=0.0, seed=189)
    b, _ = data.synthetic_daily(n_days=300, reversal=0.3, seed=189)
    # Different reversal → different return series (though same seed)
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_reversal_induces_negative_autocorrelation():
    """The reversal knob really biases against the prior return's sign."""
    flat, _ = data.synthetic_daily(n_days=2000, reversal=0.0, seed=189)
    rev, _ = data.synthetic_daily(n_days=2000, reversal=0.5, seed=189)

    def lag1_ac(bars):
        r = np.diff(np.log(bars["close"].to_numpy()))
        return np.corrcoef(r[:-1], r[1:])[0, 1]

    ac_flat = lag1_ac(flat)
    ac_rev = lag1_ac(rev)
    # Reverting tape should have more negative autocorrelation
    assert ac_rev < ac_flat
    assert ac_rev < -0.05


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    bars, _ = random_walk
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, reversal=0.0, seed=42)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real-tape tests (skipped in CI if cache absent)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_tape_shape():
    bars = data.fetch_daily(_SAMPLE_TICKER, fetch=False, cache_dir=_REAL_CACHE)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) > 100
    assert (bars["close"] > 0).all()


@requires_cache
def test_real_tape_fingerprint_stable():
    bars = data.fetch_daily(_SAMPLE_TICKER, fetch=False, cache_dir=_REAL_CACHE)
    fp1 = data.fingerprint(bars)
    fp2 = data.fingerprint(bars)
    assert fp1 == fp2
    assert len(fp1) == 12
