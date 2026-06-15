"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_btc_ratio import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate for real-data tests
# ---------------------------------------------------------------------------
_ETH_CACHE = data._cache_path(data.ETH_TICKER, data.DEFAULT_CACHE)
_BTC_CACHE = data._cache_path(data.BTC_TICKER, data.DEFAULT_CACHE)
_BOTH_CACHED = os.path.exists(_ETH_CACHE) and os.path.exists(_BTC_CACHE)

requires_cache = pytest.mark.skipif(
    not _BOTH_CACHED,
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests — no network, always run
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert set(bars.columns.get_level_values(0)) == {"BTC-USD", "ETH-USD"}
    assert set(bars.columns.get_level_values(1)) == {"open", "high", "low", "close", "volume"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, ratio_momentum=0.1, seed=7)
    b, _ = data.synthetic_daily(n_days=200, ratio_momentum=0.1, seed=7)
    assert np.allclose(a["BTC-USD"]["close"].to_numpy(), b["BTC-USD"]["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, ratio_momentum=0.1, seed=8)
    assert not np.allclose(a["BTC-USD"]["close"].to_numpy(), c["BTC-USD"]["close"].to_numpy())


def test_synthetic_prices_positive(null_tape):
    bars, _ = null_tape
    for ticker in ["BTC-USD", "ETH-USD"]:
        assert (bars[ticker]["close"] > 0).all()
        assert (bars[ticker]["high"] >= bars[ticker][["open", "close"]].max(axis=1) - 1e-9).all()
        assert (bars[ticker]["low"] <= bars[ticker][["open", "close"]].min(axis=1) + 1e-9).all()


def test_synthetic_correlated(null_tape):
    """ETH and BTC log-returns should be positively correlated (~0.80 planted)."""
    bars, _ = null_tape
    btc_r = np.diff(np.log(bars["BTC-USD"]["close"].to_numpy()))
    eth_r = np.diff(np.log(bars["ETH-USD"]["close"].to_numpy()))
    corr = np.corrcoef(btc_r, eth_r)[0, 1]
    assert corr > 0.5, f"Expected positive correlation, got {corr:.3f}"


def test_momentum_induces_ratio_autocorrelation():
    """The ratio_momentum knob really induces serial dependence in the ETH/BTC log-ratio."""
    flat, _ = data.synthetic_daily(n_days=800, ratio_momentum=0.0, seed=209)
    trend, _ = data.synthetic_daily(n_days=800, ratio_momentum=0.35, seed=209)

    def ratio_ac1(b):
        r = np.diff(np.log(b["ETH-USD"]["close"].to_numpy() / b["BTC-USD"]["close"].to_numpy()))
        return np.corrcoef(r[:-1], r[1:])[0, 1]

    ac_flat = ratio_ac1(flat)
    ac_trend = ratio_ac1(trend)
    assert abs(ac_flat) < 0.10, f"Null tape has high AC: {ac_flat:.3f}"
    assert ac_trend > 0.10, f"Trend tape has low AC: {ac_trend:.3f}"


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("ETH-USD", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars, "BTC-USD") == data.fingerprint(bars, "BTC-USD")
    other, _ = data.synthetic_daily(n_days=500, ratio_momentum=0.0, seed=99)
    assert data.fingerprint(bars, "BTC-USD") != data.fingerprint(other, "BTC-USD")


# ---------------------------------------------------------------------------
# Real-data tests — guarded by cache presence
# ---------------------------------------------------------------------------
@requires_cache
def test_real_pair_aligned_and_not_empty():
    bars = data.load_pair(fetch=False)
    assert not bars.empty
    assert set(bars.columns.get_level_values(0)) == {"BTC-USD", "ETH-USD"}
    # ETH history starts ~2017-11-09; with alignment we expect > 2000 days
    assert len(bars) > 2000, f"Expected >2000 days, got {len(bars)}"


@requires_cache
def test_real_prices_positive():
    bars = data.load_pair(fetch=False)
    for ticker in ["BTC-USD", "ETH-USD"]:
        assert (bars[ticker]["close"] > 0).all()
