"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe.

The one test that may touch a real cache parquet is cache-gated, so the suite runs fully
offline in CI."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithium_boom import data  # noqa: E402


def test_synthetic_shape_and_ohlc(random_walk):
    bars, truth = random_walk
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_theme(n_days=500, trend_strength=0.1, seed=7)
    b, _ = data.synthetic_theme(n_days=500, trend_strength=0.1, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_theme(n_days=500, trend_strength=0.1, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_trend_strength_creates_positive_autocorrelation():
    """The trend knob really induces persistence (and 0 induces ~none)."""
    flat, _ = data.synthetic_theme(n_days=4000, trend_strength=0.0, seed=302)
    tr, _ = data.synthetic_theme(n_days=4000, trend_strength=0.15, seed=302)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_tr = np.diff(np.log(tr["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_tr) > 0.10  # positive autocorrelation = momentum/trend


def test_boom_bust_envelope_present(random_walk):
    """The deterministic envelope produces a peak well inside the sample (a real boom-bust),
    not a monotone path — so 'dodge the crash' is a meaningful thing to test."""
    bars, _ = random_walk
    close = bars["close"].to_numpy()
    peak_loc = int(np.argmax(close))
    # the high is reached in the back half but not at the very end (there is a crash leg after)
    assert peak_loc > len(close) * 0.5
    assert peak_loc < len(close) - 50
    assert close[-1] < close[peak_loc]  # gave some of it back


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    bars, _ = random_walk
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_theme(n_days=4000, trend_strength=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# --- cache-gated: only runs when the real LIT parquet is present (offline CI skips it) ---
_LIT_CACHE = data._cache_path("LIT", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_LIT_CACHE), reason="offline CI — no real cache")
def test_real_cache_loads_clean():
    bars = data.fetch_daily("LIT", fetch=False)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars["close"].notna().all()  # the in-progress bar was dropped on fetch
    assert bars.index.is_monotonic_increasing
