"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher_transform import data  # noqa: E402

# ---------------------------------------------------------------------------
# Locate study-local _cache so the cache-gate guard knows what to check.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REAL_CACHE_PATH = os.path.join(STUDY_CACHE, "bars_SPY_1d.parquet")

requires_cache = pytest.mark.skipif(
    not os.path.exists(REAL_CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic-tape tests — always run (no cache needed)
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_ohlc(coin):
    bars, truth = coin
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, mean_rev=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, mean_rev=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, mean_rev=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_mean_rev_induces_negative_autocorrelation():
    """The mean_rev knob really induces negative return autocorrelation."""
    flat, _ = data.synthetic_daily(n_days=1000, mean_rev=0.0, seed=183)
    rev, _ = data.synthetic_daily(n_days=1000, mean_rev=0.40, seed=183)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_rev = np.diff(np.log(rev["close"].to_numpy()))
    assert abs(ac(r_flat)) < 0.08
    assert ac(r_rev) < -0.10


def test_synthetic_momentum_induces_positive_autocorrelation():
    """Negative mean_rev (momentum) induces positive return autocorrelation."""
    mom, _ = data.synthetic_daily(n_days=1000, mean_rev=-0.30, seed=183)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_mom = np.diff(np.log(mom["close"].to_numpy()))
    assert ac(r_mom) > 0.10


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin):
    bars, _ = coin
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real-data tests — skip when cache is absent (CI offline)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_tape_loads_and_has_ohlcv():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    assert not bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["close"] > 0).all()
    assert (bars["high"] >= bars["low"]).all()


@requires_cache
def test_real_tape_fingerprint_is_stable():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    assert data.fingerprint(bars) == data.fingerprint(bars)
