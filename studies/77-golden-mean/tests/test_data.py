"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_mean import data  # noqa: E402


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
    a, _ = data.synthetic_daily(n_days=100, mean_rev=0.1, seed=7)
    b, _ = data.synthetic_daily(n_days=100, mean_rev=0.1, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=100, mean_rev=0.1, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_mean_rev_induces_reversion():
    """The mean_rev knob really induces negative autocorrelation in returns."""
    flat, _ = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=77)
    rev, _ = data.synthetic_daily(n_days=500, mean_rev=0.50, seed=77)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["close"].to_numpy()))
    r_rev = np.diff(np.log(rev["close"].to_numpy()))
    # Mean-reverting tape should have more negative autocorrelation
    assert ac(r_rev) < ac(r_flat)


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(random_walk):
    bars, _ = random_walk
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_fib_ratios_constant():
    """The five canonical Fibonacci ratios are present and ordered."""
    assert data.FIB_RATIOS == (0.236, 0.382, 0.500, 0.618, 0.786)
    assert data.FIB_KEY == (0.382, 0.500, 0.618)
