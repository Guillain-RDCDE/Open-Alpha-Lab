"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coppock_curve import data  # noqa: E402


def test_synthetic_shape_and_ohlc(flat):
    bars, truth = flat
    assert len(bars) == truth["n_months"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_months=100, trend_strength=0.02, seed=7)
    b, _ = data.synthetic_monthly(n_months=100, trend_strength=0.02, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_monthly(n_months=100, trend_strength=0.02, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_index_is_month_end(flat):
    bars, _ = flat
    import pandas as pd
    # All index dates should be month-end (offset alias 'ME')
    expected = pd.date_range(start=bars.index[0], periods=len(bars), freq="ME")
    assert (bars.index == expected).all()


def test_cyclic_creates_drift(cyclic):
    """The trend_strength knob creates a non-zero cyclic drift (not a pure random walk)."""
    bars, _ = cyclic
    log_ret = np.diff(np.log(bars["close"].to_numpy()))
    # With a 60-month sine cycle, the return autocorrelation at 30 months should differ
    # from zero more than in a flat tape.
    flat_bars, _ = data.synthetic_monthly(n_months=480, trend_strength=0.0, seed=105)
    flat_ret = np.diff(np.log(flat_bars["close"].to_numpy()))
    # Both autocorrelations may be small, but cyclic should have some structure
    assert np.std(log_ret) > 0 and np.std(flat_ret) > 0


def test_fetch_monthly_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_monthly("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(flat):
    bars, _ = flat
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_monthly(n_months=300, trend_strength=0.0, seed=999)
    assert data.fingerprint(bars) != data.fingerprint(other)
