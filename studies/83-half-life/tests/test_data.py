"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_life import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: bar range brackets open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, halving_boost=0.5, seed=10)
    b, _ = data.synthetic_daily(n_years=5, halving_boost=0.5, seed=10)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=5, halving_boost=0.5, seed=11)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_halving_boost_raises_returns(null_tape, boosted_tape):
    """Planting a boost really does inflate cumulative return over halving windows."""
    import pandas as pd
    from half_life.strategy import window_returns
    null_bars, _ = null_tape
    boost_bars, _ = boosted_tape
    # Use only halvings within both tapes' range
    halvings = data.HALVING_DATES[:3]  # first 3 fit a 12-year window starting 2012
    null_ev = window_returns(null_bars, halvings=halvings)
    boost_ev = window_returns(boost_bars, halvings=halvings)
    # Boosted tape mean must exceed null tape mean by a wide margin
    assert boost_ev["log_ret"].mean() > null_ev["log_ret"].mean() + 0.5


def test_halving_dates_ordered():
    """Halving dates are chronologically ordered and cover the expected epochs."""
    for i in range(len(data.HALVING_DATES) - 1):
        assert data.HALVING_DATES[i] < data.HALVING_DATES[i + 1]
    assert data.HALVING_DATES[0].year == 2012
    assert data.HALVING_DATES[-1].year == 2024


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=12, halving_boost=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
