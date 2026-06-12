"""The synthetic tape is well-formed and deterministic; calendar helpers are correct."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from witching_hour import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == truth["n_days"]
    # OHLC sanity: range brackets open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["close"] > 0).all()
    assert (bars["volume"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, vol_premium=0.3, seed=7)
    b, _ = data.synthetic_daily(n_years=10, vol_premium=0.3, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, vol_premium=0.3, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_witching_dates_are_third_fridays():
    wdates = data.witching_dates("2020-01-01", "2022-12-31")
    for d in wdates:
        # Third Friday of March, June, September, or December
        assert d.weekday() == 4, f"{d} is not a Friday"
        assert d.month in (3, 6, 9, 12), f"{d} is not a quarter-end month"
        # It must be the 3rd Friday: day between 15 and 21
        assert 15 <= d.day <= 21, f"{d} day {d.day} is not in 15-21 (3rd Friday range)"


def test_witching_day_mask_count(null_tape):
    bars, _ = null_tape
    idx = pd.DatetimeIndex(bars.index)
    mask = data.witching_day_mask(idx)
    # ~4 witching days per year, 20 years → ~80 days
    assert 60 <= mask.sum() <= 90


def test_witching_week_mask_covers_day(null_tape):
    bars, _ = null_tape
    idx = pd.DatetimeIndex(bars.index)
    day_mask = data.witching_day_mask(idx)
    week_mask = data.witching_week_mask(idx)
    # Every witching day must be covered by the week mask
    assert (day_mask[day_mask].index.isin(week_mask[week_mask].index)).all()
    # Week mask should have roughly 5x more days than the day mask
    assert week_mask.sum() >= day_mask.sum() * 3


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=20, vol_premium=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
