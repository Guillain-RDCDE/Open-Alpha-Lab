"""Tests for the data layer — synthetic generator and cycle-week assignment."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fomc_cycle import data  # noqa: E402


# ---------------------------------------------------------------------------
# Cycle-week assignment
# ---------------------------------------------------------------------------
def test_cycle_week_first_fomc_day_is_week_zero():
    """The FOMC statement day itself must be labelled week 0."""
    first_fomc = data.FOMC_DATES[0]  # 1994-02-04
    idx = pd.bdate_range(start="1994-02-01", periods=30, name="date")
    weeks = data.assign_cycle_week_fast(idx)
    first_fomc_pos = idx.get_loc(first_fomc)
    assert weeks.iloc[first_fomc_pos] == 0.0


def test_cycle_weeks_advance_every_five_trading_days():
    """Weeks 0-5 each span exactly 5 consecutive trading days."""
    idx = pd.bdate_range(start="1994-02-04", periods=30, name="date")
    weeks = data.assign_cycle_week_fast(idx)
    w = weeks.dropna()
    # First 5 days should be week 0, next 5 week 1, etc.
    for expected_week in range(6):
        block = w.iloc[expected_week * 5: (expected_week + 1) * 5]
        assert (block == expected_week).all(), (
            f"Expected week {expected_week} in positions {expected_week*5}–{expected_week*5+4}"
        )


def test_cycle_week_resets_at_next_fomc():
    """At the next FOMC meeting the cycle counter resets to 0.

    We use a full index starting from the first FOMC date so the cycle-week
    counter has all trading days available and correctly identifies week 5
    ending just before the next meeting.
    """
    first_fomc = data.FOMC_DATES[0]   # 1994-02-04
    second_fomc = data.FOMC_DATES[1]  # 1994-03-22
    idx = pd.bdate_range(start=first_fomc, end="1994-03-25", name="date")
    weeks = data.assign_cycle_week_fast(idx)
    second_pos = idx.get_loc(second_fomc)
    assert weeks.iloc[second_pos] == 0.0
    # The trading day before the second meeting is the end of week 5
    assert weeks.iloc[second_pos - 1] == 5.0


def test_pre_fomc_days_are_nan():
    """Trading days before the first FOMC date in the schedule get NaN."""
    idx = pd.bdate_range(start="1994-01-03", periods=5, name="date")
    weeks = data.assign_cycle_week_fast(idx)
    assert weeks.isna().all()


def test_cycle_week_range():
    """All non-NaN cycle week labels are in {0, 1, 2, 3, 4, 5}."""
    idx = pd.bdate_range(start="1994-02-01", periods=500, name="date")
    weeks = data.assign_cycle_week_fast(idx)
    valid = weeks.dropna()
    assert valid.isin([0, 1, 2, 3, 4, 5]).all()


def test_fomc_dates_is_sorted_and_nonempty():
    """The hardcoded FOMC schedule is sorted and covers 1994-2026."""
    assert len(data.FOMC_DATES) > 200
    dates = pd.DatetimeIndex(data.FOMC_DATES)
    assert (dates[1:] > dates[:-1]).all(), "FOMC_DATES must be strictly increasing"
    assert dates.min().year == 1994
    assert dates.max().year == 2026


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_world):
    frame, truth = null_world
    assert len(frame) == truth["n_days"]
    assert "ret" in frame.columns
    assert "cycle_week" in frame.columns


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_cycle(n_days=500, even_premium=0.0, seed=10)
    b, _ = data.synthetic_cycle(n_days=500, even_premium=0.0, seed=10)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_cycle(n_days=500, even_premium=0.0, seed=11)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_synthetic_even_premium_plants_gap(null_world, signal_world):
    """The even_premium knob lifts even-week mean returns above the null case."""
    null_frame, _ = null_world
    sig_frame, _ = signal_world

    def _even_mean(f):
        v = f.dropna(subset=["cycle_week"])
        return v[v["cycle_week"].isin([0, 2, 4])]["ret"].mean()

    assert _even_mean(sig_frame) > _even_mean(null_frame) + 5e-4


def test_synthetic_null_has_small_even_odd_gap(null_world):
    """On the null tape (premium=0) the even-odd gap should be noise-sized."""
    frame, _ = null_world
    valid = frame.dropna(subset=["cycle_week"])
    even_mean = valid[valid["cycle_week"].isin([0, 2, 4])]["ret"].mean()
    odd_mean = valid[valid["cycle_week"].isin([1, 3, 5])]["ret"].mean()
    gap_bps = (even_mean - odd_mean) * 1e4
    # With n~2000 and daily vol ~1.1%, the gap should be well within ±5 bps
    assert abs(gap_bps) < 10.0, f"Null-world gap too large: {gap_bps:.2f} bps"


def test_fingerprint_stable_and_content_sensitive(null_world):
    frame, _ = null_world
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_cycle(n_days=2000, even_premium=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


def test_fetch_panel_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_panel(cache_dir=str(tmp_path), fetch=False)
