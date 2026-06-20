"""Tests for the data layer — FOMC schedule, blackout flag, synthetic generator."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fomc_blackout import data  # noqa: E402


# ---------------------------------------------------------------------------
# FOMC schedule
# ---------------------------------------------------------------------------
def test_fomc_dates_sorted_and_span():
    dates = pd.DatetimeIndex(data.FOMC_DATES)
    assert len(dates) > 200
    assert (dates[1:] > dates[:-1]).all(), "FOMC_DATES must be strictly increasing"
    assert dates.min().year == 1994
    assert dates.max().year == 2026


# ---------------------------------------------------------------------------
# Blackout flag
# ---------------------------------------------------------------------------
def test_blackout_flag_is_before_meeting_not_after():
    """The day before a meeting is in the blackout; the day after is not."""
    m = data.FOMC_DATES[50]
    idx = pd.bdate_range(start=m - pd.Timedelta(days=20), end=m + pd.Timedelta(days=20))
    flag = data.in_blackout(idx, window=10)
    before = idx[idx < m][-1]   # last trading day before the meeting
    after = idx[idx > m][0]     # first trading day after the meeting
    assert bool(flag.loc[before]) is True
    assert bool(flag.loc[after]) is False


def test_blackout_window_length_is_calendar_days():
    """Days more than `window` calendar days before a meeting are not flagged."""
    m = data.FOMC_DATES[50]
    far = m - pd.Timedelta(days=15)   # outside a 10-day window
    near = m - pd.Timedelta(days=3)   # inside it
    idx = pd.DatetimeIndex([far, near])
    flag = data.in_blackout(idx, window=10)
    assert bool(flag.iloc[0]) is False
    assert bool(flag.iloc[1]) is True


def test_blackout_fraction_is_plausible():
    """Roughly 8 meetings/yr x ~7 trading days each => ~20-25% of days flagged."""
    idx = pd.bdate_range(start="1995-01-01", end="2020-01-01")
    flag = data.in_blackout(idx, window=10)
    frac = flag.mean()
    assert 0.10 < frac < 0.35, f"blackout fraction {frac:.2f} implausible"


def test_blackout_offset_shifts_the_window():
    """A positive offset moves the window earlier; the flagged days change."""
    idx = pd.bdate_range(start="2000-01-01", end="2010-01-01")
    base = data.in_blackout(idx, window=10, offset=0)
    shifted = data.in_blackout(idx, window=10, offset=30)
    assert not base.equals(shifted)


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_world):
    frame, truth = null_world
    assert len(frame) == truth["n_days"]
    assert "ret" in frame.columns
    assert "blackout" in frame.columns
    assert frame["blackout"].dtype == bool


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_blackout(n_days=1000, seed=7)
    b, _ = data.synthetic_blackout(n_days=1000, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_blackout(n_days=1000, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_synthetic_excess_plants_gap(null_world, signal_world):
    """The blackout_excess knob lifts blackout-day mean above the null case."""
    def _blk_mean(f):
        return f.loc[f["blackout"], "ret"].mean()
    assert _blk_mean(signal_world[0]) > _blk_mean(null_world[0]) + 5e-4


def test_synthetic_null_gap_is_noise_sized(null_world):
    frame, _ = null_world
    blk = frame.loc[frame["blackout"], "ret"].mean()
    oth = frame.loc[~frame["blackout"], "ret"].mean()
    assert abs((blk - oth) * 1e4) < 10.0


def test_synthetic_vol_mult_makes_calm(calm_world):
    """blackout_vol_mult < 1 should make the blackout window genuinely calmer."""
    frame, _ = calm_world
    blk = frame.loc[frame["blackout"], "ret"].std()
    oth = frame.loc[~frame["blackout"], "ret"].std()
    assert blk < oth


def test_fingerprint_stable_and_sensitive(null_world):
    frame, _ = null_world
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_blackout(blackout_excess=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real loader — cache-gated (skips offline)
# ---------------------------------------------------------------------------
_SHARED = os.path.join(data.SHARED_CACHE, "SPY_total_return.parquet")
_LOCAL = os.path.join(data.LOCAL_CACHE, "SPY_total_return.parquet")
_HAVE_CACHE = os.path.exists(_SHARED) or os.path.exists(_LOCAL)


def test_load_real_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """With no cache anywhere, cache-only load_real raises FileNotFoundError."""
    monkeypatch.setattr(data, "LOCAL_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_real(cache_dir=str(tmp_path), fetch=False)


@pytest.mark.skipif(not _HAVE_CACHE, reason="offline CI: no shared SPY cache parquet")
def test_load_real_has_expected_shape():
    frame = data.load_real(fetch=False)
    assert "ret" in frame.columns
    assert "blackout" in frame.columns
    assert frame["blackout"].dtype == bool
    assert len(frame) > 5000
    assert frame.index.is_monotonic_increasing
