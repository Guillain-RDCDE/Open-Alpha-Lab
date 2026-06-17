"""Tests for the data layer of Study 297 (March-Madness).

All tests are offline and deterministic — they use the hardcoded tournament
window table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded tournament window table
# ---------------------------------------------------------------------------
def test_table_columns(tourney_table):
    expected = {"year", "start", "end", "n_calendar_days"}
    assert expected.issubset(set(tourney_table.columns))


def test_table_covers_modern_era(tourney_table):
    """Windows span 1985 (first 64-team bracket) through 2025."""
    assert tourney_table["year"].min() == 1985
    assert tourney_table["year"].max() == 2025


def test_table_skips_2020(tourney_table):
    """2020 tournament was cancelled (COVID-19) — no window in the table."""
    assert 2020 not in set(tourney_table["year"])


def test_table_one_window_per_year(tourney_table):
    """Exactly one window per covered year (no duplicates)."""
    assert tourney_table["year"].is_unique


def test_table_windows_are_three_weeks(tourney_table):
    """Each window is ~19 calendar days (Thu first round -> Mon final)."""
    span = tourney_table["n_calendar_days"]
    assert (span >= 14).all()
    assert (span <= 22).all()


def test_table_start_before_end(tourney_table):
    assert (tourney_table["start"] < tourney_table["end"]).all()


def test_table_windows_in_march_april(tourney_table):
    """Tournament windows live in March/April."""
    assert tourney_table["start"].dt.month.isin([3]).all()
    assert tourney_table["end"].dt.month.isin([3, 4]).all()


def test_tournament_dates_nonempty():
    days = data.tournament_dates("1985-01-01", "2025-12-31")
    assert len(days) > 700  # ~40 windows * ~19 days


def test_is_tournament_mask_aligns():
    idx = pd.date_range("2019-03-15", "2019-04-10", freq="D")
    mask = data.is_tournament(idx)
    assert mask.index.equals(idx)
    # 2019 window is 2019-03-21 .. 2019-04-08 (inclusive)
    assert mask.loc[pd.Timestamp("2019-03-21")]
    assert mask.loc[pd.Timestamp("2019-04-08")]
    assert not mask.loc[pd.Timestamp("2019-03-15")]
    assert not mask.loc[pd.Timestamp("2019-04-10")]


def test_is_tournament_empty_index():
    empty = pd.DatetimeIndex([])
    mask = data.is_tournament(empty)
    assert len(mask) == 0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    ret, mask, truth = synthetic_null
    assert len(ret) == truth["n_obs"]
    assert len(mask) == len(ret)
    assert mask.dtype == bool


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=42)
    b, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_different_seeds_differ():
    a, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=1)
    b, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, in/out mean returns are close (no planted effect)."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=0.0, seed=297
    )
    in_mean = ret[mask].mean()
    out_mean = ret[~mask].mean()
    assert abs(in_mean - out_mean) < 1e-3  # < 10 bps/day


def test_synthetic_signal_creates_drag():
    """A planted negative penalty makes in-tournament returns lower."""
    ret, mask, _ = data.synthetic_daily(
        n_years=80, madness_penalty_bp=-40.0, seed=297
    )
    assert ret[mask].mean() < ret[~mask].mean()


def test_synthetic_frac_tournament_is_small():
    """Tournament days are ~5-6% of all trading days."""
    _, mask, truth = data.synthetic_daily(n_years=41, seed=297)
    assert 0.03 < truth["frac_tournament"] < 0.08


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    a, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=10)
    b, _, _ = data.synthetic_daily(n_years=10, madness_penalty_bp=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real tape loader: offline contract
# ---------------------------------------------------------------------------
def test_fetch_daily_raises_without_any_cache(tmp_path):
    """With no repo cache and fetch=False, fetch_daily raises FileNotFoundError.

    We point cache_dir at an empty tmp dir and temporarily blank the study-local
    cache path so neither source resolves.
    """
    orig = data.STUDY_CACHE
    try:
        data.STUDY_CACHE = str(tmp_path / "nope")
        with pytest.raises(FileNotFoundError):
            data.fetch_daily(cache_dir=str(tmp_path), fetch=False)
    finally:
        data.STUDY_CACHE = orig
