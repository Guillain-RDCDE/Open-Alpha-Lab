"""Tests for the data layer of Study 287 (Easter-Effect).

All tests are offline and deterministic -- they use the Computus algorithm, the
hardcoded Easter session table, and the synthetic generator only.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easter_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Computus -- the anonymous Gregorian algorithm
# ---------------------------------------------------------------------------
def test_easter_sunday_known_dates():
    """Spot-check Easter Sunday against well-known dates."""
    assert data.easter_sunday(1950) == dt.date(1950, 4, 9)
    assert data.easter_sunday(2000) == dt.date(2000, 4, 23)
    assert data.easter_sunday(2024) == dt.date(2024, 3, 31)
    assert data.easter_sunday(2025) == dt.date(2025, 4, 20)


def test_easter_sunday_is_always_sunday():
    """Computus must always land on a Sunday (weekday 6)."""
    for y in range(1900, 2101):
        assert data.easter_sunday(y).weekday() == 6


def test_good_friday_two_days_before():
    for y in range(1950, 2026):
        assert data.good_friday(y) == data.easter_sunday(y) - dt.timedelta(days=2)
        assert data.good_friday(y).weekday() == 4  # Friday


def test_easter_monday_one_day_after():
    for y in range(1950, 2026):
        assert data.easter_monday(y) == data.easter_sunday(y) + dt.timedelta(days=1)
        assert data.easter_monday(y).weekday() == 0  # Monday


# ---------------------------------------------------------------------------
# Hardcoded Easter session table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(easter_table):
    """76 Easter weekends (1950-2025)."""
    assert len(easter_table) == 76


def test_table_columns(easter_table):
    expected = {"year", "easter", "good_friday", "pre_date", "pre_wd", "post_date", "post_wd"}
    assert expected.issubset(set(easter_table.columns))


def test_table_years_sequential(easter_table):
    assert list(easter_table["year"]) == list(range(1950, 2026))


def test_table_easter_matches_computus(easter_table):
    """The frozen 'easter' column must equal the Computus algorithm -- self-checking."""
    for _, row in easter_table.iterrows():
        assert str(data.easter_sunday(int(row["year"]))) == row["easter"]
        assert str(data.good_friday(int(row["year"]))) == row["good_friday"]


def test_table_pre_is_thursday_post_is_monday(easter_table):
    """The pre-holiday session is always Maundy Thursday; post is Easter Monday."""
    assert set(easter_table["pre_wd"].unique()) == {3}
    assert set(easter_table["post_wd"].unique()) == {0}


def test_table_pre_before_gf_post_after_easter(easter_table):
    """pre_date strictly before Good Friday; post_date strictly after Easter Sunday."""
    for _, row in easter_table.iterrows():
        pre = pd.Timestamp(row["pre_date"])
        post = pd.Timestamp(row["post_date"])
        gf = pd.Timestamp(row["good_friday"])
        es = pd.Timestamp(row["easter"])
        assert pre < gf
        assert post > es
        assert pre.weekday() == row["pre_wd"]
        assert post.weekday() == row["post_wd"]


def test_easter_table_is_copy():
    """easter_table() returns a copy -- mutating it does not affect the original."""
    t1 = data.easter_table()
    t2 = data.easter_table()
    t1.loc[0, "pre_wd"] = 9
    assert t2.loc[0, "pre_wd"] != 9


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert set(df.columns) >= {"ret", "is_pre"}
    assert df["is_pre"].sum() == 76  # 1950-2025 inclusive


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_premium():
    """On the null tape, pre vs non-pre means should be close."""
    df, _ = data.synthetic_daily(start_year=1900, end_year=2150, premium_bps=0.0, seed=287)
    pre = df.loc[df["is_pre"], "ret"].mean()
    non = df.loc[~df["is_pre"], "ret"].mean()
    assert abs(pre - non) < 0.001  # < 10 bps


def test_synthetic_signal_creates_premium():
    """A planted premium should lift the pre-holiday mean above the rest."""
    df, _ = data.synthetic_daily(start_year=1900, end_year=2150, premium_bps=100.0, seed=287)
    pre = df.loc[df["is_pre"], "ret"].mean()
    non = df.loc[~df["is_pre"], "ret"].mean()
    assert pre > non + 0.005  # premium clearly visible


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=10)
    df2, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=11)
    assert data.fingerprint(df1["ret"]) == data.fingerprint(df1["ret"])
    assert data.fingerprint(df1["ret"]) != data.fingerprint(df2["ret"])


# ---------------------------------------------------------------------------
# Real-data accessors raise cleanly without the cache (offline guarantee)
# ---------------------------------------------------------------------------
def test_fetch_gspc_daily_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(fetch=False, cache_dir=str(tmp_path))


def test_build_event_frame_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.build_event_frame(fetch=False, cache_dir=str(tmp_path))


def test_unconditional_daily_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.unconditional_daily(fetch=False, cache_dir=str(tmp_path))
