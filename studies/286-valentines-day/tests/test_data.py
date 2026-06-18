"""Tests for the data layer of Study 286 (Valentines-Day).

All tests are offline and deterministic -- they use the hardcoded Valentine's
trading-date table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valentines_day import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Valentine's trading-date table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(val_table):
    """76 Valentine sessions (1950-2025)."""
    assert len(val_table) == 76


def test_table_columns(val_table):
    expected = {"year", "session_date", "weekday"}
    assert expected.issubset(set(val_table.columns))


def test_table_years_sequential(val_table):
    """Years run 1950..2025 with no gaps."""
    assert list(val_table["year"]) == list(range(1950, 2026))


def test_table_session_on_or_after_feb14(val_table):
    """Every session date is on or after Feb 14 and in February of its year."""
    for _, row in val_table.iterrows():
        d = pd.Timestamp(row["session_date"])
        assert d.year == row["year"]
        assert d.month == 2
        assert d.day >= 14


def test_table_weekday_is_weekday(val_table):
    """Sessions only fall on Mon-Fri (weekday 0..4)."""
    assert set(val_table["weekday"].unique()).issubset({0, 1, 2, 3, 4})


def test_table_known_dates(val_table):
    """Spot-check a few rolled-forward dates."""
    # 1953 Feb 14 was a Saturday -> rolled to Mon Feb 16.
    r53 = val_table[val_table["year"] == 1953].iloc[0]
    assert r53["session_date"] == "1953-02-16"
    # 2025 Feb 14 was a Friday -> Feb 14 itself.
    r25 = val_table[val_table["year"] == 2025].iloc[0]
    assert r25["session_date"] == "2025-02-14"


def test_valentine_table_is_copy():
    """valentine_table() returns a copy -- mutating it does not affect the original."""
    t1 = data.valentine_table()
    t2 = data.valentine_table()
    t1.loc[0, "weekday"] = 9
    assert t2.loc[0, "weekday"] != 9


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert set(df.columns) >= {"ret", "is_valentine"}
    assert df["is_valentine"].sum() == 76  # 1950-2025 inclusive


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_premium():
    """On the null tape, Valentine vs non-Valentine means should be close."""
    df, _ = data.synthetic_daily(start_year=1900, end_year=2150, premium_bps=0.0, seed=286)
    val = df.loc[df["is_valentine"], "ret"].mean()
    non = df.loc[~df["is_valentine"], "ret"].mean()
    assert abs(val - non) < 0.001  # < 10 bps


def test_synthetic_signal_creates_premium():
    """A planted premium should lift the Valentine mean above the rest."""
    df, _ = data.synthetic_daily(start_year=1900, end_year=2150, premium_bps=100.0, seed=286)
    val = df.loc[df["is_valentine"], "ret"].mean()
    non = df.loc[~df["is_valentine"], "ret"].mean()
    assert val > non + 0.005  # premium clearly visible


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=10)
    df2, _ = data.synthetic_daily(start_year=1990, end_year=2000, premium_bps=0.0, seed=11)
    assert data.fingerprint(df1["ret"]) == data.fingerprint(df1["ret"])
    assert data.fingerprint(df1["ret"]) != data.fingerprint(df2["ret"])


def test_fetch_gspc_daily_raises_without_cache(tmp_path):
    """Without the ^GSPC cache, fetch_gspc_daily(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(fetch=False, cache_dir=str(tmp_path))


def test_build_event_frame_raises_without_cache(tmp_path):
    """Without the cache, the event-frame builder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.build_event_frame(fetch=False, cache_dir=str(tmp_path))
