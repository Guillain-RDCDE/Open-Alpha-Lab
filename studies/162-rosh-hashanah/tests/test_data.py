"""The holiday event table is well-formed; the synthetic tape is deterministic; cache is safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosh_hashanah import data


# ---------------------------------------------------------------------------
# Holiday event table
# ---------------------------------------------------------------------------
def test_holiday_table_length():
    """We have ~47 rows covering 1980–2026."""
    assert len(data.HOLIDAY_DF) >= 40


def test_holiday_table_columns():
    assert set(data.HOLIDAY_DF.columns) >= {"year", "rosh_hashanah", "yom_kippur"}


def test_yom_kippur_always_after_rosh_hashanah():
    diff = (data.HOLIDAY_DF["yom_kippur"] - data.HOLIDAY_DF["rosh_hashanah"]).dt.days
    assert (diff > 0).all()
    # YK is always 9 calendar days after RH (10 Tishrei - 1 Tishrei = 9 days)
    assert (diff == 9).all(), f"Expected all gaps to be 9 days, got: {diff.unique()}"


def test_holiday_years_monotone():
    assert data.HOLIDAY_DF["year"].is_monotonic_increasing


def test_holiday_df_years_cover_1980_to_2026():
    years = set(data.HOLIDAY_DF["year"])
    assert 1980 in years
    assert 2026 in years


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_world):
    df, truth = null_world
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"window_ret", "rest_ret", "annual_ret"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=40, holiday_drag_bp=-50.0, seed=42)
    b, _ = data.synthetic_annual(n_years=40, holiday_drag_bp=-50.0, seed=42)
    assert np.allclose(a["window_ret"].to_numpy(), b["window_ret"].to_numpy())
    c, _ = data.synthetic_annual(n_years=40, holiday_drag_bp=-50.0, seed=99)
    assert not np.allclose(a["window_ret"].to_numpy(), c["window_ret"].to_numpy())


def test_holiday_drag_lowers_window_mean(holiday_drag_world, null_world):
    """A planted drag should reduce the mean window return vs the null."""
    drag_df, _ = holiday_drag_world
    null_df, _ = null_world
    assert drag_df["window_ret"].mean() < null_df["window_ret"].mean()


def test_null_window_similar_to_rest(null_world):
    """Without a planted drag, window and rest means are roughly equal."""
    df, _ = null_world
    diff = abs(df["window_ret"].mean() - df["rest_ret"].mean())
    # In the null, the difference should be small relative to vol
    assert diff < 0.02  # 2 percentage points is well within noise for 80 years


def test_fetch_gspc_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive(null_world):
    df, _ = null_world
    col_df = df.rename(columns={"window_ret": "close"})
    assert data.fingerprint(col_df) == data.fingerprint(col_df)
    other, _ = data.synthetic_annual(n_years=40, seed=7)
    other_col = other.rename(columns={"window_ret": "close"})
    assert data.fingerprint(col_df) != data.fingerprint(other_col)
