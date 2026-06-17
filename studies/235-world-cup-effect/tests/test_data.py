"""Tests for the data layer of Study 235 (World-Cup-Effect).

All tests are offline and deterministic — they use the hardcoded World Cup
table and the synthetic generator only. No yfinance or network calls.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_cup_effect import data  # noqa: E402

_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "sp500_daily.parquet")
)


# ---------------------------------------------------------------------------
# World Cup table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(wc_table):
    """We have 19 World Cup editions (1950-2022)."""
    assert len(wc_table) == 19


def test_table_columns(wc_table):
    expected = {"edition", "start_date", "end_date", "host", "winner"}
    assert expected.issubset(set(wc_table.columns))


def test_table_editions_unique(wc_table):
    """Each edition year appears exactly once."""
    assert wc_table["edition"].nunique() == 19


def test_table_start_before_end(wc_table):
    """For each edition, start_date precedes end_date."""
    assert (wc_table["start_date"] < wc_table["end_date"]).all()


def test_table_editions_cover_range(wc_table):
    """Editions span 1950 through 2022."""
    assert wc_table["edition"].min() == 1950
    assert wc_table["edition"].max() == 2022


def test_table_known_winners(wc_table):
    """Spot-check several well-known results."""
    row_1966 = wc_table[wc_table["edition"] == 1966].iloc[0]
    assert "England" in row_1966["winner"]

    row_1970 = wc_table[wc_table["edition"] == 1970].iloc[0]
    assert "Brazil" in row_1970["winner"]

    row_2022 = wc_table[wc_table["edition"] == 2022].iloc[0]
    assert "Argentina" in row_2022["winner"]
    assert "Qatar" in row_2022["host"]


def test_table_is_copy(wc_table):
    """worldcup_table() returns a copy — mutating it does not affect the original."""
    t1 = data.worldcup_table()
    t2 = data.worldcup_table()
    t1.loc[t1.index[0], "winner"] = "XXX"
    assert t2.loc[t2.index[0], "winner"] != "XXX"


def test_table_window_durations_reasonable(wc_table):
    """Each World Cup lasts between 15 and 40 days (older editions were shorter)."""
    durations = (wc_table["end_date"] - wc_table["start_date"]).dt.days
    assert (durations >= 15).all()
    assert (durations <= 40).all()


# ---------------------------------------------------------------------------
# Control windows
# ---------------------------------------------------------------------------
def test_control_windows_length(wc_table):
    """Control windows have the same duration as their matched WC window."""
    ctrl = data.control_windows(wc_table)
    wc_durations = (wc_table["end_date"] - wc_table["start_date"]).dt.days.values
    ctrl_durations = ctrl["duration_days"].values
    np.testing.assert_array_equal(ctrl_durations, wc_durations)


def test_control_windows_2_years_before(wc_table):
    """Control window starts approximately 2 years before the WC start."""
    ctrl = data.control_windows(wc_table)
    for (_, wc_row), (_, ctrl_row) in zip(wc_table.iterrows(), ctrl.iterrows()):
        diff_years = (wc_row["start_date"] - ctrl_row["ctrl_start"]).days / 365.25
        assert 1.9 < diff_years < 2.1


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"date", "return", "in_wc"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=42)
    assert np.allclose(a["return"].to_numpy(), b["return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=2)
    assert not np.allclose(a["return"].to_numpy(), b["return"].to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, WC/non-WC mean returns should be close."""
    df, _ = data.synthetic_daily(n_days=5000, signal_bps=0.0, seed=235)
    wc_mean = df.loc[df["in_wc"], "return"].mean()
    non_wc_mean = df.loc[~df["in_wc"], "return"].mean()
    # With 5000 days and no signal, the gap should be tiny (< 5 bps/day)
    assert abs(wc_mean - non_wc_mean) < 5e-4


def test_synthetic_signal_creates_difference():
    """A planted signal (500 bps/day) should create a detectable difference."""
    df, _ = data.synthetic_daily(n_days=5000, signal_bps=500.0, seed=235)
    wc_mean = df.loc[df["in_wc"], "return"].mean()
    non_wc_mean = df.loc[~df["in_wc"], "return"].mean()
    # 500 bps/day = 5%/day — should be large enough to detect even with noise
    assert wc_mean > non_wc_mean + 0.01


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_daily(n_days=100, signal_bps=0.0, seed=11)
    df1 = df1.rename(columns={"return": "sp500_return"})
    df2 = df2.rename(columns={"return": "sp500_return"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


@pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="real-tape cache absent offline/CI"
)
def test_fetch_sp500_daily_from_cache():
    """If the cache exists, fetch_sp500_daily should load it without yfinance."""
    df = data.fetch_sp500_daily(
        cache_dir=os.path.dirname(_CACHE_PATH)
    )
    assert "sp500_return" in df.columns
    assert len(df) > 100
