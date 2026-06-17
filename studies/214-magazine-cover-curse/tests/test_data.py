"""Tests for the data layer of Study 214 (Magazine-Cover-Curse).

All tests are offline and deterministic — they use the hardcoded cover
table and the synthetic generator only. No yfinance cache required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magazine_cover_curse import data  # noqa: E402

CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "gspc_monthly.parquet")
)


# ---------------------------------------------------------------------------
# Hardcoded cover table
# ---------------------------------------------------------------------------
def test_cover_table_has_expected_rows(cover_df):
    """We have 37 covers in the hardcoded table."""
    assert len(cover_df) == 37


def test_cover_table_columns(cover_df):
    expected = {"date", "magazine", "headline", "tone", "notes"}
    assert expected.issubset(set(cover_df.columns))


def test_cover_table_tones_valid(cover_df):
    """Only 'euphoric' and 'doom' appear in the tone column."""
    assert set(cover_df["tone"].unique()) == {"euphoric", "doom"}


def test_cover_table_dates_are_timestamps(cover_df):
    """date column should be parseable as Timestamps."""
    assert pd.api.types.is_datetime64_any_dtype(cover_df["date"])


def test_cover_table_date_range(cover_df):
    """Covers span 1979 through 2023."""
    assert cover_df["date"].dt.year.min() == 1979
    assert cover_df["date"].dt.year.max() == 2023


def test_cover_table_tone_split_is_reasonable(cover_df):
    """Both tones have between 10 and 30 covers each."""
    n_doom = (cover_df["tone"] == "doom").sum()
    n_euph = (cover_df["tone"] == "euphoric").sum()
    assert 10 <= n_doom <= 30
    assert 10 <= n_euph <= 30
    assert n_doom + n_euph == 37


def test_cover_table_known_entries(cover_df):
    """Spot-check famous covers are present."""
    # Death of Equities must be there
    death = cover_df[cover_df["headline"].str.contains("Death of Equities", case=False)]
    assert len(death) >= 1
    assert death.iloc[0]["tone"] == "doom"

    # Barron's Melt-Up cover
    melt = cover_df[cover_df["headline"].str.contains("Melt-Up", case=False)]
    assert len(melt) >= 1
    assert melt.iloc[0]["tone"] == "euphoric"


def test_cover_table_returns_copy(cover_df):
    """cover_table() returns a copy — mutating it does not affect the module constant."""
    t1 = data.cover_table()
    t2 = data.cover_table()
    t1.loc[0, "tone"] = "xxx"
    assert t2.loc[0, "tone"] != "xxx"


def test_cover_table_no_empty_headlines(cover_df):
    """No blank headlines."""
    assert (cover_df["headline"].str.strip().str.len() > 0).all()


def test_cover_table_no_duplicate_dates(cover_df):
    """No two covers have exactly the same date string."""
    # date + magazine combination should be unique enough
    combos = cover_df["date"].astype(str) + cover_df["magazine"]
    assert combos.nunique() == len(cover_df)


# ---------------------------------------------------------------------------
# Synthetic event generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_events"]
    assert set(df.columns) >= {"event_id", "tone", "fwd_12m"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=42)
    assert np.allclose(a["fwd_12m"].to_numpy(), b["fwd_12m"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=2)
    assert not np.allclose(a["fwd_12m"].to_numpy(), b["fwd_12m"].to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, doom/euphoric mean returns should be close."""
    df, _ = data.synthetic_events(n_events=1000, signal_bps=0.0, seed=214)
    doom_mean = df.loc[df["tone"] == "doom", "fwd_12m"].mean()
    euph_mean = df.loc[df["tone"] == "euphoric", "fwd_12m"].mean()
    assert abs(doom_mean - euph_mean) < 0.03


def test_synthetic_signal_creates_difference():
    """A planted signal of 3000 bps should create a large doom > euphoric difference."""
    df, _ = data.synthetic_events(n_events=500, signal_bps=3_000.0, seed=214)
    doom_mean = df.loc[df["tone"] == "doom", "fwd_12m"].mean()
    euph_mean = df.loc[df["tone"] == "euphoric", "fwd_12m"].mean()
    assert doom_mean > euph_mean + 0.4


def test_synthetic_tones_valid(synthetic_null):
    df, _ = synthetic_null
    assert set(df["tone"].unique()).issubset({"doom", "euphoric"})


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_stable():
    df, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=10)
    assert data.fingerprint(df) == data.fingerprint(df)


def test_fingerprint_content_sensitive():
    df1, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_events(n_events=30, signal_bps=0.0, seed=11)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real cache guard
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_fetch_gspc_monthly_returns_dataframe():
    """If the cache is present, fetch_gspc_monthly returns a non-empty DataFrame."""
    prices = data.fetch_gspc_monthly()
    assert isinstance(prices, pd.DataFrame)
    assert "close" in prices.columns
    assert len(prices) > 100


@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_compute_forward_returns_columns():
    """compute_forward_returns adds fwd_6m and fwd_12m columns."""
    covers = data.cover_table()
    prices = data.fetch_gspc_monthly()
    df = data.compute_forward_returns(covers, prices, horizons_months=[6, 12])
    assert "fwd_6m" in df.columns
    assert "fwd_12m" in df.columns
    assert "event_month" in df.columns
    # Should have at least some non-NaN values
    assert df["fwd_12m"].notna().sum() > 10
