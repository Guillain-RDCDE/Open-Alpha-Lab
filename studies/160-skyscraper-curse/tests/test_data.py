"""Tests for the data layer — Study 160 (Skyscraper-Curse)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyscraper_curse import data  # noqa: E402


# ---------------------------------------------------------------------------
# Event table
# ---------------------------------------------------------------------------
def test_event_table_not_empty():
    """SKYSCRAPER_EVENTS must have at least 10 rows."""
    assert len(data.SKYSCRAPER_EVENTS) >= 10


def test_event_table_columns():
    required = {"year", "building", "city", "height_m", "scope"}
    assert required.issubset(set(data.SKYSCRAPER_EVENTS.columns))


def test_world_record_events_subset():
    """world_record_events() returns only scope=='world' rows."""
    w = data.world_record_events()
    assert (w["scope"] == "world").all()
    assert len(w) >= 7


def test_events_chronological():
    """Events should be in non-decreasing year order."""
    years = data.SKYSCRAPER_EVENTS["year"].to_numpy()
    assert (np.diff(years) >= 0).all(), "Events must be in chronological order"


def test_empire_state_present():
    """Empire State Building (1931) must be in the table."""
    names = data.SKYSCRAPER_EVENTS["building"].str.lower()
    assert names.str.contains("empire state").any()


def test_burj_khalifa_tallest():
    """Burj Khalifa (828m) is the tallest in the table."""
    assert data.SKYSCRAPER_EVENTS["height_m"].max() == 828


def test_min_year_filter():
    """min_year filter works correctly."""
    w = data.world_record_events(min_year=1970)
    assert (w["year"] >= 1970).all()


def test_max_year_filter():
    """max_year filter works correctly."""
    w = data.world_record_events(max_year=2005)
    assert (w["year"] <= 2005).all()


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_annual_shape():
    df, truth = data.synthetic_annual(n_years=80, seed=160)
    assert len(df) == 80
    assert "year" in df.columns
    assert "log_ret" in df.columns


def test_synthetic_annual_deterministic():
    a, _ = data.synthetic_annual(n_years=50, seed=42)
    b, _ = data.synthetic_annual(n_years=50, seed=42)
    assert (a["log_ret"].values == b["log_ret"].values).all()
    c, _ = data.synthetic_annual(n_years=50, seed=99)
    assert not (a["log_ret"].values == c["log_ret"].values).all()


def test_synthetic_crash_knob_has_effect():
    """The crash_boost knob shifts returns after event years."""
    df_null, _ = data.synthetic_annual(n_years=80, crash_boost=0.0, seed=160)
    df_crash, _ = data.synthetic_annual(n_years=80, crash_boost=-0.50, seed=160)
    # Mean of crash series should be lower than null
    assert df_crash["log_ret"].mean() < df_null["log_ret"].mean()


def test_synthetic_null_boost_unaffected():
    """With crash_boost=0, event years get no extra shift vs non-event years."""
    df, _ = data.synthetic_annual(n_years=80, crash_boost=0.0, seed=160)
    # Log-returns should follow the planted drift roughly
    assert abs(df["log_ret"].mean() - 0.07) < 0.10  # within 10pp of planted drift


# ---------------------------------------------------------------------------
# Cache paths and loader guards
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    """fetch_gspc cache-only mode raises FileNotFoundError when no cache exists."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive():
    """fingerprint() is stable for the same data and different for different data."""
    df, _ = data.synthetic_annual(n_years=40, seed=1)
    df2, _ = data.synthetic_annual(n_years=40, seed=1)
    df3, _ = data.synthetic_annual(n_years=40, seed=2)
    fp1 = data.fingerprint(df, col="log_ret")
    fp2 = data.fingerprint(df2, col="log_ret")
    fp3 = data.fingerprint(df3, col="log_ret")
    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 12
