"""Tests for the data layer of Study 296 (Oscars-Effect).

All tests are offline and deterministic — they use the hardcoded Oscar table and
the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oscars_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Oscar table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(oscar_table):
    """31 ceremonies (67th through 97th, 1995–2025)."""
    assert len(oscar_table) == 31


def test_table_columns(oscar_table):
    expected = {"ceremony", "film_year", "date", "best_pic", "genre"}
    assert expected.issubset(set(oscar_table.columns))


def test_table_ceremony_numbers_sequential(oscar_table):
    assert list(oscar_table["ceremony"]) == list(range(67, 98))


def test_table_dates_are_datetime_and_ordered(oscar_table):
    assert pd.api.types.is_datetime64_any_dtype(oscar_table["date"])
    assert oscar_table["date"].is_monotonic_increasing


def test_table_dates_span_window(oscar_table):
    assert oscar_table["date"].min().year == 1995
    assert oscar_table["date"].max().year == 2025


def test_table_known_winners(oscar_table):
    """Spot-check several well-known Best Picture winners."""
    row67 = oscar_table[oscar_table["ceremony"] == 67].iloc[0]
    assert "Forrest Gump" in row67["best_pic"]
    row96 = oscar_table[oscar_table["ceremony"] == 96].iloc[0]
    assert "Oppenheimer" in row96["best_pic"]
    row97 = oscar_table[oscar_table["ceremony"] == 97].iloc[0]
    assert "Anora" in row97["best_pic"]


def test_table_film_year_one_before_ceremony(oscar_table):
    """The ceremony in year Y honors films of year Y-1."""
    yrs = oscar_table["date"].dt.year.to_numpy()
    fy = oscar_table["film_year"].to_numpy()
    assert np.all(yrs - fy == 1)


def test_table_is_copy():
    t1 = data.oscar_table()
    t2 = data.oscar_table()
    t1.loc[0, "best_pic"] = "XXX"
    assert t2.loc[0, "best_pic"] != "XXX"


# ---------------------------------------------------------------------------
# Event-window construction
# ---------------------------------------------------------------------------
def test_next_trading_day_is_strictly_after():
    idx = pd.bdate_range("2020-01-01", periods=20)
    d = data.next_trading_day(pd.Timestamp("2020-01-05"), idx)  # Sunday
    assert d > pd.Timestamp("2020-01-05")
    assert d in idx


def test_event_windows_shape_and_no_lookahead(synthetic_null):
    ret, cer, _ = synthetic_null
    w = data.event_windows(ret, cer, pre=1, post=3)
    # rel_day spans -1..+3 → 5 rows per ceremony at most
    assert set(w["rel_day"].unique()).issubset({-1, 0, 1, 2, 3})
    # event day 0 must be strictly after the ceremony date (one execution lag)
    for cnum, grp in w.groupby("ceremony"):
        cdate = cer.loc[cer["ceremony"] == cnum, "date"].iloc[0]
        day0 = grp.loc[grp["rel_day"] == 0, "date"]
        assert (day0 > cdate).all()


def test_event_windows_one_event_per_ceremony(synthetic_null):
    ret, cer, _ = synthetic_null
    w = data.event_windows(ret, cer, pre=1, post=3)
    day0 = w[w["rel_day"] == 0]
    assert len(day0) == day0["ceremony"].nunique()


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    ret, cer, truth = synthetic_null
    assert len(ret) == truth["n_obs"]
    assert truth["n_planted"] == truth["n_ceremonies"]


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=42)
    b, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_different_seeds_differ():
    a, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=1)
    b, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_null_has_no_event_bump():
    """On the null tape, event-day-0 mean ≈ baseline (no planted effect)."""
    ret, cer, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=296)
    w = data.event_windows(ret, cer, pre=1, post=3)
    ev = w.loc[w["rel_day"] == 0, "ret"].mean()
    base = ret.mean()
    assert abs(ev - base) < 0.01  # within 100 bps with n≈31


def test_synthetic_signal_creates_bump():
    """A strong planted bump lifts the event-day-0 mean above baseline."""
    ret, cer, _ = data.synthetic_daily(post_oscar_bp=300.0, seed=296)
    w = data.event_windows(ret, cer, pre=1, post=3)
    ev = w.loc[w["rel_day"] == 0, "ret"].mean()
    base = ret.mean()
    assert ev > base + 0.01  # well above baseline


def test_fingerprint_stable_and_content_sensitive():
    a, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=10)
    b, _, _ = data.synthetic_daily(post_oscar_bp=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_daily_raises_without_cache(tmp_path, monkeypatch):
    """With every cache location pointed at an empty dir, fetch_daily(fetch=False) raises."""
    monkeypatch.setattr(data, "STUDY_CACHE", str(tmp_path / "study"))
    monkeypatch.setattr(data, "DEFAULT_CACHE", str(tmp_path / "repo"))
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(cache_dir=str(tmp_path / "study"), fetch=False)
