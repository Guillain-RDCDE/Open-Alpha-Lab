"""Tests for the data layer of Study 299 (Keynote-Drift).

All tests are offline and deterministic -- they use the hardcoded keynote table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keynote_drift import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded keynote table
# ---------------------------------------------------------------------------
def test_table_has_events(keynotes):
    """We have a healthy number of keynotes (40-60)."""
    assert 40 <= len(keynotes) <= 60


def test_table_columns(keynotes):
    expected = {"date", "event_type", "label"}
    assert expected.issubset(set(keynotes.columns))


def test_table_dates_are_datetime_and_sorted(keynotes):
    assert str(keynotes["date"].dtype).startswith("datetime64")
    assert list(keynotes["date"]) == sorted(keynotes["date"])


def test_table_event_types_valid(keynotes):
    assert set(keynotes["event_type"].unique()) <= {"WWDC", "September", "Spring"}


def test_table_year_range(keynotes):
    assert keynotes["date"].dt.year.min() == 2008
    assert keynotes["date"].dt.year.max() == 2025


def test_table_known_events(keynotes):
    """Spot-check a couple of well-known keynotes."""
    years = keynotes["date"].dt.year
    # WWDC every year, September event most years.
    assert (keynotes["event_type"] == "WWDC").sum() >= 15
    assert (keynotes["event_type"] == "September").sum() >= 15
    # 2017 had the iPhone X event in September.
    assert ((years == 2017) & (keynotes["event_type"] == "September")).any()


def test_table_is_copy():
    """keynote_table() returns a copy -- mutating it does not affect the original."""
    t1 = data.keynote_table()
    t2 = data.keynote_table()
    t1.loc[0, "event_type"] = "XXX"
    assert t2.loc[0, "event_type"] != "XXX"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert "adj_close" in df.columns
    assert len(df) == truth["n_days"]
    assert (df["adj_close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_prices(drift_bps=0.0, seed=42)
    b, _ = data.synthetic_prices(drift_bps=0.0, seed=42)
    assert np.allclose(a["adj_close"].to_numpy(), b["adj_close"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_prices(drift_bps=0.0, seed=1)
    b, _ = data.synthetic_prices(drift_bps=0.0, seed=2)
    assert not np.allclose(a["adj_close"].to_numpy(), b["adj_close"].to_numpy())


def test_synthetic_null_has_no_keynote_drift():
    """On the null tape, returns around keynotes carry no special structure."""
    from keynote_drift import strategy as st
    df, _ = data.synthetic_prices(drift_bps=0.0, seed=299)
    kt = data.keynote_table()
    cs = st.car_stats(df, kt, n_permutations=300, seed=299)
    # Abnormal pre-CAR should be small (within ~1pp) when nothing is planted.
    assert abs(cs["mean_pre_car"]) < 0.01


def test_synthetic_signal_creates_pre_drift():
    """A strong planted drift should show up as a positive abnormal pre-CAR."""
    from keynote_drift import strategy as st
    df, _ = data.synthetic_prices(drift_bps=100.0, seed=299)
    kt = data.keynote_table()
    cs = st.car_stats(df, kt, n_permutations=300, seed=299)
    assert cs["mean_pre_car"] > 0.01
    assert cs["t_pre"] > 2.0


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_prices(drift_bps=0.0, seed=10)
    df2, _ = data.synthetic_prices(drift_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_prices_raises_without_cache(tmp_path):
    """Without the cache, fetch_prices(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_prices(fetch=False, cache_path=str(tmp_path / "nope.parquet"))
