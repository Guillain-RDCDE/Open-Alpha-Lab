"""Tests for the data layer of Study 284 (Equinox-Effect).

All tests are offline and deterministic — they use the hardcoded equinox/solstice
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equinox_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded equinox/solstice table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(ev_table):
    """99 years (1928–2026) x 4 events = 396 rows."""
    assert len(ev_table) == 396


def test_table_columns(ev_table):
    expected = {"date", "kind", "season"}
    assert expected.issubset(set(ev_table.columns))


def test_table_dates_are_datetime_and_sorted(ev_table):
    assert pd.api.types.is_datetime64_any_dtype(ev_table["date"])
    dates = ev_table["date"].tolist()
    assert dates == sorted(dates)


def test_table_dates_in_range(ev_table):
    assert ev_table["date"].min().year == 1928
    assert ev_table["date"].max().year == 2026


def test_table_kinds_valid(ev_table):
    assert set(ev_table["kind"].unique()) == {"equinox", "solstice"}


def test_table_seasons_valid(ev_table):
    assert set(ev_table["season"].unique()) == {"spring", "summer", "autumn", "winter"}


def test_table_kind_season_consistency(ev_table):
    """Equinoxes are spring/autumn; solstices are summer/winter."""
    eqx = ev_table[ev_table["kind"] == "equinox"]
    sol = ev_table[ev_table["kind"] == "solstice"]
    assert set(eqx["season"].unique()) == {"spring", "autumn"}
    assert set(sol["season"].unique()) == {"summer", "winter"}


def test_table_four_events_per_year(ev_table):
    """Every year has exactly four events, one per season."""
    counts = ev_table.groupby(ev_table["date"].dt.year)["season"].nunique()
    assert (counts == 4).all()


def test_table_known_dates(ev_table):
    """Spot-check well-known equinox/solstice dates (verified vs USNO)."""
    dates = set(ev_table["date"].dt.strftime("%Y-%m-%d"))
    # 2024: Mar 20 equinox, Jun 20 solstice, Sep 22 equinox, Dec 21 solstice.
    assert "2024-03-20" in dates
    assert "2024-06-20" in dates
    assert "2024-09-22" in dates
    assert "2024-12-21" in dates
    row = ev_table[ev_table["date"] == pd.Timestamp("2024-03-20")].iloc[0]
    assert row["kind"] == "equinox"
    assert row["season"] == "spring"


def test_table_months_are_plausible(ev_table):
    """Each season's events fall in the expected month (3/6/9/12)."""
    m = ev_table.assign(month=ev_table["date"].dt.month)
    assert set(m.loc[m["season"] == "spring", "month"]) <= {3}
    assert set(m.loc[m["season"] == "summer", "month"]) <= {6}
    assert set(m.loc[m["season"] == "autumn", "month"]) <= {9}
    assert set(m.loc[m["season"] == "winter", "month"]) <= {12}


def test_equinox_table_is_copy():
    """equinox_table() returns a copy — mutating it does not affect the original."""
    t1 = data.equinox_table()
    t2 = data.equinox_table()
    t1.loc[0, "kind"] = "XXX"
    assert t2.loc[0, "kind"] != "XXX"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert "ret" in df.columns
    assert len(df) == truth["n_days"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(signal_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(signal_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(signal_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(signal_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_mean_near_base():
    """On the null tape, the daily mean is close to the planted base drift."""
    df, truth = data.synthetic_panel(signal_bps=0.0, seed=284)
    mean_bps = df["ret"].mean() * 1e4
    assert abs(mean_bps - truth["base_bps"]) < 5.0


def test_synthetic_signal_shifts_event_days():
    """A planted drag should pull event-day returns below the base drift."""
    ev = data.equinox_table()
    df0, _ = data.synthetic_panel(signal_bps=0.0, seed=284)
    df1, _ = data.synthetic_panel(signal_bps=-300.0, seed=284)
    idx = df0.index
    e0, e1 = [], []
    for d in ev["date"]:
        pos = idx.searchsorted(d)
        if 0 <= pos < len(idx):
            e0.append(df0["ret"].iloc[pos])
            e1.append(df1["ret"].iloc[pos])
    assert np.mean(e1) < np.mean(e0) - 0.01  # ~300 bps lower


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_panel(signal_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_gspc_daily raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(cache_dir=str(tmp_path), fetch=False)
