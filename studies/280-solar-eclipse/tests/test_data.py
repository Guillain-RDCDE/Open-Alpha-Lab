"""Tests for the data layer of Study 280 (Solar-Eclipse).

All tests are offline and deterministic — they use the hardcoded eclipse table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solar_eclipse import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded eclipse table
# ---------------------------------------------------------------------------
def test_table_has_many_rows(ecl_table):
    """We have a sizeable list of central eclipses (1928 onward)."""
    assert len(ecl_table) >= 70


def test_table_columns(ecl_table):
    expected = {"date", "kind", "us_path"}
    assert expected.issubset(set(ecl_table.columns))


def test_table_dates_are_datetime_and_sorted(ecl_table):
    assert pd.api.types.is_datetime64_any_dtype(ecl_table["date"])
    dates = ecl_table["date"].tolist()
    assert dates == sorted(dates)


def test_table_dates_in_range(ecl_table):
    assert ecl_table["date"].min().year >= 1928
    assert ecl_table["date"].max().year <= 2027


def test_table_kinds_valid(ecl_table):
    assert set(ecl_table["kind"].unique()).issubset({"total", "annular", "hybrid"})


def test_table_us_path_is_bool(ecl_table):
    vals = set(ecl_table["us_path"].unique())
    assert vals.issubset({True, False})


def test_table_has_known_eclipses(ecl_table):
    """Spot-check famous US-crossing total eclipses."""
    dates = set(ecl_table["date"].dt.strftime("%Y-%m-%d"))
    # The Great American Eclipse, 2017-08-21 (US path, total).
    assert "2017-08-21" in dates
    row = ecl_table[ecl_table["date"] == pd.Timestamp("2017-08-21")].iloc[0]
    assert row["kind"] == "total"
    assert bool(row["us_path"]) is True
    # 2024-04-08 total, also US path.
    assert "2024-04-08" in dates


def test_table_has_some_us_paths(ecl_table):
    """A handful of central paths crossed the contiguous US."""
    n_us = int(ecl_table["us_path"].sum())
    assert 5 <= n_us <= 25


def test_eclipse_table_is_copy():
    """eclipse_table() returns a copy — mutating it does not affect the original."""
    t1 = data.eclipse_table()
    t2 = data.eclipse_table()
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
    df, truth = data.synthetic_panel(signal_bps=0.0, seed=280)
    mean_bps = df["ret"].mean() * 1e4
    assert abs(mean_bps - truth["base_bps"]) < 5.0


def test_synthetic_signal_shifts_eclipse_days():
    """A planted drag should pull eclipse-day returns below the base drift."""
    ecl = data.eclipse_table()
    df0, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    df1, _ = data.synthetic_panel(signal_bps=-300.0, seed=280)
    idx = df0.index
    ev0, ev1 = [], []
    for d in ecl["date"]:
        pos = idx.searchsorted(d)
        if 0 <= pos < len(idx):
            ev0.append(df0["ret"].iloc[pos])
            ev1.append(df1["ret"].iloc[pos])
    assert np.mean(ev1) < np.mean(ev0) - 0.01  # ~300 bps lower


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_panel(signal_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_gspc_daily raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(cache_dir=str(tmp_path), fetch=False)
