"""Tests for the data layer of Study 270 (Underwear-Index).

All tests are offline and deterministic — they use the hardcoded underwear table,
the NBER recession calendar, and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from underwear_index import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded underwear table
# ---------------------------------------------------------------------------
def test_table_year_range(uw_table):
    assert uw_table["year"].min() == 1992
    assert uw_table["year"].max() == 2024
    assert len(uw_table) == 33


def test_table_columns(uw_table):
    expected = {"year", "index", "yoy", "dip", "recession"}
    assert expected.issubset(set(uw_table.columns))


def test_table_index_anchored_at_100(uw_table):
    row = uw_table[uw_table["year"] == 1992].iloc[0]
    assert abs(row["index"] - 100.0) < 1e-9


def test_table_index_monotone_except_dips(uw_table):
    """The index falls only in dip years; everywhere else it rises."""
    df = uw_table.dropna(subset=["yoy"])
    assert (df.loc[~df["dip"], "yoy"] > 0).all()
    assert (df.loc[df["dip"], "yoy"] < 0).all()


def test_table_dips_are_a_handful(uw_table):
    """There should be a small number of dips (4) — the tiny-n spine of the study."""
    assert int(uw_table["dip"].sum()) == 4


def test_recession_flag_matches_calendar(uw_table):
    flagged = set(uw_table.loc[uw_table["recession"], "year"])
    assert flagged == data.US_RECESSIONS & set(range(1992, 2025))


def test_recessions_are_known_years():
    assert data.US_RECESSIONS == {2001, 2007, 2008, 2009, 2020}


def test_table_is_copy():
    t1 = data.underwear_table()
    t2 = data.underwear_table()
    t1.loc[0, "index"] = -999.0
    assert t2.loc[0, "index"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "index", "yoy", "dip", "recession"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=50, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_annual(n_years=50, signal_bps=0.0, seed=42)
    assert np.allclose(a["index"].to_numpy(), b["index"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_annual(n_years=50, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_annual(n_years=50, signal_bps=0.0, seed=2)
    assert not np.allclose(a["index"].to_numpy(), b["index"].to_numpy())


def test_synthetic_null_dip_rate_near_base(synthetic_null):
    """On the null tape, dips occur about as often in recessions as anywhere else."""
    df, _ = synthetic_null
    rec_dip = df.loc[df["recession"], "dip"].mean()
    norec_dip = df.loc[~df["recession"], "dip"].mean()
    # No planted link → the gap should be small.
    assert abs(rec_dip - norec_dip) < 0.25


def test_synthetic_signal_creates_dips_in_recessions(synthetic_signal):
    """A strong planted drag should make dips much more common in recession years."""
    df, _ = synthetic_signal
    rec_dip = df.loc[df["recession"], "dip"].mean()
    norec_dip = df.loc[~df["recession"], "dip"].mean()
    assert rec_dip > norec_dip + 0.2


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_annual(n_years=40, signal_bps=0.0, seed=10)
    b, _ = data.synthetic_annual(n_years=40, signal_bps=0.0, seed=11)
    a = a.rename(columns={"index": "sp500_return"})
    b = b.rename(columns={"index": "sp500_return"})
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_sp500_annual_raises_without_cache(tmp_path):
    """Without the Shiller cache, fetch_sp500_annual(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(cache_dir=str(tmp_path), fetch=False)
