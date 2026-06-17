"""Tests for the data layer of Study 272 (Champagne-Index).

All tests are offline and deterministic — they use the hardcoded champagne table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champagne_index import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded champagne table
# ---------------------------------------------------------------------------
def test_table_columns(champ_table):
    expected = {"year", "shipments_mn", "ship_growth"}
    assert expected.issubset(set(champ_table.columns))


def test_table_years_sorted_and_unique(champ_table):
    years = list(champ_table["year"])
    assert years == sorted(years)
    assert len(years) == len(set(years))


def test_table_year_range(champ_table):
    assert champ_table["year"].min() == 1999
    assert champ_table["year"].max() == 2024


def test_table_shipments_positive(champ_table):
    assert (champ_table["shipments_mn"] > 0).all()
    # plausible magnitude: worldwide champagne is hundreds of millions of bottles
    assert champ_table["shipments_mn"].between(200, 400).all()


def test_table_first_growth_is_nan(champ_table):
    """The first row has no prior year, so its YoY growth is NaN."""
    assert np.isnan(champ_table["ship_growth"].iloc[0])
    assert champ_table["ship_growth"].iloc[1:].notna().all()


def test_table_known_landmarks(champ_table):
    """Spot-check the narrative landmarks the indicator trades on."""
    by_year = champ_table.set_index("year")["shipments_mn"]
    # 2007 pre-GFC peak is a local maximum vs its neighbours
    assert by_year[2007] > by_year[2006]
    assert by_year[2007] > by_year[2008]
    # 2020 COVID dip is the global minimum of the sample
    assert by_year.idxmin() == 2020
    # 2021 euphoric rebound jumps sharply off the 2020 low
    assert by_year[2021] > by_year[2020] * 1.2


def test_champagne_table_is_copy():
    """champagne_table() returns a fresh copy each call."""
    t1 = data.champagne_table()
    t2 = data.champagne_table()
    t1.loc[0, "shipments_mn"] = -999.0
    assert t2.loc[0, "shipments_mn"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape():
    df, truth = data.synthetic_annual(n_years=40, signal_bps=0.0, seed=272)
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "shipments_mn", "ship_growth", "sp500_fwd_return"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=42)
    assert np.allclose(a["sp500_fwd_return"].to_numpy(), b["sp500_fwd_return"].to_numpy())
    assert np.allclose(a["ship_growth"].to_numpy(), b["ship_growth"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=2)
    assert not np.allclose(a["sp500_fwd_return"].to_numpy(), b["sp500_fwd_return"].to_numpy())


def test_synthetic_null_has_no_link():
    """On the null tape, champagne growth and forward return are ~uncorrelated."""
    df, _ = data.synthetic_annual(n_years=2000, signal_bps=0.0, seed=272)
    c = np.corrcoef(df["ship_growth"], df["sp500_fwd_return"])[0, 1]
    assert abs(c) < 0.10


def test_synthetic_signal_creates_negative_link():
    """A planted euphoria premium makes growth and forward return negatively correlated."""
    df, _ = data.synthetic_annual(n_years=2000, signal_bps=2_000.0, seed=272)
    c = np.corrcoef(df["ship_growth"], df["sp500_fwd_return"])[0, 1]
    assert c < -0.10


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=11)
    df1 = df1.rename(columns={"sp500_fwd_return": "sp500_return"})
    df2 = df2.rename(columns={"sp500_fwd_return": "sp500_return"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real-data loader: cache-only, no network in tests
# ---------------------------------------------------------------------------
def test_fetch_raises_without_cache(tmp_path):
    """Without the ^GSPC cache and fetch=False, the loader raises FileNotFoundError."""
    fake = str(tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(fetch=False, cache_path=fake)


def test_fetch_bad_convention_raises(tmp_path):
    fake = str(tmp_path / "missing.parquet")
    with pytest.raises(ValueError):
        data.fetch_sp500_annual(fetch=False, cache_path=fake, convention="sideways")
