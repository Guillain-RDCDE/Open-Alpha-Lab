"""Tests for the data layer of Study 283 (Hurricane-Season).

All tests are offline and deterministic — they use the hardcoded landfall table,
the in-season calendar helper, and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hurricane_season import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded landfall table
# ---------------------------------------------------------------------------
def test_landfall_columns(landfalls):
    expected = {"storm", "season_year", "landfall", "category", "insured_bn"}
    assert expected.issubset(set(landfalls.columns))


def test_landfall_count_reasonable(landfalls):
    """There are several dozen dateable major landfalls in the modern era."""
    assert 20 <= len(landfalls) <= 60


def test_landfall_years_in_range(landfalls):
    assert landfalls["season_year"].min() >= 1992
    assert landfalls["season_year"].max() <= 2024


def test_landfall_dates_in_season(landfalls):
    """Every hardcoded landfall date falls within the Jun 1 - Nov 30 season."""
    mask = data.in_season(pd.DatetimeIndex(landfalls["landfall"]))
    assert bool(mask.all())


def test_landfall_categories_valid(landfalls):
    assert set(landfalls["category"].unique()).issubset({1, 2, 3, 4, 5})


def test_landfall_known_storms(landfalls):
    """Spot-check several well-known storms."""
    storms = set(landfalls["storm"])
    assert {"Andrew", "Katrina", "Sandy", "Harvey", "Ian"}.issubset(storms)
    katrina = landfalls[landfalls["storm"] == "Katrina"].iloc[0]
    assert katrina["landfall"] == pd.Timestamp("2005-08-29")


def test_landfall_table_is_copy():
    t1 = data.landfall_table()
    t2 = data.landfall_table()
    t1.loc[0, "storm"] = "ZZZ"
    assert t2.loc[0, "storm"] != "ZZZ"


# ---------------------------------------------------------------------------
# in_season calendar helper
# ---------------------------------------------------------------------------
def test_in_season_boundaries():
    idx = pd.DatetimeIndex([
        "2020-05-31",  # off (day before)
        "2020-06-01",  # in  (first day)
        "2020-08-15",  # in
        "2020-11-30",  # in  (last day)
        "2020-12-01",  # off (day after)
        "2020-01-15",  # off (winter)
    ])
    m = data.in_season(idx).to_numpy()
    assert list(m) == [False, True, True, True, False, False]


def test_in_season_about_half():
    """Over a multi-year daily span, in-season is roughly half the days."""
    idx = pd.bdate_range("1992-01-01", "2024-12-31")
    frac = data.in_season(idx).mean()
    assert 0.45 <= frac <= 0.55


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert set(df.columns) >= {"ret", "in_season"}
    assert df["in_season"].dtype == bool


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, season_bps=0.0, seed=42)
    b, _ = data.synthetic_daily(n_years=5, season_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(n_years=5, season_bps=0.0, seed=1)
    b, _ = data.synthetic_daily(n_years=5, season_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_season_gap():
    """On the null tape, in-season and off-season means are close."""
    df, _ = data.synthetic_daily(n_years=200, season_bps=0.0, seed=283)
    gap = df.loc[df["in_season"], "ret"].mean() - df.loc[~df["in_season"], "ret"].mean()
    assert abs(gap) < 5e-4  # < 5 bps/day on a large sample


def test_synthetic_signal_creates_drag():
    """A planted negative in-season drag shows up as a negative gap."""
    df, _ = data.synthetic_daily(n_years=200, season_bps=-50.0, seed=283)
    gap = df.loc[df["in_season"], "ret"].mean() - df.loc[~df["in_season"], "ret"].mean()
    assert gap < -20e-4  # at least -20 bps/day


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(n_years=3, season_bps=0.0, seed=10)
    df2, _ = data.synthetic_daily(n_years=3, season_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Cache-only real loaders raise without cache (offline guarantee)
# ---------------------------------------------------------------------------
def test_fetch_market_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_market(cache_dir=str(tmp_path), fetch=False)


def test_fetch_insurers_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_insurers(cache_dir=str(tmp_path), fetch=False)
