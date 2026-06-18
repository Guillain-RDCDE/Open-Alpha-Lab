"""Tests for the data layer of Study 285 (St-Patricks-Day).

All tests are offline and deterministic — they use the synthetic generator and
pure date arithmetic only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_patricks_day import data  # noqa: E402


# ---------------------------------------------------------------------------
# Calendar arithmetic
# ---------------------------------------------------------------------------
def test_stpat_trading_dates_on_or_after_17():
    """Every St. Patrick session is March and on or after the 17th."""
    idx = pd.bdate_range("2000-01-03", "2010-12-31")
    ev = data.stpat_trading_dates(idx)
    assert len(ev) == 11  # one per year 2000..2010
    assert (ev.month == 3).all()
    assert (ev.day >= 17).all()


def test_stpat_one_event_per_year():
    """Exactly one St. Patrick session per calendar year present in the index."""
    idx = pd.bdate_range("1990-01-01", "2020-12-31")
    ev = data.stpat_trading_dates(idx)
    assert len(set(ev.year)) == len(ev)


def test_stpat_rolls_forward_over_weekend():
    """When March 17 is a weekend, the event rolls to the next trading day."""
    # 2018-03-17 is a Saturday -> first trading day on/after is Monday 2018-03-19.
    idx = pd.bdate_range("2018-01-01", "2018-12-31")
    ev = data.stpat_trading_dates(idx)
    e2018 = ev[ev.year == 2018][0]
    assert e2018 == pd.Timestamp("2018-03-19")


def test_placebo_is_late_march():
    idx = pd.bdate_range("2000-01-03", "2005-12-31")
    plac = data.first_trading_on_or_after(idx, data.PLACEBO_DAY)
    assert (plac.month == 3).all()
    assert (plac.day >= data.PLACEBO_DAY).all()


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert set(df.columns) >= {"ret", "is_stpat", "is_placebo", "is_march"}
    assert len(df) == truth["n_days"]
    assert truth["n_stpat"] > 0


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=30, stpat_effect=0.0, seed=42)
    b, _ = data.synthetic_daily(n_years=30, stpat_effect=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(n_years=30, stpat_effect=0.0, seed=1)
    b, _ = data.synthetic_daily(n_years=30, stpat_effect=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_bump():
    """On the null tape, the St. Patrick mean is close to the overall mean."""
    df, _ = data.synthetic_daily(n_years=250, stpat_effect=0.0, seed=285)
    sp = df.loc[df["is_stpat"], "ret"].mean()
    other = df.loc[~df["is_stpat"], "ret"].mean()
    assert abs(sp - other) < 0.002  # < 20 bps on a large sample


def test_synthetic_signal_creates_bump():
    """A planted effect makes the St. Patrick mean exceed the rest."""
    df, _ = data.synthetic_daily(n_years=250, stpat_effect=8.0, seed=285)
    sp = df.loc[df["is_stpat"], "ret"].mean()
    other = df.loc[~df["is_stpat"], "ret"].mean()
    assert sp > other


def test_synthetic_masks_disjoint():
    """A day cannot be both the St. Patrick session and the placebo."""
    df, _ = data.synthetic_daily(n_years=50, stpat_effect=0.0, seed=285)
    assert not (df["is_stpat"] & df["is_placebo"]).any()


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(n_years=20, stpat_effect=0.0, seed=10)
    df2, _ = data.synthetic_daily(n_years=20, stpat_effect=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without any cache and fetch=False, fetch_gspc raises FileNotFoundError."""
    # Point both shared and local cache lookups at an empty temp dir.
    orig_shared = data._GSPC_SHARED
    data._GSPC_SHARED = str(tmp_path / "nope.parquet")
    try:
        with pytest.raises(FileNotFoundError):
            data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))
    finally:
        data._GSPC_SHARED = orig_shared
