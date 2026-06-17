"""Tests for the data layer of Study 276 (Sneaker-Resale).

All tests are offline and deterministic — they use the hardcoded sneaker index
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sneaker_resale import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded sneaker-resale index
# ---------------------------------------------------------------------------
def test_index_covers_expected_years(sneaker_df):
    """The index spans 2006 through 2024."""
    assert sneaker_df["year"].min() == 2006
    assert sneaker_df["year"].max() == 2024


def test_index_columns(sneaker_df):
    assert {"year", "level", "sneaker_return"}.issubset(set(sneaker_df.columns))


def test_index_base_100(sneaker_df):
    """The index is normalised to 100 in the base year 2006."""
    base = sneaker_df.loc[sneaker_df["year"] == 2006, "level"].iloc[0]
    assert base == 100.0


def test_index_first_return_is_nan(sneaker_df):
    """The first year has no prior level, so its return is NaN."""
    first = sneaker_df.sort_values("year").iloc[0]
    assert np.isnan(first["sneaker_return"])


def test_index_boom_then_bust(sneaker_df):
    """The index peaks in 2021 and falls in 2022-2024 (the documented bust)."""
    lv = sneaker_df.set_index("year")["level"]
    assert lv[2021] == lv.max()  # peak
    assert lv[2024] < lv[2021]   # gave back a chunk
    assert lv[2016] > lv[2015]   # the boom is rising


def test_index_returns_in_sane_range(sneaker_df):
    """Annual returns are bounded (no absurd values)."""
    r = sneaker_df["sneaker_return"].dropna()
    assert r.min() > -0.50
    assert r.max() < 0.50


def test_index_is_copy(sneaker_df):
    """sneaker_index() returns a fresh frame — mutating it is harmless."""
    a = data.sneaker_index()
    b = data.sneaker_index()
    a.loc[0, "level"] = -999.0
    assert b.loc[0, "level"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert {"year", "sneaker_return", "gspc_return"}.issubset(set(df.columns))


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=30, alpha_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(n_years=30, alpha_bps=0.0, seed=42)
    assert np.allclose(a["sneaker_return"].to_numpy(), b["sneaker_return"].to_numpy())
    assert np.allclose(a["gspc_return"].to_numpy(), b["gspc_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_years=30, alpha_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(n_years=30, alpha_bps=0.0, seed=2)
    assert not np.allclose(a["sneaker_return"].to_numpy(), b["sneaker_return"].to_numpy())


def test_synthetic_null_no_alpha():
    """With alpha_bps=0 and large n, sneaker mean ~ gspc mean (no edge)."""
    df, _ = data.synthetic_panel(n_years=2000, alpha_bps=0.0, smooth_rho=0.0, seed=276)
    gap = df["sneaker_return"].mean() - df["gspc_return"].mean()
    assert abs(gap) < 0.02


def test_synthetic_alpha_creates_edge():
    """A large planted alpha makes the sneaker leg beat the S&P leg on average."""
    df, _ = data.synthetic_panel(n_years=2000, alpha_bps=5000.0, smooth_rho=0.0, seed=276)
    assert df["sneaker_return"].mean() > df["gspc_return"].mean() + 0.3


def test_synthetic_smoothing_creates_autocorr():
    """Smoothing (rho>0) raises the first-order autocorrelation of the sneaker leg."""
    raw, _ = data.synthetic_panel(n_years=2000, smooth_rho=0.0, seed=276)
    sm, _ = data.synthetic_panel(n_years=2000, smooth_rho=0.6, seed=276)
    ac_raw = raw["sneaker_return"].autocorr(1)
    ac_sm = sm["sneaker_return"].autocorr(1)
    assert ac_sm > ac_raw + 0.2


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_panel(n_years=30, seed=10)
    b, _ = data.synthetic_panel(n_years=30, seed=11)
    assert data.fingerprint(a, col="gspc_return") == data.fingerprint(a, col="gspc_return")
    assert data.fingerprint(a, col="gspc_return") != data.fingerprint(b, col="gspc_return")


def test_fetch_gspc_annual_raises_without_cache(tmp_path):
    """Without the ^GSPC cache and fetch=False, fetch_gspc_annual raises."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_annual(cache_dir=str(tmp_path), fetch=False)
