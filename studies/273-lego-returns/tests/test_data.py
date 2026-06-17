"""Tests for the data layer of Study 273 (Lego-Returns). Offline & deterministic."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lego_returns import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded LEGO index
# ---------------------------------------------------------------------------
def test_lego_index_columns(lego_tbl):
    assert {"year", "lego_index", "lego_return"}.issubset(set(lego_tbl.columns))


def test_lego_index_year_range(lego_tbl):
    assert lego_tbl["year"].min() == 1986
    assert lego_tbl["year"].max() == 2024


def test_lego_index_base_100(lego_tbl):
    assert lego_tbl.loc[lego_tbl["year"] == 1986, "lego_index"].iloc[0] == 100.0


def test_lego_index_first_return_nan(lego_tbl):
    assert np.isnan(lego_tbl["lego_return"].iloc[0])


def test_lego_index_levels_positive(lego_tbl):
    assert (lego_tbl["lego_index"] > 0).all()


def test_lego_index_returns_reasonable(lego_tbl):
    """Annual price-only returns should sit in a sane collectibles band."""
    r = lego_tbl["lego_return"].dropna()
    assert r.min() > -0.30
    assert r.max() < 0.40


def test_lego_index_is_copy():
    a = data.lego_index()
    b = data.lego_index()
    a.loc[0, "lego_index"] = -999.0
    assert b.loc[0, "lego_index"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert {"year", "lego_return", "market_return"}.issubset(set(df.columns))


def test_synthetic_deterministic():
    a, _ = data.synthetic_annual(n_years=40, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_annual(n_years=40, premium_bps=0.0, seed=42)
    assert np.allclose(a["lego_return"], b["lego_return"])
    assert np.allclose(a["market_return"], b["market_return"])


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_annual(n_years=40, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_annual(n_years=40, premium_bps=0.0, seed=2)
    assert not np.allclose(a["lego_return"], b["lego_return"])


def test_synthetic_null_no_premium():
    """On the null tape, Lego and market means are close (large n)."""
    df, _ = data.synthetic_annual(n_years=4000, premium_bps=0.0, seed=273)
    gap = df["lego_return"].mean() - df["market_return"].mean()
    assert abs(gap) < 0.02


def test_synthetic_signal_creates_premium():
    """A planted premium makes Lego mean exceed the market mean."""
    df, _ = data.synthetic_annual(n_years=2000, premium_bps=2_000.0, seed=273)
    gap = df["lego_return"].mean() - df["market_return"].mean()
    assert gap > 0.10


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_annual(n_years=30, premium_bps=0.0, seed=10)
    b, _ = data.synthetic_annual(n_years=30, premium_bps=0.0, seed=11)
    assert data.fingerprint(a, "market_return") == data.fingerprint(a, "market_return")
    assert data.fingerprint(a, "market_return") != data.fingerprint(b, "market_return")


def test_fetch_market_annual_raises_without_cache(tmp_path):
    """Without the S&P cache, fetch_market_annual raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_market_annual(cache_dir=str(tmp_path))
