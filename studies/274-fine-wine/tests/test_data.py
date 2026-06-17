"""Tests for the data layer of Study 274 (Fine-Wine).

All tests are offline and deterministic — they use the hardcoded Liv-ex 100
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fine_wine import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Liv-ex 100 table
# ---------------------------------------------------------------------------
def test_table_columns(wine_table):
    assert {"year", "level"}.issubset(set(wine_table.columns))


def test_table_year_range(wine_table):
    assert wine_table["year"].min() == 2003
    assert wine_table["year"].max() == 2025


def test_table_years_sequential(wine_table):
    yrs = list(wine_table["year"])
    assert yrs == list(range(2003, 2026))


def test_table_levels_positive(wine_table):
    assert (wine_table["level"] > 0).all()


def test_table_levels_are_float(wine_table):
    assert wine_table["level"].dtype == float


def test_table_is_copy():
    t1 = data.livex_table()
    t2 = data.livex_table()
    t1.loc[0, "level"] = -999.0
    assert t2.loc[0, "level"] != -999.0


def test_annual_returns_shape():
    r = data.livex_annual_returns()
    # 23 levels -> 22 returns
    assert len(r) == 22
    assert {"year", "wine_return"}.issubset(set(r.columns))


def test_annual_returns_match_levels():
    t = data.livex_table().sort_values("year").reset_index(drop=True)
    r = data.livex_annual_returns()
    expected = t["level"].iloc[1] / t["level"].iloc[0] - 1
    got = r.loc[r["year"] == t["year"].iloc[1], "wine_return"].iloc[0]
    assert abs(expected - got) < 1e-9


def test_annual_returns_have_drawdowns():
    """The 2008 GFC and the 2011 / 2023-2024 fine-wine bears are negative years."""
    r = data.livex_annual_returns().set_index("year")["wine_return"]
    assert r.loc[2008] < 0
    assert r.loc[2011] < 0
    assert r.loc[2023] < 0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_uncorrelated):
    df, truth = synthetic_uncorrelated
    assert len(df) == truth["n_years"]
    assert {"year", "wine_return", "wine_true", "eq_return"}.issubset(set(df.columns))


def test_synthetic_deterministic():
    a, _ = data.synthetic_annual(n_years=50, corr=0.3, smooth=0.2, seed=7)
    b, _ = data.synthetic_annual(n_years=50, corr=0.3, smooth=0.2, seed=7)
    assert np.allclose(a["wine_return"].to_numpy(), b["wine_return"].to_numpy())
    assert np.allclose(a["eq_return"].to_numpy(), b["eq_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_annual(n_years=50, seed=1)
    b, _ = data.synthetic_annual(n_years=50, seed=2)
    assert not np.allclose(a["wine_return"].to_numpy(), b["wine_return"].to_numpy())


def test_synthetic_null_low_true_correlation():
    """corr=0 → the underlying (unsmoothed) wine factor is ~uncorrelated with equity."""
    df, _ = data.synthetic_annual(n_years=2000, corr=0.0, smooth=0.0, seed=274)
    c = np.corrcoef(df["wine_true"], df["eq_return"])[0, 1]
    assert abs(c) < 0.08


def test_synthetic_planted_correlation_present_in_truth_series():
    """corr=0.6 → the unsmoothed wine_true is strongly correlated with equity."""
    df, _ = data.synthetic_annual(n_years=2000, corr=0.6, smooth=0.0, seed=274)
    c = np.corrcoef(df["wine_true"], df["eq_return"])[0, 1]
    assert c > 0.45


def test_synthetic_smoothing_lowers_observed_correlation():
    """Smoothing depresses the OBSERVED correlation below the true correlation."""
    df, _ = data.synthetic_annual(n_years=3000, corr=0.6, smooth=0.6, seed=274)
    true_c = np.corrcoef(df["wine_true"], df["eq_return"])[0, 1]
    obs_c = np.corrcoef(df["wine_return"], df["eq_return"])[0, 1]
    assert obs_c < true_c


def test_synthetic_smoothing_creates_autocorrelation():
    """Smoothing plants positive first-order autocorrelation in the observed series."""
    df, _ = data.synthetic_annual(n_years=3000, corr=0.0, smooth=0.6, seed=274)
    w = df["wine_return"].to_numpy()
    ac = np.corrcoef(w[:-1], w[1:])[0, 1]
    assert ac > 0.2


# ---------------------------------------------------------------------------
# Fingerprint and cache guard
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_annual(n_years=30, corr=0.0, seed=10)
    df2, _ = data.synthetic_annual(n_years=30, corr=0.0, seed=11)
    df1 = df1.rename(columns={"eq_return": "sp500_return"})
    df2 = df2.rename(columns={"eq_return": "sp500_return"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_sp500_annual_raises_without_cache(tmp_path):
    """With no cache and fetch=False, fetch_sp500_annual raises FileNotFoundError."""
    fake = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(fetch=False, cache_path=fake)
