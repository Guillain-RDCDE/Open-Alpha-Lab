"""Tests for the data layer of Study 266 (Misery-Index).

All tests are offline and deterministic -- they use the hardcoded misery table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misery_index import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded misery table
# ---------------------------------------------------------------------------
def test_table_columns(misery_tbl):
    expected = {"year", "cpi_yoy", "unemp", "misery"}
    assert expected.issubset(set(misery_tbl.columns))


def test_table_year_range(misery_tbl):
    assert misery_tbl["year"].min() == 1948
    assert misery_tbl["year"].max() == 2025


def test_table_years_contiguous(misery_tbl):
    years = list(misery_tbl["year"])
    assert years == list(range(1948, 2026))


def test_misery_is_sum(misery_tbl):
    """misery == cpi_yoy + unemp for every row."""
    assert np.allclose(
        misery_tbl["misery"].to_numpy(),
        (misery_tbl["cpi_yoy"] + misery_tbl["unemp"]).to_numpy(),
    )


def test_unemployment_plausible(misery_tbl):
    """Unemployment rate is a sane percentage in every year."""
    assert misery_tbl["unemp"].min() >= 2.0
    assert misery_tbl["unemp"].max() <= 12.0


def test_known_misery_peak(misery_tbl):
    """The early-1980s stagflation peak is the historic misery high."""
    peak_year = int(misery_tbl.loc[misery_tbl["misery"].idxmax(), "year"])
    assert peak_year in (1980, 1981, 1982)
    assert misery_tbl["misery"].max() > 18.0


def test_known_values(misery_tbl):
    """Spot-check a couple of well-known year-end prints."""
    r1979 = misery_tbl[misery_tbl["year"] == 1979].iloc[0]
    assert abs(r1979["cpi_yoy"] - 13.3) < 1e-6
    r2021 = misery_tbl[misery_tbl["year"] == 2021].iloc[0]
    assert abs(r2021["cpi_yoy"] - 7.0) < 1e-6


def test_table_is_copy():
    """misery_table() returns a copy -- mutating it does not affect the original."""
    t1 = data.misery_table()
    t2 = data.misery_table()
    t1.loc[0, "misery"] = -999.0
    assert t2.loc[0, "misery"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "misery", "fwd_return"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=42)
    b, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=42)
    assert np.allclose(a["fwd_return"].to_numpy(), b["fwd_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=1)
    b, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=2)
    assert not np.allclose(a["fwd_return"].to_numpy(), b["fwd_return"].to_numpy())


def test_synthetic_null_has_no_link():
    """On the null tape, misery and forward return are uncorrelated."""
    df, _ = data.synthetic_panel(n_years=2000, beta=0.0, seed=266)
    corr = np.corrcoef(df["misery"], df["fwd_return"])[0, 1]
    assert abs(corr) < 0.08


def test_synthetic_signal_creates_link():
    """A planted beta creates a positive misery->forward-return correlation."""
    df, _ = data.synthetic_panel(n_years=2000, beta=0.10, seed=266)
    corr = np.corrcoef(df["misery"], df["fwd_return"])[0, 1]
    assert corr > 0.2


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=10)
    df2, _ = data.synthetic_panel(n_years=50, beta=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_sp500_annual_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_sp500_annual raises."""
    fake = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(fetch=False, cache_path=fake)
