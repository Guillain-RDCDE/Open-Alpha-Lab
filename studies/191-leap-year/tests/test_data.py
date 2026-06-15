"""The synthetic tape is well-formed and deterministic; the Shiller loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap_year import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — any test that reads the shared Shiller parquet must use this.
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.SHILLER_PATH),
    reason="Shiller cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"]
    assert set(["ret", "is_leap", "is_election", "term_year"]).issubset(df.columns)
    assert df.index.name == "year"


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=50, leap_premium=0.05, seed=7)
    b, _ = data.synthetic_annual(n_years=50, leap_premium=0.05, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_annual(n_years=50, leap_premium=0.05, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_leap_year_logic():
    """Standard Gregorian leap-year rules are applied correctly."""
    # Classic examples: 2000 leap, 1900 not leap, 1996 leap, 2001 not leap.
    df, _ = data.synthetic_annual(n_years=200, start_year=1900, seed=0)
    assert bool(df.loc[2000, "is_leap"]) is True
    assert bool(df.loc[1900, "is_leap"]) is False
    assert bool(df.loc[1996, "is_leap"]) is True
    assert bool(df.loc[2001, "is_leap"]) is False


def test_election_year_logic():
    """Election years are always divisible by 4 in the modern era."""
    df, _ = data.synthetic_annual(n_years=20, start_year=1928, seed=0)
    for year in df.index:
        assert bool(df.loc[year, "is_election"]) == (year % 4 == 0)


def test_term_year_values():
    """Year-of-term cycles 1-2-3-4 correctly."""
    df, _ = data.synthetic_annual(n_years=8, start_year=1928, seed=0)
    # 1928 % 4 == 0 -> term_year 1
    assert int(df.loc[1928, "term_year"]) == 1
    assert int(df.loc[1929, "term_year"]) == 2
    assert int(df.loc[1930, "term_year"]) == 3
    assert int(df.loc[1931, "term_year"]) == 4
    assert int(df.loc[1932, "term_year"]) == 1


def test_leap_premium_shifts_mean(null_tape, signal_tape):
    """Planted leap premium shifts the mean return of leap years upward."""
    df_null, _ = null_tape
    df_sig, _ = signal_tape
    null_leap_mean = df_null.loc[df_null["is_leap"], "ret"].mean()
    sig_leap_mean = df_sig.loc[df_sig["is_leap"], "ret"].mean()
    assert sig_leap_mean > null_leap_mean


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_annual(n_years=100, leap_premium=0.0, seed=999)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_load_shiller_raises_without_cache(tmp_path):
    """load_shiller raises FileNotFoundError when the parquet is absent."""
    with pytest.raises(FileNotFoundError):
        data.load_shiller(path=str(tmp_path / "nonexistent.parquet"))


# ---------------------------------------------------------------------------
# Real tape (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_shiller_columns_and_range():
    df = data.load_shiller(start_year=1872)
    assert "ret" in df.columns
    assert "is_leap" in df.columns
    assert "term_year" in df.columns
    assert df.index.min() <= 1929  # covers the modern GSPC era
    assert df["ret"].notna().sum() >= 100  # plenty of data


@requires_cache
def test_shiller_modern_leap_count():
    """There should be ~24-25 leap years in the modern GSPC era (1928-2025)."""
    df = data.load_shiller(start_year=1928)
    n_leap = int(df["is_leap"].sum())
    assert 20 <= n_leap <= 30, f"Expected ~25 leap years, got {n_leap}"


@requires_cache
def test_shiller_fingerprint_stable():
    a = data.load_shiller()
    b = data.load_shiller()
    assert data.fingerprint(a) == data.fingerprint(b)
