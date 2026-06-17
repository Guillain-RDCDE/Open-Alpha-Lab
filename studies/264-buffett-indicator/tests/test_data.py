"""Tests for the data layer of Study 264 (Buffett-Indicator).

All tests are offline and deterministic -- they use the hardcoded table and the
synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffett_indicator import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Buffett-Indicator table
# ---------------------------------------------------------------------------
def test_table_columns(bi_table):
    assert {"year", "ratio"}.issubset(set(bi_table.columns))


def test_table_year_range(bi_table):
    assert bi_table["year"].min() == 1971
    assert bi_table["year"].max() == 2025


def test_table_years_sequential(bi_table):
    """Years run 1971..2025 with no gaps or duplicates."""
    assert list(bi_table["year"]) == list(range(1971, 2026))


def test_table_ratios_are_plausible(bi_table):
    """Buffett Indicator levels live roughly between 30% and 250% of GDP."""
    assert bi_table["ratio"].min() >= 30.0
    assert bi_table["ratio"].max() <= 250.0


def test_table_dotcom_and_2021_peaks(bi_table):
    """Sanity spot-checks: 1999 dot-com peak high, 1974 trough low, 2021 record high."""
    r = bi_table.set_index("year")["ratio"]
    assert r[1974] < 60.0           # post oil-shock trough
    assert r[1999] > 150.0          # dot-com peak
    assert r[2021] > 190.0          # post-COVID record
    assert r[2021] > r[1999]        # 2021 eclipsed the dot-com peak


def test_table_is_copy(bi_table):
    """buffett_table() returns a copy -- mutating it does not affect the source."""
    t1 = data.buffett_table()
    t2 = data.buffett_table()
    t1.loc[0, "ratio"] = -999.0
    assert t2.loc[0, "ratio"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "ratio", "fwd_ret"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=7)
    b, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=7)
    assert np.allclose(a["fwd_ret"].to_numpy(), b["fwd_ret"].to_numpy())
    assert np.allclose(a["ratio"].to_numpy(), b["ratio"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=1)
    b, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=2)
    assert not np.allclose(a["fwd_ret"].to_numpy(), b["fwd_ret"].to_numpy())


def test_synthetic_null_has_no_relationship():
    """On the null tape, ratio and fwd_ret are essentially uncorrelated."""
    df, _ = data.synthetic_panel(n_years=2000, beta=0.0, seed=264)
    corr = np.corrcoef(df["ratio"], df["fwd_ret"])[0, 1]
    assert abs(corr) < 0.10


def test_synthetic_signal_has_negative_relationship():
    """A strong planted negative beta produces a clearly negative correlation."""
    df, _ = data.synthetic_panel(n_years=2000, beta=-0.40, seed=264)
    corr = np.corrcoef(df["ratio"], df["fwd_ret"])[0, 1]
    assert corr < -0.3


def test_synthetic_valuation_is_mean_reverting(synthetic_signal):
    """The AR(1) valuation level stays bounded and varies around its mean."""
    df, truth = synthetic_signal
    assert df["ratio"].min() >= 20.0
    assert df["ratio"].max() <= 320.0
    assert df["ratio"].std() > 5.0


# ---------------------------------------------------------------------------
# Fingerprint + cache guard
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=10)
    df2, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_sp500_annual_raises_without_cache(tmp_path):
    """With no cache and fetch=False, the loader raises FileNotFoundError."""
    missing = str(tmp_path / "no_such.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(fetch=False, cache_path=missing)
