"""Tests for the data layer of Study 289 (Diwali-Muhurat).

All tests are offline and deterministic — they use the hardcoded Diwali table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diwali_muhurat import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Diwali table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(diwali_tbl):
    """We have 23 Diwalis (2003–2025)."""
    assert len(diwali_tbl) == 23


def test_table_columns(diwali_tbl):
    assert {"year", "diwali"}.issubset(set(diwali_tbl.columns))


def test_table_years_sequential(diwali_tbl):
    assert list(diwali_tbl["year"]) == list(range(2003, 2026))


def test_table_diwali_is_datetime(diwali_tbl):
    assert pd.api.types.is_datetime64_any_dtype(diwali_tbl["diwali"])


def test_table_diwali_in_oct_or_nov(diwali_tbl):
    """Diwali always falls in October or November in the Gregorian calendar."""
    months = diwali_tbl["diwali"].dt.month.unique()
    assert set(months).issubset({10, 11})


def test_table_known_dates(diwali_tbl):
    """Spot-check a couple of well-known Diwali dates."""
    d2025 = diwali_tbl.loc[diwali_tbl["year"] == 2025, "diwali"].iloc[0]
    assert d2025 == pd.Timestamp("2025-10-21")
    d2017 = diwali_tbl.loc[diwali_tbl["year"] == 2017, "diwali"].iloc[0]
    assert d2017 == pd.Timestamp("2017-10-19")


def test_table_is_copy():
    """diwali_table() returns a copy — mutating it does not affect the original."""
    t1 = data.diwali_table()
    t2 = data.diwali_table()
    t1.loc[0, "year"] = 9999
    assert t2.loc[0, "year"] != 9999


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    px, truth = synthetic_null
    assert "INDA" in px.columns
    assert isinstance(px.index, pd.DatetimeIndex)
    assert len(px) == truth["n_days"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(premium_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(premium_bps=0.0, seed=42)
    assert np.allclose(a["INDA"].to_numpy(), b["INDA"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(premium_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(premium_bps=0.0, seed=2)
    assert not np.allclose(a["INDA"].to_numpy(), b["INDA"].to_numpy())


def test_synthetic_prices_positive(synthetic_null):
    px, _ = synthetic_null
    assert (px["INDA"] > 0).all()


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_panel(premium_bps=0.0, seed=10)
    b, _ = data.synthetic_panel(premium_bps=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_inda_raises_without_cache(tmp_path):
    """Without the INDA cache, fetch_inda(fetch=False) raises FileNotFoundError."""
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_inda(fetch=False, cache_path=missing)
