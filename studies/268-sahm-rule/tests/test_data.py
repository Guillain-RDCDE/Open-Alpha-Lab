"""Tests for the data layer of Study 268 (Sahm-Rule).

All tests are offline and deterministic -- they use the hardcoded unemployment
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahm_rule import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded unemployment table
# ---------------------------------------------------------------------------
def test_unrate_table_columns():
    df = data.unrate_table()
    assert {"date", "unrate"}.issubset(set(df.columns))


def test_unrate_table_spans_expected_range():
    df = data.unrate_table()
    assert df["date"].min() == pd.Timestamp("1959-01-01")
    assert df["date"].max() == pd.Timestamp("2025-12-01")


def test_unrate_is_monthly_and_dense():
    """1959-01 .. 2025-12 inclusive = 67 years * 12 = 804 months, no gaps."""
    s = data.unrate_series()
    assert len(s) == 804
    # Monthly spacing: every consecutive gap is between 28 and 31 days.
    gaps = s.index.to_series().diff().dropna().dt.days
    assert gaps.min() >= 28 and gaps.max() <= 31


def test_unrate_values_are_plausible():
    s = data.unrate_series()
    # Postwar US unemployment lives in [2.5%, 15%] (the 2020 COVID spike is ~14.8).
    assert s.min() >= 2.5
    assert s.max() <= 15.0


def test_unrate_known_landmarks():
    """Spot-check a few well-known prints."""
    s = data.unrate_series()
    # COVID spike: April 2020 = 14.8 (record high in the series)
    assert s.loc[pd.Timestamp("2020-04-01")] == pytest.approx(14.8, abs=0.05)
    # Volcker recession trough of joblessness: Nov/Dec 1982 ~ 10.8
    assert s.loc[pd.Timestamp("1982-12-01")] == pytest.approx(10.8, abs=0.05)
    # Late-2019 low ~ 3.5-3.6
    assert s.loc[pd.Timestamp("2019-09-01")] == pytest.approx(3.5, abs=0.1)


def test_unrate_series_is_copy():
    a = data.unrate_table()
    b = data.unrate_table()
    a.loc[0, "unrate"] = -99.0
    assert b.loc[0, "unrate"] != -99.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    s, truth = synthetic_null
    assert len(s) == truth["n_months"]
    assert isinstance(s, pd.Series)


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_unrate(shock_pp=0.0, seed=7)
    b, _ = data.synthetic_unrate(shock_pp=0.0, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_unrate(shock_pp=0.0, seed=1)
    b, _ = data.synthetic_unrate(shock_pp=0.0, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shock_raises_level():
    """A planted shock should raise the post-shock level meaningfully."""
    s, truth = data.synthetic_unrate(shock_pp=3.0, shock_at=180, shock_len=9, seed=268)
    pre = s.iloc[170:180].mean()
    post = s.iloc[189:199].mean()
    assert post > pre + 1.0


def test_synthetic_null_stays_bounded():
    s, _ = data.synthetic_unrate(shock_pp=0.0, seed=268)
    assert s.min() >= 2.0 and s.max() <= 20.0


# ---------------------------------------------------------------------------
# Real-data loader: cache-only contract
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_gspc_daily raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_unrate(shock_pp=0.0, seed=10)
    b, _ = data.synthetic_unrate(shock_pp=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)
