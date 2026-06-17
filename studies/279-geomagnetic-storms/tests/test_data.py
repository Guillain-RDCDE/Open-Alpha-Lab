"""Tests for the data layer of Study 279 (Geomagnetic-Storms).

All tests are offline and deterministic — they use the hardcoded Ap index and
the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomagnetic_storms import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded geomagnetic Ap index
# ---------------------------------------------------------------------------
def test_kp_index_columns(kp_index):
    assert "ap" in kp_index.columns


def test_kp_index_covers_range(kp_index):
    assert kp_index.index.year.min() == 1932
    assert kp_index.index.year.max() == 2025


def test_kp_index_is_monthly(kp_index):
    """~12 observations per year across ~94 years."""
    n_years = 2025 - 1932 + 1
    assert abs(len(kp_index) - 12 * n_years) <= 1


def test_kp_index_positive(kp_index):
    """Geomagnetic activity is strictly positive."""
    assert (kp_index["ap"] > 0).all()


def test_kp_index_deterministic():
    """The Ap index is a pure function of the calendar — byte-identical on re-call."""
    a = data.kp_monthly_index()
    b = data.kp_monthly_index()
    assert np.array_equal(a["ap"].to_numpy(), b["ap"].to_numpy())


def test_kp_index_has_solar_cycle_variation(kp_index):
    """The series oscillates with the solar cycle — meaningful spread."""
    ap = kp_index["ap"]
    assert ap.max() > 2 * ap.min()
    assert ap.std() > 3.0


def test_kp_index_known_storm_spike(kp_index):
    """March 1989 (the Quebec blackout storm) is among the most active months."""
    mar1989 = kp_index[(kp_index.index.year == 1989) & (kp_index.index.month == 3)]["ap"].iloc[0]
    assert mar1989 >= kp_index["ap"].quantile(0.90)


def test_kp_index_is_copy():
    """kp_monthly_index() returns a copy — mutating it does not affect the source."""
    a = data.kp_monthly_index()
    b = data.kp_monthly_index()
    a.iloc[0, 0] = -999.0
    assert b.iloc[0, 0] != -999.0


# ---------------------------------------------------------------------------
# Storm classification
# ---------------------------------------------------------------------------
def test_storm_classification_labels(kp_index):
    reg = data.storm_classification(kp_index["ap"], hi_q=0.80, lo_q=0.20)
    assert set(reg.unique()).issubset({"stormy", "calm", "normal"})


def test_storm_classification_proportions(kp_index):
    """Top/bottom quintiles give ~20% stormy and ~20% calm."""
    reg = data.storm_classification(kp_index["ap"], hi_q=0.80, lo_q=0.20)
    frac_stormy = (reg == "stormy").mean()
    frac_calm = (reg == "calm").mean()
    assert 0.12 <= frac_stormy <= 0.30
    assert 0.12 <= frac_calm <= 0.30


def test_storm_classification_ordering(kp_index):
    """Stormy months have higher Ap than calm months by construction."""
    ap = kp_index["ap"]
    reg = data.storm_classification(ap, hi_q=0.80, lo_q=0.20)
    assert ap[reg == "stormy"].min() >= ap[reg == "calm"].max()


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_months"]
    assert set(df.columns) >= {"ret", "ap", "regime"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=42)
    b, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=1)
    b, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_gap():
    """On the null tape, calm and stormy mean returns are close (no planted effect)."""
    df, _ = data.synthetic_monthly(n_months=3000, storm_bps=0.0, seed=279)
    calm = df.loc[df["regime"] == "calm", "ret"].mean()
    stormy = df.loc[df["regime"] == "stormy", "ret"].mean()
    # Null: gap is sampling noise, far below the planted-signal gaps (>= 0.01).
    assert abs(calm - stormy) < 0.006


def test_synthetic_signal_creates_gap():
    """A planted penalty makes stormy-month returns lower than calm-month returns."""
    df, _ = data.synthetic_monthly(n_months=2000, storm_bps=300.0, seed=279)
    calm = df.loc[df["regime"] == "calm", "ret"].mean()
    stormy = df.loc[df["regime"] == "stormy", "ret"].mean()
    assert calm > stormy + 0.01


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=10)
    df2, _ = data.synthetic_monthly(n_months=120, storm_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real-data loader is cache-only / offline by default
# ---------------------------------------------------------------------------
def test_fetch_raises_without_cache(tmp_path):
    """Without the ^GSPC cache, fetch_gspc_monthly raises FileNotFoundError (no network)."""
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_monthly(fetch=False, cache_path=missing)
