"""Tests for the data layer — synthetic generator and Shiller loader."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from real_rate_regime import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_world):
    df, truth = null_world
    assert len(df) == truth["n_months"]
    assert set(["real_rate", "real_rate_chg", "fwd12_real"]).issubset(df.columns)


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(n_months=200, regime_effect=0.0, seed=7)
    b, _ = data.synthetic_world(n_months=200, regime_effect=0.0, seed=7)
    assert np.allclose(a["real_rate"].to_numpy(), b["real_rate"].to_numpy())
    c, _ = data.synthetic_world(n_months=200, regime_effect=0.0, seed=8)
    assert not np.allclose(a["real_rate"].to_numpy(), c["real_rate"].to_numpy())


def test_synthetic_seed_changes_result():
    a, _ = data.synthetic_world(n_months=100, seed=1)
    b, _ = data.synthetic_world(n_months=100, seed=2)
    assert not np.allclose(a["real_rate"].dropna().to_numpy(), b["real_rate"].dropna().to_numpy())


def test_synthetic_real_rate_chg_first_12_are_nan(null_world):
    df, _ = null_world
    assert df["real_rate_chg"].iloc[:12].isna().all()
    assert not df["real_rate_chg"].iloc[12:].isna().all()


def test_synthetic_null_has_no_regime_effect(null_world):
    """With regime_effect=0, real_rate_chg and fwd12_real should be near-uncorrelated."""
    df, _ = null_world
    clean = df.dropna()
    corr = np.corrcoef(clean["real_rate_chg"], clean["fwd12_real"])[0, 1]
    assert abs(corr) < 0.2  # a loose bound; pure noise gives ~0


def test_synthetic_effect_world_has_correlation(effect_world):
    """With a planted negative regime_effect, rising rates should lower equity returns."""
    df, truth = effect_world
    assert truth["regime_effect"] < 0
    clean = df.dropna()
    corr = np.corrcoef(clean["real_rate_chg"], clean["fwd12_real"])[0, 1]
    assert corr < -0.1  # the planted effect must create a negative correlation


def test_fingerprint_stable_and_content_sensitive(null_world):
    df, _ = null_world
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_world(n_months=600, seed=42)
    assert data.fingerprint(df) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Shiller loader — only tests that don't require the real file to be present
# ---------------------------------------------------------------------------
def test_fetch_shiller_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_shiller(cache_dir=str(tmp_path))
