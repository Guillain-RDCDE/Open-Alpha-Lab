"""Tests for the data layer of Study 278 (Sunshine-Effect).

All tests are offline and deterministic -- they use the hardcoded NYC cloud
climatology and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sunshine_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Climatology table
# ---------------------------------------------------------------------------
def test_climatology_has_twelve_months(climatology):
    assert len(climatology) == 12
    assert list(climatology["month"]) == list(range(1, 13))


def test_climatology_columns(climatology):
    assert {"month", "mean_octas", "sd_octas"}.issubset(set(climatology.columns))


def test_climatology_octas_in_range(climatology):
    """Mean sky cover must be a valid octas value (0..8)."""
    assert (climatology["mean_octas"] >= 0).all()
    assert (climatology["mean_octas"] <= 8).all()
    assert (climatology["sd_octas"] > 0).all()


def test_climatology_seasonal_shape():
    """NYC is cloudier in winter (Dec) than in late summer (Aug/Sep)."""
    clim = data.NYC_CLOUD_CLIMATOLOGY
    assert clim[12]["mean_octas"] > clim[8]["mean_octas"]
    assert clim[1]["mean_octas"] > clim[9]["mean_octas"]


# ---------------------------------------------------------------------------
# Cloud-cover-for-dates
# ---------------------------------------------------------------------------
def test_cloud_cover_deterministic():
    dates = pd.bdate_range("2010-01-04", periods=300)
    a = data.cloud_cover_for_dates(dates, base_seed=278)
    b = data.cloud_cover_for_dates(dates, base_seed=278)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_cloud_cover_seed_changes_values():
    dates = pd.bdate_range("2010-01-04", periods=300)
    a = data.cloud_cover_for_dates(dates, base_seed=1)
    b = data.cloud_cover_for_dates(dates, base_seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_cloud_cover_clipped_to_octas_range():
    dates = pd.bdate_range("2000-01-03", periods=2520)
    c = data.cloud_cover_for_dates(dates, base_seed=278)
    assert c.min() >= data.OCTAS_MIN
    assert c.max() <= data.OCTAS_MAX


def test_cloud_cover_tracks_climatology_means():
    """Over many years, the per-month average should be near the climatology mean."""
    dates = pd.bdate_range("1990-01-01", periods=8800)
    c = data.cloud_cover_for_dates(dates, base_seed=278)
    df = pd.DataFrame({"octas": c.to_numpy(), "month": [d.month for d in dates]})
    for m in range(1, 13):
        obs = df.loc[df["month"] == m, "octas"].mean()
        exp = data.NYC_CLOUD_CLIMATOLOGY[m]["mean_octas"]
        # Clipping pulls the mean up slightly near the 0 bound; allow a margin.
        assert abs(obs - exp) < 0.4


# ---------------------------------------------------------------------------
# De-seasonalisation
# ---------------------------------------------------------------------------
def test_deseasonalize_zeroes_monthly_means():
    dates = pd.bdate_range("1990-01-01", periods=8800)
    c = data.cloud_cover_for_dates(dates, base_seed=278)
    anom = data.deseasonalize_cloud(c)
    months = np.array([d.month for d in dates])
    for m in range(1, 13):
        assert abs(anom.to_numpy()[months == m].mean()) < 1e-9


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"ret", "cloud_octas", "cloud_anom"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=42)
    b, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=1)
    b, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_cloud_signal():
    """With no planted effect, the cloud->return correlation is ~0."""
    df, _ = data.synthetic_cloud_returns(n_days=10000, effect_bps_per_octa=0.0, seed=278)
    corr = np.corrcoef(df["cloud_anom"], df["ret"])[0, 1]
    assert abs(corr) < 0.05


def test_synthetic_signal_creates_negative_relationship():
    """A planted effect makes cloud negatively related to returns (sunshine helps)."""
    df, _ = data.synthetic_cloud_returns(n_days=10000, effect_bps_per_octa=5.0, seed=278)
    corr = np.corrcoef(df["cloud_anom"], df["ret"])[0, 1]
    assert corr < -0.05


# ---------------------------------------------------------------------------
# Fingerprint + cache guard
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=10)
    b, _ = data.synthetic_cloud_returns(n_days=500, effect_bps_per_octa=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_index_daily_raises_without_cache(tmp_path):
    """Without the ^GSPC cache, fetch_index_daily(fetch=False) raises FileNotFoundError."""
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_index_daily(fetch=False, cache_path=missing)


def test_build_real_panel_raises_without_cache(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.build_real_panel(fetch=False, cache_path=missing)
