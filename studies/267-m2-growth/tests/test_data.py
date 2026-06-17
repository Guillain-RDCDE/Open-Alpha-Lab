"""Tests for the data layer of Study 267 (M2-Growth).

All tests are offline and deterministic — hardcoded M2 series and the synthetic
generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2_growth import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded M2 YoY series
# ---------------------------------------------------------------------------
def test_m2_series_is_monthly_and_named(m2_series):
    assert isinstance(m2_series.index, pd.DatetimeIndex)
    assert m2_series.name == "m2_yoy"
    # Month-end frequency: consecutive gaps are 28-31 days.
    diffs = m2_series.index.to_series().diff().dropna().dt.days
    assert diffs.between(28, 31).all()


def test_m2_series_covers_range(m2_series):
    assert m2_series.index.min().year == 1960
    assert m2_series.index.max().year == 2025


def test_m2_series_no_nans(m2_series):
    assert m2_series.notna().all()


def test_m2_series_hits_covid_peak(m2_series):
    """The COVID money explosion: M2 YoY peaks well above 20% in early 2021."""
    covid = m2_series["2021-01-31":"2021-06-30"]
    assert covid.max() > 20.0


def test_m2_series_goes_negative_in_2023(m2_series):
    """The first sustained negative M2 YoY prints land in 2023."""
    y2023 = m2_series["2023-01-31":"2023-12-31"]
    assert y2023.min() < 0.0


def test_m2_series_1970s_double_digit(m2_series):
    """The 1970s saw double-digit money growth at its peaks."""
    seventies = m2_series["1970-01-31":"1979-12-31"]
    assert seventies.max() > 10.0


def test_m2_anchors_sorted_and_well_formed():
    dates = [(y, m) for (y, m, _) in data.M2_YOY_ANCHORS]
    assert dates == sorted(dates)
    for (y, m, v) in data.M2_YOY_ANCHORS:
        assert 1 <= m <= 12
        assert 1960 <= y <= 2026
        assert -50.0 < v < 60.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_months"]
    assert set(df.columns) == {"m2_yoy", "sp_ret"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=42)
    b, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=42)
    assert np.allclose(a["sp_ret"].to_numpy(), b["sp_ret"].to_numpy())
    assert np.allclose(a["m2_yoy"].to_numpy(), b["m2_yoy"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=1)
    b, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=2)
    assert not np.allclose(a["sp_ret"].to_numpy(), b["sp_ret"].to_numpy())


def test_synthetic_m2_is_persistent(synthetic_null):
    """M2 growth is autocorrelated (AR(1)) — lag-1 autocorr is high."""
    df, _ = synthetic_null
    ac1 = df["m2_yoy"].autocorr(1)
    assert ac1 > 0.8


def test_synthetic_null_has_no_predictive_link():
    """On the null tape, lagged M2 does not move next-month returns."""
    df, _ = data.synthetic_panel(n_months=2000, premium=0.0, seed=267)
    lag_m2 = df["m2_yoy"].shift(1)
    nxt = df["sp_ret"]
    corr = pd.concat([lag_m2, nxt], axis=1).dropna().corr().iloc[0, 1]
    assert abs(corr) < 0.10


def test_synthetic_signal_creates_predictive_link():
    """A planted premium creates a positive lagged-M2 -> next-return correlation."""
    df, _ = data.synthetic_panel(n_months=2000, premium=3.0, seed=267)
    lag_m2 = df["m2_yoy"].shift(1)
    nxt = df["sp_ret"]
    corr = pd.concat([lag_m2, nxt], axis=1).dropna().corr().iloc[0, 1]
    assert corr > 0.1


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=10)
    b, _ = data.synthetic_panel(n_months=120, premium=0.0, seed=11)
    assert data.fingerprint(a["sp_ret"]) == data.fingerprint(a["sp_ret"])
    assert data.fingerprint(a["sp_ret"]) != data.fingerprint(b["sp_ret"])


# ---------------------------------------------------------------------------
# Real loader is cache-only (no network in tests)
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_monthly(fetch=False, cache_path=missing)


def test_build_real_panel_raises_without_cache(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.build_real_panel(fetch=False, cache_path=missing)
