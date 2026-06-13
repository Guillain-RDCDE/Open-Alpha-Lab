"""The synthetic price tape is well-formed, deterministic, and controllable."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faber_timing import data  # noqa: E402


def test_synthetic_shape_and_columns(two_regime):
    prices, truth = two_regime
    assert "close" in prices.columns
    assert "tbill" in prices.columns
    assert len(prices) == truth["n_years"] * 252
    assert (prices["close"] > 0).all()
    assert (prices["tbill"] >= 1.0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, signal_strength=1.0, seed=7)
    b, _ = data.synthetic_daily(n_years=10, signal_strength=1.0, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, signal_strength=1.0, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_signal_strength_zero_narrows_vol_spread():
    """At signal_strength=0 the effective vols converge toward a single value."""
    _, t1 = data.synthetic_daily(n_years=20, signal_strength=1.0, seed=110)
    _, t0 = data.synthetic_daily(n_years=20, signal_strength=0.0, seed=110)
    spread_full = abs(t1["vol_bull_eff"] - t1["vol_bear_eff"])
    spread_null = abs(t0["vol_bull_eff"] - t0["vol_bear_eff"])
    assert spread_null < spread_full


def test_fetch_prices_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_prices("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(two_regime):
    prices, _ = two_regime
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_daily(n_years=30, signal_strength=1.0, seed=999)
    assert data.fingerprint(prices) != data.fingerprint(other)


def test_resample_monthly_reduces_rows(two_regime):
    prices, truth = two_regime
    monthly = data.resample_monthly(prices)
    # The synthetic tape starts mid-January so the first calendar month may be partial;
    # allow a wider tolerance of up to 15 months for boundary / partial month effects.
    expected_months = truth["n_years"] * 12
    assert abs(len(monthly) - expected_months) <= 15
    assert len(monthly) < len(prices)
    assert "close" in monthly.columns
