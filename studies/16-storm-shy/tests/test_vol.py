"""Volatility is forecastable on the clustered tape and not on the flat one — the single fact the
whole overlay stands on."""

import numpy as np

from storm_shy import vol


def test_returns_roundtrip(regime):
    close, _ = regime
    r = vol.to_returns(close)
    assert len(r) == len(close) - 1
    assert r.notna().all()


def test_estimators_are_positive(regime_returns):
    rv = vol.realized_vol(regime_returns, window=21).dropna()
    ew = vol.ewma_vol(regime_returns).dropna()
    assert (rv > 0).all()
    assert (ew.iloc[1:] > 0).all()                  # first EWMA point is an undefined 0 (one obs)


def test_variance_is_forecastable_with_regime(regime_returns):
    """Persistent regime -> strongly autocorrelated variance: high AR(1) rho and lag-1 autocorr."""
    f = vol.forecastability(regime_returns, horizon=21)
    assert f["rho"] > 0.25
    assert f["autocorr_lag1"] > 0.25
    assert f["vol_of_vol"] > 0.6                     # well above the flat sampling floor (~0.32)


def test_variance_is_not_forecastable_when_flat(flat_returns):
    """No regime -> nothing to forecast: AR(1) persistence collapses to ~0. (vol_of_vol can't reach
    zero — block variances always carry chi-square sampling noise — so persistence is the tell.)"""
    f = vol.forecastability(flat_returns, horizon=21)
    assert abs(f["rho"]) < 0.2
    assert abs(f["autocorr_lag1"]) < 0.2
