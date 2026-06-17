"""The synthetic panel is well-formed, deterministic, and carries the planted signal."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mohanram_g_score import data  # noqa: E402


def test_synthetic_shapes_match(null_panel):
    signal, fwd_ret, truth = null_panel
    assert signal.shape == fwd_ret.shape
    assert signal.index.name == "year"
    assert fwd_ret.index.name == "year"


def test_synthetic_gscore_range(null_panel):
    """G-score is discrete in {0..8}."""
    signal, _, _ = null_panel
    vals = signal.stack().dropna()
    assert (vals >= 0).all()
    assert (vals <= 8).all()


def test_synthetic_is_deterministic():
    a, b_a, _ = data.synthetic_panel(n_firms=100, n_years=10, g_premium=0.05, seed=7)
    c, d_c, _ = data.synthetic_panel(n_firms=100, n_years=10, g_premium=0.05, seed=7)
    assert np.allclose(a.to_numpy(), c.to_numpy())
    e, _, _ = data.synthetic_panel(n_firms=100, n_years=10, g_premium=0.05, seed=8)
    assert not np.allclose(a.to_numpy(), e.to_numpy())


def test_seed_changes_output():
    a, _, _ = data.synthetic_panel(seed=1)
    b, _, _ = data.synthetic_panel(seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_null_truth_flag(null_panel):
    _, _, truth = null_panel
    assert not truth.has_premium
    assert truth.g_premium == 0.0


def test_live_truth_flag(live_panel):
    _, _, truth = live_panel
    assert truth.has_premium


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """Absent cache -> graceful empty DataFrames, not a crash."""
    sig, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert sig.empty
    assert fwd.empty


# Real cache tests -- skip if absent (CI / offline)
_CACHE_PATHS = [
    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                 "_cache", f)
    for f in (
        "_edgar_NetIncomeLoss.parquet",
        "_edgar_Assets.parquet",
        "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet",
        "_edgar_Revenues.parquet",
        "_edgar_yrret.parquet",
    )
]
_CACHE_PRESENT = all(os.path.exists(p) for p in _CACHE_PATHS)


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-tape EDGAR cache absent offline/CI")
def test_fetch_panel_real_shapes():
    sig, fwd = data.fetch_panel()
    assert not sig.empty
    assert not fwd.empty
    assert sig.shape[0] >= 10  # at least 10 years
    assert sig.shape[1] >= 50  # at least 50 tickers


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-tape EDGAR cache absent offline/CI")
def test_fetch_panel_gscore_range():
    sig, _ = data.fetch_panel()
    vals = sig.stack().dropna()
    assert (vals >= 0).all()
    assert (vals <= 8).all()


def test_fingerprint_stable_and_sensitive(null_panel):
    signal, _, _ = null_panel
    fp = data.fingerprint(signal)
    assert fp == data.fingerprint(signal)
    other, _, _ = data.synthetic_panel(n_firms=200, n_years=20, g_premium=0.0, seed=99)
    assert fp != data.fingerprint(other)
