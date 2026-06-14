"""The synthetic panel is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net_operating_assets import data  # noqa: E402


def test_synthetic_panel_shape(null_panel):
    sig, fwd, truth = null_panel
    assert sig.shape == fwd.shape
    assert sig.shape[0] == 17   # n_years
    assert sig.shape[1] == 200  # n_firms


def test_synthetic_panel_index(null_panel):
    sig, fwd, truth = null_panel
    assert list(sig.index) == list(range(2008, 2025))
    assert sig.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_panel_is_deterministic():
    a_sig, a_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=-0.05, seed=5)
    b_sig, b_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=-0.05, seed=5)
    assert np.allclose(a_sig.to_numpy(), b_sig.to_numpy())
    assert np.allclose(a_fwd.to_numpy(), b_fwd.to_numpy())
    # Different seed → different data
    c_sig, _, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=-0.05, seed=6)
    assert not np.allclose(a_sig.to_numpy(), c_sig.to_numpy())


def test_synthetic_null_truth(null_panel):
    _, _, truth = null_panel
    assert truth.premium == 0.0
    assert not truth.has_premium


def test_synthetic_live_truth(live_panel):
    _, _, truth = live_panel
    assert truth.premium == -0.08
    assert truth.has_premium


def test_premium_creates_noa_spread():
    """A negative premium plants a real high-NOA underperformance."""
    from net_operating_assets import strategy as st
    sig_null, fwd_null, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=0.0, seed=153)
    sig_live, fwd_live, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=-0.12, seed=153)
    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)
    # With planted premium: low-NOA minus high-NOA hedge should be positive
    assert h_live["hedge"].mean() > h_null["hedge"].mean()


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    sig, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert sig.empty
    assert fwd.empty


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    sig, _, _ = null_panel
    assert data.fingerprint(sig) == data.fingerprint(sig)
    other, _, _ = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=99)
    assert data.fingerprint(sig) != data.fingerprint(other)
