"""The synthetic panel is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sales_growth import data  # noqa: E402

EDGAR_REVENUES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "_edgar_Revenues.parquet"
)
EDGAR_YRRET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "_edgar_yrret.parquet"
)

requires_cache = pytest.mark.skipif(
    not os.path.exists(EDGAR_REVENUES) or not os.path.exists(EDGAR_YRRET),
    reason="EDGAR cache absent (offline CI); covered by synthetic tests",
)


def test_synthetic_panel_shape(null_panel):
    sig, fwd, truth = null_panel
    assert sig.shape == fwd.shape
    assert sig.shape[0] == 18   # n_years
    assert sig.shape[1] == 200  # n_firms


def test_synthetic_panel_index(null_panel):
    sig, fwd, truth = null_panel
    assert list(sig.index) == list(range(2007, 2025))
    assert sig.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_panel_is_deterministic():
    a_sig, a_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=-0.05, seed=5)
    b_sig, b_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=-0.05, seed=5)
    assert np.allclose(a_sig.to_numpy(), b_sig.to_numpy())
    assert np.allclose(a_fwd.to_numpy(), b_fwd.to_numpy())
    # Different seed -> different data
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


def test_premium_creates_growth_spread():
    """A negative premium plants a real high-growth underperformance."""
    from sales_growth import strategy as st
    sig_null, fwd_null, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=0.0, seed=199)
    sig_live, fwd_live, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=-0.12, seed=199)
    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)
    # With planted premium: low-growth (q1) should clearly beat high-growth (q5)
    assert h_live["hedge"].mean() > h_null["hedge"].mean()


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    sig, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert sig.empty
    assert fwd.empty


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    sig, _, _ = null_panel
    assert data.fingerprint(sig) == data.fingerprint(sig)
    other, _, _ = data.synthetic_panel(n_firms=200, n_years=18, premium=0.0, seed=99)
    assert data.fingerprint(sig) != data.fingerprint(other)


@requires_cache
def test_real_panel_loads_and_is_aligned():
    sig, fwd = data.fetch_panel()
    assert not sig.empty
    assert not fwd.empty
    assert sig.shape[0] == fwd.shape[0], "signal and forward return frames must have same years"
    assert set(sig.index) == set(fwd.index), "index must match between signal and fwd returns"


@requires_cache
def test_real_panel_growth_finite():
    sig, fwd = data.fetch_panel()
    # Each year should have a reasonable number of non-NaN values
    nonnan_per_year = sig.notna().sum(axis=1)
    assert (nonnan_per_year >= 20).all(), "Need at least 20 valid tickers per year"
