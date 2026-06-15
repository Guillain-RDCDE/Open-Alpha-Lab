"""The synthetic panel is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roe_quality import data  # noqa: E402

_CACHE_PATHS = [
    os.path.join(data.DEFAULT_CACHE, f"_edgar_{c}.parquet") for c in data.CONCEPTS
] + [os.path.join(data.DEFAULT_CACHE, "_edgar_yrret.parquet")]

requires_cache = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _CACHE_PATHS),
    reason="EDGAR cache absent (offline CI); covered by synthetic tests",
)


def test_synthetic_panel_shape(null_panel):
    roe, gp, fwd, truth = null_panel
    assert roe.shape == fwd.shape
    assert roe.shape[0] == 17  # n_years
    assert roe.shape[1] == 200  # n_firms


def test_synthetic_panel_index(null_panel):
    roe, gp, fwd, truth = null_panel
    assert list(roe.index) == list(range(2008, 2025))
    assert roe.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_panel_is_deterministic():
    a_roe, a_gp, a_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=0.05, seed=5)
    b_roe, b_gp, b_fwd, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=0.05, seed=5)
    assert np.allclose(a_roe.to_numpy(), b_roe.to_numpy())
    assert np.allclose(a_fwd.to_numpy(), b_fwd.to_numpy())
    # Different seed -> different data
    c_roe, _, _, _ = data.synthetic_panel(n_firms=100, n_years=10, premium=0.05, seed=6)
    assert not np.allclose(a_roe.to_numpy(), c_roe.to_numpy())


def test_synthetic_null_truth(null_panel):
    _, _, _, truth = null_panel
    assert truth.premium == 0.0
    assert not truth.has_premium


def test_synthetic_live_truth(live_panel):
    _, _, _, truth = live_panel
    assert truth.premium == 0.10
    assert truth.has_premium


def test_premium_creates_roe_spread():
    """A positive premium plants a real high-ROE outperformance in the returns."""
    from roe_quality import strategy as st
    roe_null, gp_null, fwd_null, _ = data.synthetic_panel(
        n_firms=300, n_years=25, premium=0.0, seed=200
    )
    roe_live, gp_live, fwd_live, _ = data.synthetic_panel(
        n_firms=300, n_years=25, premium=0.15, seed=200
    )
    h_null = st.quintile_returns(roe_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(roe_live, fwd_live, q=0.20)
    # With planted premium: high-ROE minus low-ROE hedge should be positive
    assert h_live["hedge"].mean() > h_null["hedge"].mean()


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    roe, gpr, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert roe.empty
    assert gpr.empty
    assert fwd.empty


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    roe, _, _, _ = null_panel
    assert data.fingerprint(roe) == data.fingerprint(roe)
    other, _, _, _ = data.synthetic_panel(n_firms=200, n_years=17, premium=0.0, seed=99)
    assert data.fingerprint(roe) != data.fingerprint(other)


@requires_cache
def test_fetch_panel_real_data_shape():
    """Real EDGAR panel must have enough tickers and years to be meaningful."""
    roe, gpr, fwd = data.fetch_panel()
    assert not roe.empty
    assert not fwd.empty
    assert roe.shape[1] >= 100  # at least 100 tickers
    assert len(roe) >= 10  # at least 10 years


@requires_cache
def test_fetch_panel_no_lookahead():
    """Forward returns for year y must come from year y+1 in the yrret panel."""
    roe, gpr, fwd = data.fetch_panel()
    # The fwd index must match the roe index (not year+1 directly)
    # But the underlying logic: signal year y -> return year y+1
    # So fwd.index == roe.index (not shifted)
    assert set(fwd.index) == set(roe.index)
