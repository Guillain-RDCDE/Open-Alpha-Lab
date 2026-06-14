"""The synthetic panel is well-formed, deterministic, and has the right shape."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amihud_illiquidity import data  # noqa: E402


def test_synthetic_shape(null_panel):
    illiq, fwd_ret, truth = null_panel
    assert illiq.shape == (truth["n_years"], truth["n_firms"])
    assert fwd_ret.shape == (truth["n_years"], truth["n_firms"])
    assert illiq.index.name == "year"
    assert fwd_ret.index.name == "year"


def test_synthetic_illiq_positive(null_panel):
    """ILLIQ values should be strictly positive (they are |r|/DolVol, always > 0)."""
    illiq, _, _ = null_panel
    assert (illiq.values > 0).all()


def test_synthetic_is_deterministic():
    a_i, a_f, _ = data.synthetic_panel(n_firms=50, n_years=10, premium=0.05, seed=42)
    b_i, b_f, _ = data.synthetic_panel(n_firms=50, n_years=10, premium=0.05, seed=42)
    assert np.allclose(a_i.values, b_i.values)
    assert np.allclose(a_f.values, b_f.values)
    # Different seed -> different values
    c_i, _, _ = data.synthetic_panel(n_firms=50, n_years=10, premium=0.05, seed=99)
    assert not np.allclose(a_i.values, c_i.values)


def test_synthetic_premium_flag(null_panel, premium_panel):
    _, _, truth_null = null_panel
    _, _, truth_prem = premium_panel
    assert not truth_null["has_premium"]
    assert truth_prem["has_premium"]
    assert truth_prem["premium"] > 0.0


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """Without a cache, fetch_panel returns two empty DataFrames (offline-safe)."""
    illiq, fwd_ret = data.fetch_panel(fetch=False, study_cache_dir=str(tmp_path))
    assert illiq.empty
    assert fwd_ret.empty


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    illiq, _, _ = null_panel
    assert data.fingerprint(illiq) == data.fingerprint(illiq)
    illiq2, _, _ = data.synthetic_panel(n_firms=200, n_years=20, premium=0.0, seed=99)
    assert data.fingerprint(illiq) != data.fingerprint(illiq2)
