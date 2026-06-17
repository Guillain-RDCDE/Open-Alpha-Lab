"""The synthetic panel is well-formed, deterministic, and tunable."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graham_ncav import data  # noqa: E402


def test_synthetic_shape(has_premium):
    ncav, fwd, truth = has_premium
    assert ncav.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == ncav.shape


def test_synthetic_index_columns(has_premium):
    ncav, fwd, truth = has_premium
    assert ncav.index.name == "year"
    assert fwd.index.name == "year"
    assert list(ncav.index) == list(fwd.index)
    assert list(ncav.columns) == list(fwd.columns)


def test_synthetic_ncav_range(has_premium):
    """NCAV ratios are clipped to a plausible range."""
    ncav, fwd, truth = has_premium
    assert (ncav >= -3.0).all().all()
    assert (ncav <= 3.0).all().all()


def test_synthetic_is_deterministic():
    n1, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    n2, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    assert np.allclose(n1.values, n2.values)
    n3, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=99)
    assert not np.allclose(n1.values, n3.values)


def test_synthetic_null_knob(no_premium):
    """With ncav_premium=0 the truth dict correctly flags no premium."""
    _, _, truth = no_premium
    assert truth["ncav_premium"] == 0.0
    assert not truth["has_premium"]


def test_synthetic_premium_knob(has_premium):
    """With ncav_premium>0 the truth dict flags a premium."""
    _, _, truth = has_premium
    assert truth["has_premium"]


def test_fingerprint_stable_and_content_sensitive(has_premium):
    ncav, _, _ = has_premium
    fp1 = data.fingerprint(ncav)
    fp2 = data.fingerprint(ncav)
    assert fp1 == fp2  # deterministic
    n2, _, _ = data.synthetic_panel(n_firms=200, n_years=20, seed=999)
    assert fp1 != data.fingerprint(n2)


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """fetch_panel with a non-existent cache directory returns empty frames."""
    ncav, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert ncav.empty
    assert fwd.empty
