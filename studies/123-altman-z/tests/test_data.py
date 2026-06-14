"""The synthetic panel is well-formed, deterministic, and tunable."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altman_z import data  # noqa: E402


def test_synthetic_shape(has_premium):
    z, fwd, truth = has_premium
    assert z.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == z.shape


def test_synthetic_index_columns(has_premium):
    z, fwd, truth = has_premium
    assert z.index.name == "year"
    assert fwd.index.name == "year"
    assert list(z.index) == list(fwd.index)
    assert list(z.columns) == list(fwd.columns)


def test_synthetic_z_range(has_premium):
    """Z-scores are in a plausible range (0 to 20) and entirely non-negative."""
    z, fwd, truth = has_premium
    assert (z >= 0.0).all().all()
    assert (z <= 20.0).all().all()


def test_synthetic_is_deterministic():
    z1, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    z2, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    assert np.allclose(z1.values, z2.values)
    z3, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=99)
    assert not np.allclose(z1.values, z3.values)


def test_synthetic_null_knob(no_premium):
    """With z_premium=0 the truth dict correctly flags no premium."""
    _, _, truth = no_premium
    assert truth["z_premium"] == 0.0
    assert not truth["has_premium"]


def test_synthetic_premium_knob(has_premium):
    """With z_premium>0 the truth dict flags a premium."""
    _, _, truth = has_premium
    assert truth["has_premium"]


def test_fingerprint_stable_and_content_sensitive(has_premium):
    z, _, _ = has_premium
    fp1 = data.fingerprint(z)
    fp2 = data.fingerprint(z)
    assert fp1 == fp2  # deterministic
    z2, _, _ = data.synthetic_panel(n_firms=200, n_years=20, seed=999)
    assert fp1 != data.fingerprint(z2)


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """fetch_panel with a non-existent cache directory returns empty frames."""
    z, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert z.empty
    assert fwd.empty
